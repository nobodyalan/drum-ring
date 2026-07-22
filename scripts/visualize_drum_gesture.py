"""
Visualize drum gesture recording: ring IMU + Arduino serial + peak alignment.

Shows 4 panels:
  1. Raw accelerometer (xyz)       — original sensor readings
  2. Raw gyroscope (xyz)           — angular velocity
  3. Linear acceleration (xyz)     — gravity-compensated (OnlineLinearAccelerationEstimator)
  4. Magnitude comparison           — |raw_acc|, |linear_acc|, |gyro|
     + Arduino events (red dashed) + aligned peaks (blue solid) + offset annotations

Peak alignment matches the train_local_drum_gestures.py logic exactly:
  For each Arduino serial FALLING/RISING edge, search ±peak_search window
  around the nearest IMU timestamp for the max linear-acceleration norm.

Usage:
  # List available recordings
  python scripts/visualize_drum_gesture.py --list

  # Visualize a specific recording (by stamp substring match)
  python scripts/visualize_drum_gesture.py --stamp 1783340555

  # Visualize by user+label filter
  python scripts/visualize_drum_gesture.py --user wxt --label 1

  # Time range / click range
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --range 5 20
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --clicks 3 12

  # Custom peak search window
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --peak-search 30

  # Save to file
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --save plot.png

  # Keep decrypted files for inspection
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --keep-decrypted
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.drum_gesture_recognizer import OnlineLinearAccelerationEstimator

# ═══════════════════════════════════════════════════════════════════
# Local decryption (handles CRLF/LF stamp variation)
# ═══════════════════════════════════════════════════════════════════

_INDEX_PATH = ROOT / "apps" / "shared" / ".ui_index.json"


def _load_index() -> dict:
    return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))


def _derive_key(seed: str, salt: bytes, rounds: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=rounds,
    )
    return base64.urlsafe_b64encode(kdf.derive(seed.encode("utf-8")))


def _decrypt_file(enc_path: Path) -> tuple[bytes, str]:
    """Decrypt a .enc file, handling both \\n and \\r\\n stamp variants."""
    blob = enc_path.read_bytes()
    idx = _load_index()
    phrase = "".join(chr(a ^ b) for a, b in zip(idx["left"], idx["right"]))
    stamp_b64 = idx["stamp"]
    stamp_raw = base64.b64decode(stamp_b64)  # e.g. b'RGCRYPT1\n'

    # Try exact stamp match first, then CRLF variant
    candidates = [stamp_raw]
    if stamp_raw.endswith(b"\n"):
        candidates.append(stamp_raw[:-1] + b"\r\n")
    elif stamp_raw.endswith(b"\r\n"):
        candidates.append(stamp_raw[:-2] + b"\n")

    for stamp in candidates:
        if blob.startswith(stamp):
            payload_bytes = blob[len(stamp):]
            break
    else:
        raise ValueError(
            f"File {enc_path.name} has unexpected header: {blob[:20]!r}"
        )

    payload = json.loads(payload_bytes.decode("utf-8"))
    salt = base64.b64decode(payload["salt"])
    token = str(payload["blob"]).encode("ascii")
    key = _derive_key(phrase, salt, int(payload["rounds"]))
    try:
        data = Fernet(key).decrypt(token)
    except InvalidToken as exc:
        raise ValueError(f"Decryption failed for {enc_path.name}") from exc
    return data, str(payload["name"])

# ── defaults matching train_local_drum_gestures.py ──
DEFAULT_PEAK_SEARCH_EACH_SIDE = 30       # frames (±150ms @200Hz)
DEFAULT_WINDOW = 15                       # frames
DEFAULT_LEAD_END_OFFSETS = (6, 4, 2, 0)  # lead offsets
DEFAULT_SERIAL_EDGE = "FALLING"
DEFAULT_DATA_DIR = "data"
EXCLUDE_BEFORE_PEAK = 30                  # frames before peak (no negatives)
EXCLUDE_AFTER_PEAK = 24                   # frames after peak (no negatives)
# total exclusion zone per peak = [peak-30, peak+24+14] = [peak-30, peak+38] frames

_TEMP_DIR_HANDLES: list[tempfile.TemporaryDirectory] = []

# ── plot style ──
ACC_COLORS = {"x": "#e67e22", "y": "#9b59b6", "z": "#1abc9c"}
GYRO_COLORS = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db"}
MAG_COLORS = {
    "raw_acc": "#e74c3c",
    "linear_acc": "#2980b9",
    "gyro": "#27ae60",
}
ARDUINO_COLOR = "#e74c3c"
PEAK_COLOR = "#2980b9"
HIT_ZONE_COLOR = "#27ae60"
WINDOW_POS_COLOR = "#c0392b"
WINDOW_POS_ALPHA = 0.12
EXCLUSION_ZONE_COLOR = "#e74c3c"   # red tint — no negatives sampled here
NEG_SAFE_COLOR = "#bdc3c7"         # gray tint — negatives sampled from here


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def parse_recording_name(name: str) -> tuple[str, str, int] | None:
    """Parse stamp, user, label from IMU or serial filename."""
    match = re.search(r"_(\d{10})_([A-Za-z0-9_]+?)_(\d+)(?:\.npy|\.csv)$", name)
    if not match:
        return None
    stamp, user, label_text = match.groups()
    return stamp, user, int(label_text)


def imu_time_seconds(imu: np.ndarray) -> np.ndarray:
    """Extract relative time in seconds from IMU array."""
    if imu.shape[1] > 1 and np.nanmax(imu[:, 1]) > 0:
        t = imu[:, 1]
    else:
        t = imu[:, 0]
    return t.astype(np.float64) - float(t[0])


def estimate_linear_acceleration(imu: np.ndarray) -> np.ndarray:
    """Run OnlineLinearAccelerationEstimator over full recording.

    Returns (N, 6) array: [linear_acc_x, linear_acc_y, linear_acc_z, gyro_x, gyro_y, gyro_z]
    """
    t = imu_time_seconds(imu)
    acc = imu[:, 2:5].astype(np.float32)
    gyro = imu[:, 5:8].astype(np.float32)
    estimator = OnlineLinearAccelerationEstimator()
    rows = np.zeros((imu.shape[0], 6), dtype=np.float32)
    last_t = float(t[0])
    for idx in range(imu.shape[0]):
        current_t = float(t[idx])
        dt = current_t - last_t if idx else 1.0 / 200.0
        linear_acc, _gravity = estimator.update(acc[idx], gyro[idx], dt)
        rows[idx, :3] = linear_acc
        rows[idx, 3:] = gyro[idx]
        last_t = current_t
    return rows


def serial_event_times(serial_path: Path, edge: str) -> np.ndarray:
    """Extract event timestamps from Arduino serial CSV for a given edge type."""
    df = pd.read_csv(serial_path)
    mask = df["line"].astype(str).str.contains(edge, na=False)
    return df.loc[mask, "elapsed_s"].to_numpy(dtype=np.float64)


def align_peaks(
    linear_imu: np.ndarray,
    imu_t: np.ndarray,
    event_times: np.ndarray,
    search_each_side: int = DEFAULT_PEAK_SEARCH_EACH_SIDE,
) -> tuple[list[int], list[float]]:
    """Align Arduino event times to nearest linear-acceleration magnitude peak.

    Returns (peak_indices, peak_times).
    """
    peaks: list[int] = []
    peak_times: list[float] = []
    for event_t in event_times:
        event_idx = int(np.argmin(np.abs(imu_t - event_t)))
        start = event_idx - search_each_side
        end = event_idx + search_each_side + 1
        if start < 0 or end > linear_imu.shape[0]:
            continue
        norm = np.linalg.norm(linear_imu[start:end, :3], axis=1)
        peak_offset = int(np.argmax(norm))
        peak_idx = start + peak_offset
        peaks.append(peak_idx)
        peak_times.append(float(imu_t[peak_idx]))
    return peaks, peak_times


def _safe_output_name(name: str) -> str:
    return Path(str(name).replace(":", "_")).name


def decrypt_dataset(data_dir: Path, keep_decrypted: bool = False) -> Path:
    """Decrypt all .enc files from data_dir into a temporary (or persistent) directory."""
    if keep_decrypted:
        decrypted_dir = data_dir.parent / "data_decrypted"
        if decrypted_dir.exists():
            shutil.rmtree(decrypted_dir)
        decrypted_dir.mkdir(parents=True)
        for path in sorted(data_dir.glob("*.enc")):
            data, original_name = _decrypt_file(path)
            (decrypted_dir / _safe_output_name(original_name)).write_bytes(data)
        return decrypted_dir

    tmp = tempfile.TemporaryDirectory(prefix="drum_gesture_viz_")
    decrypted_dir = Path(tmp.name)
    for path in sorted(data_dir.glob("*.enc")):
        data, original_name = _decrypt_file(path)
        (decrypted_dir / _safe_output_name(original_name)).write_bytes(data)
    _TEMP_DIR_HANDLES.append(tmp)
    return decrypted_dir


def collect_recordings(decrypted_dir: Path) -> dict[str, dict]:
    """Collect and match IMU+serial recordings from decrypted directory.

    Returns dict keyed by stamp, each with imu_path, serial_path, user, label.
    """
    imu_by_stamp: dict[str, tuple[Path, str, int]] = {}
    serial_by_stamp: dict[str, tuple[Path, str, int]] = {}

    for path in sorted(decrypted_dir.glob("*.npy")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            continue
        stamp, user, label = parsed
        imu_by_stamp[stamp] = (path, user, label)

    for path in sorted(decrypted_dir.glob("serial_*.csv")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            continue
        stamp, user, label = parsed
        serial_by_stamp[stamp] = (path, user, label)

    recordings: dict[str, dict] = {}
    for stamp, (imu_path, imu_user, imu_label) in sorted(imu_by_stamp.items()):
        serial_item = serial_by_stamp.get(stamp)
        if serial_item is None:
            continue
        serial_path, serial_user, serial_label = serial_item
        if (imu_user, imu_label) != (serial_user, serial_label):
            continue
        recordings[stamp] = {
            "stamp": stamp,
            "user": imu_user,
            "label": imu_label,
            "imu_path": imu_path,
            "serial_path": serial_path,
        }
    return recordings


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def _add_event_lines(ax, event_times, peak_times, t_min, t_max):
    """Draw Arduino events (red dashed) and aligned peaks (blue solid)."""
    for i, (ard_t, peak_t) in enumerate(zip(event_times, peak_times)):
        if t_min <= ard_t <= t_max:
            ax.axvline(x=ard_t, color=ARDUINO_COLOR, lw=1.0, alpha=0.50,
                       ls="--", zorder=2)
        if t_min <= peak_t <= t_max:
            ax.axvline(x=peak_t, color=PEAK_COLOR, lw=1.6, alpha=0.70,
                       ls="-", zorder=3)


def _add_hit_zones(ax, peak_times, pre_ms, post_ms, t_min, t_max):
    """Shade hit zones around aligned peaks (green)."""
    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0
    for peak_t in peak_times:
        if t_min <= peak_t <= t_max or True:
            ax.axvspan(peak_t - pre_s, peak_t + post_s,
                       color=HIT_ZONE_COLOR, alpha=0.08, zorder=0)


def _add_peak_annotations(ax, event_times, peak_times, t_min, t_max):
    """Annotate each peak with Arduino→peak offset in ms."""
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for i, (ard_t, peak_t) in enumerate(zip(event_times, peak_times)):
        if not (t_min <= peak_t <= t_max):
            continue
        offset_ms = (peak_t - ard_t) * 1000
        ax.annotate(
            f"#{i + 1}\n{offset_ms:+.0f}ms",
            xy=(peak_t, y_max - y_range * 0.10),
            fontsize=5.5, color="#c0392b", fontweight="bold",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=PEAK_COLOR,
                      alpha=0.75, lw=0.6),
        )


def _add_window_zones(ax, feature_imu, peaks, window, lead_offsets,
                      t_rel, t_min, t_max):
    """Shade positive-window regions (light green) over the feature timeline."""
    pre_peak = window - 1  # window ends at (peak - lead)
    for peak in peaks:
        for lead in lead_offsets:
            end_exclusive = peak - lead + 1
            start = end_exclusive - window
            if start < 0 or end_exclusive > feature_imu.shape[0]:
                continue
            win_t0 = t_rel[start]
            win_t1 = t_rel[end_exclusive - 1]
            if win_t1 < t_min or win_t0 > t_max:
                continue
            ax.axvspan(max(win_t0, t_min), min(win_t1, t_max),
                       color=WINDOW_POS_COLOR, alpha=WINDOW_POS_ALPHA, zorder=0,
                       ec=WINDOW_POS_COLOR, lw=0.8)


def _add_exclusion_zones(ax, feature_imu, peaks, t_rel, t_min, t_max,
                         exclude_before, exclude_after, window):
    """Shade the exclusion zone around each peak where negatives CANNOT be sampled.

    Matching train_local_drum_gestures.py safe_negative_end_indices():
      forbidden = [peak - exclude_before, peak + exclude_after + window - 1]
    """
    for peak in peaks:
        excl_start_idx = max(0, peak - exclude_before)
        excl_end_idx = min(feature_imu.shape[0] - 1,
                           peak + exclude_after + window - 1)
        excl_t0 = t_rel[excl_start_idx]
        excl_t1 = t_rel[excl_end_idx]
        if excl_t1 < t_min or excl_t0 > t_max:
            continue
        ax.axvspan(max(excl_t0, t_min), min(excl_t1, t_max),
                   color=EXCLUSION_ZONE_COLOR, alpha=0.07, zorder=0, lw=0)


def _add_negative_safe_zones(ax, feature_imu, peaks, t_rel, t_min, t_max,
                             exclude_before, exclude_after, window):
    """Shade safe regions where negative windows ARE sampled from.

    These are all areas OUTSIDE the exclusion zones.
    We draw gray spans between adjacent exclusion zones and at edges.
    """
    n_frames = feature_imu.shape[0]
    # Build list of (start_idx, end_idx) exclusion intervals
    excluded: list[tuple[int, int]] = []
    for peak in peaks:
        s = max(0, peak - exclude_before)
        e = min(n_frames - 1, peak + exclude_after + window - 1)
        excluded.append((s, e))

    if not excluded:
        # No peaks → entire recording is safe for negatives
        ax.axvspan(t_rel[0], t_rel[-1], color=NEG_SAFE_COLOR,
                   alpha=0.04, zorder=0, lw=0)
        return

    # Merge overlapping exclusion intervals
    excluded.sort()
    merged: list[tuple[int, int]] = [excluded[0]]
    for s, e in excluded[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Safe regions are the gaps between merged exclusion zones
    # Gap before first exclusion
    if merged[0][0] > 0:
        t0 = float(t_rel[0])
        t1 = float(t_rel[merged[0][0]])
        if t1 > t_min and t0 < t_max:
            ax.axvspan(max(t0, t_min), min(t1, t_max),
                       color=NEG_SAFE_COLOR, alpha=0.04, zorder=0, lw=0)
    # Gaps between exclusions
    for i in range(len(merged) - 1):
        t0 = float(t_rel[merged[i][1]])
        t1 = float(t_rel[merged[i + 1][0]])
        if t1 > t_min and t0 < t_max:
            ax.axvspan(max(t0, t_min), min(t1, t_max),
                       color=NEG_SAFE_COLOR, alpha=0.04, zorder=0, lw=0)
    # Gap after last exclusion
    if merged[-1][1] < n_frames - 1:
        t0 = float(t_rel[merged[-1][1]])
        t1 = float(t_rel[-1])
        if t1 > t_min and t0 < t_max:
            ax.axvspan(max(t0, t_min), min(t1, t_max),
                       color=NEG_SAFE_COLOR, alpha=0.04, zorder=0, lw=0)


def _setup_time_axes(axes, t_min, t_max):
    """Configure nice time-axis ticks based on visible duration."""
    duration = t_max - t_min
    if duration <= 3:
        major, minor = 0.2, 0.05
    elif duration <= 10:
        major, minor = 0.5, 0.1
    elif duration <= 30:
        major, minor = 1.0, 0.2
    elif duration <= 60:
        major, minor = 2.0, 0.5
    else:
        major, minor = 5.0, 1.0

    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, which="major", alpha=0.25, lw=0.6)
        ax.grid(True, which="minor", alpha=0.10, lw=0.3)
        ax.tick_params(which="major", labelsize=8, length=5)
        ax.tick_params(which="minor", length=3)
        ax.set_xlim(t_min, t_max)


def plot_recording(
    recording: dict,
    decrypted_dir: Path,
    serial_edge: str = DEFAULT_SERIAL_EDGE,
    peak_search_each_side: int = DEFAULT_PEAK_SEARCH_EACH_SIDE,
    window: int = DEFAULT_WINDOW,
    lead_offsets: tuple[int, ...] = DEFAULT_LEAD_END_OFFSETS,
    t_range: tuple[float, float] | None = None,
    click_range: tuple[int, int] | None = None,
    hit_pre_ms: int = 70,
    hit_post_ms: int = 150,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Main plotting routine.

    Produces a 4-panel figure:
      1. Raw accelerometer (xyz)
      2. Raw gyroscope (xyz)
      3. Linear acceleration (xyz)
      4. Magnitude overlay + hit zones + peak annotations
    """
    # ── load and process data ──
    raw_imu = np.load(recording["imu_path"])
    if raw_imu.ndim != 2 or raw_imu.shape[1] < 8:
        raise ValueError(f"Unsupported IMU shape: {raw_imu.shape}")

    t_rel = imu_time_seconds(raw_imu)
    raw_acc = raw_imu[:, 2:5].astype(np.float32)
    raw_gyro = raw_imu[:, 5:8].astype(np.float32)

    # Linear acceleration (gravity-compensated)
    linear_imu = estimate_linear_acceleration(raw_imu)
    linear_acc = linear_imu[:, :3]
    linear_gyro = linear_imu[:, 3:6]  # same as raw_gyro, passed through

    # Serial events + peak alignment
    events = serial_event_times(recording["serial_path"], serial_edge)
    peaks_idx, peak_times = align_peaks(linear_imu, t_rel, events, peak_search_each_side)

    # Derived magnitudes
    raw_acc_mag = np.linalg.norm(raw_acc, axis=1)
    linear_acc_mag = np.linalg.norm(linear_acc, axis=1)
    gyro_mag = np.linalg.norm(raw_gyro, axis=1)

    # ── determine time range ──
    if click_range is not None:
        c_start, c_end = click_range
        c_start = max(0, c_start)
        c_end = min(len(peak_times), c_end)
        if c_start < len(peak_times):
            t_min = peak_times[c_start] - 0.5
            t_max = peak_times[c_end - 1] + 0.8 if c_end > 0 else t_rel[-1]
        else:
            t_min, t_max = float(t_rel[0]), float(t_rel[-1])
    elif t_range is not None:
        t_min, t_max = t_range
    else:
        t_min, t_max = float(t_rel[0]), float(t_rel[-1])

    # Filter events/peaks to visible range
    events_in = events[(events >= t_min) & (events <= t_max)]
    peaks_in = np.array([p for p in peak_times if t_min <= p <= t_max])
    peaks_idx_in = [p for i, p in enumerate(peaks_idx)
                    if t_min <= peak_times[i] <= t_max]

    # ── build figure ──
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1.4],
                  hspace=0.10)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    ax_acc, ax_gyro, ax_lin, ax_mag = axes

    # ── Panel 1: Raw Accelerometer ──
    for ch, letter in enumerate("xyz"):
        ax_acc.plot(t_rel, raw_acc[:, ch], color=ACC_COLORS[letter],
                    lw=0.7, alpha=0.85, label=f"{letter}")
    _add_event_lines(ax_acc, events_in, peaks_in, t_min, t_max)
    _add_window_zones(ax_acc, linear_imu, peaks_idx_in, window, lead_offsets,
                      t_rel, t_min, t_max)
    ax_acc.set_ylabel("raw acc\n(g)", fontsize=10, fontweight="bold")
    ax_acc.legend(loc="upper right", fontsize=7, ncol=3,
                  framealpha=0.5, edgecolor="none")
    ax_acc.tick_params(labelsize=8)
    ax_acc.set_title("Raw Accelerometer (sensor-frame, includes gravity)",
                     fontsize=9, color="gray", loc="left")

    # ── Panel 2: Raw Gyroscope ──
    for ch, letter in enumerate("xyz"):
        ax_gyro.plot(t_rel, raw_gyro[:, ch], color=GYRO_COLORS[letter],
                     lw=0.7, alpha=0.85, label=f"{letter}")
    _add_event_lines(ax_gyro, events_in, peaks_in, t_min, t_max)
    _add_window_zones(ax_gyro, linear_imu, peaks_idx_in, window, lead_offsets,
                      t_rel, t_min, t_max)
    ax_gyro.set_ylabel("raw gyro\n(rad/s)", fontsize=10, fontweight="bold")
    ax_gyro.legend(loc="upper right", fontsize=7, ncol=3,
                   framealpha=0.5, edgecolor="none")
    ax_gyro.tick_params(labelsize=8)
    ax_gyro.set_title("Raw Gyroscope", fontsize=9, color="gray", loc="left")

    # ── Panel 3: Linear Acceleration ──
    for ch, letter in enumerate("xyz"):
        ax_lin.plot(t_rel, linear_acc[:, ch], color=ACC_COLORS[letter],
                    lw=0.7, alpha=0.85, label=f"{letter}")
    _add_event_lines(ax_lin, events_in, peaks_in, t_min, t_max)
    _add_hit_zones(ax_lin, peaks_in, hit_pre_ms, hit_post_ms, t_min, t_max)
    _add_window_zones(ax_lin, linear_imu, peaks_idx_in, window, lead_offsets,
                      t_rel, t_min, t_max)
    ax_lin.set_ylabel("linear acc\n(g)", fontsize=10, fontweight="bold")
    ax_lin.legend(loc="upper right", fontsize=7, ncol=3,
                  framealpha=0.5, edgecolor="none")
    ax_lin.tick_params(labelsize=8)
    ax_lin.set_title("Linear Acceleration (gravity-compensated, OnlineLinearAccelerationEstimator)",
                     fontsize=9, color="gray", loc="left")

    # ── Panel 4: Magnitudes + Annotations ──
    ax_mag.plot(t_rel, raw_acc_mag, color=MAG_COLORS["raw_acc"],
                lw=1.0, alpha=0.75, label="|raw acc|")
    ax_mag.plot(t_rel, linear_acc_mag, color=MAG_COLORS["linear_acc"],
                lw=1.3, alpha=0.85, label="|linear acc|")
    ax_mag.plot(t_rel, gyro_mag, color=MAG_COLORS["gyro"],
                lw=0.8, alpha=0.55, label="|gyro|")

    # Fill under linear_acc magnitude
    ax_mag.fill_between(t_rel, 0, linear_acc_mag,
                        color=MAG_COLORS["linear_acc"], alpha=0.06)

    _add_event_lines(ax_mag, events_in, peaks_in, t_min, t_max)
    _add_hit_zones(ax_mag, peaks_in, hit_pre_ms, hit_post_ms, t_min, t_max)
    # Positive window zones (green)
    _add_window_zones(ax_mag, linear_imu, peaks_idx_in, window, lead_offsets,
                      t_rel, t_min, t_max)
    _add_peak_annotations(ax_mag, events_in, peaks_in, t_min, t_max)

    ax_mag.set_ylabel("magnitude", fontsize=10, fontweight="bold")
    ax_mag.set_xlabel("time (s)", fontsize=11)
    ax_mag.legend(loc="upper right", fontsize=8, ncol=3,
                  framealpha=0.6, edgecolor="none")
    ax_mag.tick_params(labelsize=8)
    ax_mag.set_title("Magnitudes + Hit Zones + Positive Windows",
                     fontsize=9, color="gray", loc="left")

    # ── time axis formatting ──
    _setup_time_axes(axes, t_min, t_max)

    # ── legend for Arduino/Peak lines + zone annotations ──
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=ARDUINO_COLOR, lw=1.0, ls="--", alpha=0.6,
               label=f"Arduino {serial_edge}"),
        Line2D([0], [0], color=PEAK_COLOR, lw=1.6, ls="-", alpha=0.7,
               label="aligned peak (max |linear acc|)"),
        matplotlib.patches.Patch(color=WINDOW_POS_COLOR, alpha=WINDOW_POS_ALPHA,
                                 label=f"pos windows (lead={lead_offsets}, W={window})"),
    ]
    ax_mag.legend(handles=legend_elements, loc="lower left",
                  fontsize=6.5, framealpha=0.7, edgecolor="none",
                  ncol=2)

    # ── title ──
    user = recording["user"]
    label = recording["label"]
    stamp = recording["stamp"]
    total_events = len(events)
    aligned_count = len(peaks_idx)
    n_shown = len(events_in)

    title_text = title or (
        f"Drum Gesture Recording: user={user}  label={label}  stamp={stamp}"
    )
    fig.suptitle(title_text, fontsize=13, fontweight="bold", y=0.997)

    # ── footer with stats ──
    if aligned_count > 0:
        offsets_ms = (np.array(peak_times) - events[:aligned_count]) * 1000
        stats_text = (
            f"Total events: {total_events}  |  Aligned: {aligned_count}  |  "
            f"Peak offset: mean={offsets_ms.mean():+.1f}ms  "
            f"std={offsets_ms.std():.1f}ms  "
            f"min={offsets_ms.min():+.1f}ms  max={offsets_ms.max():+.1f}ms  |  "
            f"Shown: {n_shown} events"
        )
    else:
        stats_text = f"Total events: {total_events}  |  Aligned: {aligned_count}  |  Shown: {n_shown}"
    fig.text(0.99, 0.003, stats_text,
             ha="right", va="bottom", fontsize=7.5, color="gray",
             transform=fig.transFigure)

    # ── footer legend ──
    footer = (
        "red dashed = Arduino trigger  |  blue solid = aligned peak  |  "
        "green zone = hit zone  |  light hatching = positive training windows"
    )
    fig.text(0.01, 0.003, footer,
             ha="left", va="bottom", fontsize=7, color="gray",
             transform=fig.transFigure)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {save_path}")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize drum gesture ring IMU data + Arduino peak alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_drum_gesture.py --list
  python scripts/visualize_drum_gesture.py --stamp 1783340555
  python scripts/visualize_drum_gesture.py --user wxt --label 1
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --range 5 20
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --clicks 3 12
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --save plot.png
  python scripts/visualize_drum_gesture.py --stamp 1783340555 --edge RISING
        """,
    )
    # Selection filters
    parser.add_argument("--data-dir", default=str(ROOT / DEFAULT_DATA_DIR),
                        help=f"Directory with encrypted .enc files (default: {DEFAULT_DATA_DIR}/)")
    parser.add_argument("--stamp", default="",
                        help="Substring match on recording timestamp")
    parser.add_argument("--user", default="",
                        help="Filter by user name")
    parser.add_argument("--label", type=int, default=None,
                        help="Filter by gesture label (1-4)")
    parser.add_argument("--list", action="store_true",
                        help="List available recordings and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Run peak alignment on all matching recordings and print offset "
                             "statistics (mean/std/min/max in ms and samples). "
                             "Uses the EXACT same peak_indices logic as train_local_drum_gestures.py.")

    # Display range
    parser.add_argument("--range", nargs=2, type=float, default=None,
                        metavar=("T_MIN", "T_MAX"),
                        help="Time range in seconds")
    parser.add_argument("--clicks", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Show click #START to #END (1-indexed)")

    # Peak alignment
    parser.add_argument("--edge", choices=["FALLING", "RISING"],
                        default=DEFAULT_SERIAL_EDGE,
                        help="Arduino serial edge type (default: FALLING)")
    parser.add_argument("--peak-search", type=int,
                        default=DEFAULT_PEAK_SEARCH_EACH_SIDE,
                        help=f"Peak search radius in frames (default: {DEFAULT_PEAK_SEARCH_EACH_SIDE})")
    parser.add_argument("--lead-offsets", type=str, default="6,4,2,0",
                        help="Comma-separated lead offsets, e.g. 18,16,14,12 (default: 6,4,2,0)")

    # Hit zone display
    parser.add_argument("--hit-pre", type=int, default=70,
                        help="Hit zone pre-ms for green shading (default: 70)")
    parser.add_argument("--hit-post", type=int, default=150,
                        help="Hit zone post-ms for green shading (default: 150)")

    # Output
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of displaying")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")
    parser.add_argument("--keep-decrypted", action="store_true",
                        help="Keep decrypted files under data_decrypted/ for inspection")

    return parser.parse_args()


def list_recordings(decrypted_dir: Path) -> None:
    """Print a summary of all available recordings."""
    recordings = collect_recordings(decrypted_dir)
    if not recordings:
        print("No matching IMU+serial recordings found.")
        return

    print(f"\n  Available recordings in {decrypted_dir}/\n")
    print(f"  {'Stamp':<14} {'User':<8} {'Label':<6} {'IMU file'}")
    print(f"  {'-'*14} {'-'*8} {'-'*6} {'-'*40}")
    for stamp, rec in sorted(recordings.items()):
        print(f"  {stamp:<14} {rec['user']:<8} {rec['label']:<6} {rec['imu_path'].name}")
    print(f"\n  Total: {len(recordings)} recording(s)\n")


# ═══════════════════════════════════════════════════════════════════
# Peak offset statistics (EXACT train_local_drum_gestures.py logic)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OffsetStats:
    """Per-recording peak alignment statistics."""
    stamp: str
    user: str
    label: int
    total_events: int
    aligned_peaks: int
    skipped: int
    duration_s: float
    frames: int
    offset_ms_mean: float | None
    offset_ms_std: float | None
    offset_ms_min: float | None
    offset_ms_max: float | None
    offset_samples_mean: float | None  # matches train_local_drum_gestures.py convention
    offset_samples_std: float | None
    offset_samples_min: float | None
    offset_samples_max: float | None
    all_offsets_ms: list[float] | None  # raw data for aggregate stats


def compute_offset_stats(
    recordings: list[dict],
    decrypted_dir: Path,
    serial_edge: str,
    peak_search_each_side: int,
) -> list[OffsetStats]:
    """Compute peak-alignment offset statistics for each recording.

    Uses the **exact same** peak_indices algorithm as train_local_drum_gestures.py:
      - For each Arduino serial event, find nearest IMU timestamp index
      - Search ±peak_search_each_side frames around that index
      - Pick the frame with maximum ||linear_acc|| in that window
      - Report the time difference (peak - arduino_event) in ms and samples
    """
    results: list[OffsetStats] = []
    for rec in recordings:
        raw_imu = np.load(rec["imu_path"])
        if raw_imu.ndim != 2 or raw_imu.shape[1] < 8:
            continue

        t_rel = imu_time_seconds(raw_imu)
        duration_s = float(t_rel[-1])
        n_frames = int(raw_imu.shape[0])

        events = serial_event_times(rec["serial_path"], serial_edge)
        linear_imu = estimate_linear_acceleration(raw_imu)

        # ── EXACT replica of train_local_drum_gestures.py: peak_indices() ──
        peak_indices_list: list[int] = []
        for event_t in events:
            event_idx = int(np.argmin(np.abs(t_rel - event_t)))
            start = event_idx - peak_search_each_side
            end = event_idx + peak_search_each_side + 1
            if start < 0 or end > linear_imu.shape[0]:
                continue
            norm = np.linalg.norm(linear_imu[start:end, :3], axis=1)
            peak_indices_list.append(start + int(np.argmax(norm)))

        # ── EXACT replica of train_local_drum_gestures.py: peak_minus_event ──
        offsets_samples: list[float] = []
        offsets_ms: list[float] = []
        for event_t, peak_idx in zip(events[:len(peak_indices_list)], peak_indices_list):
            # peak_minus_event = imu_t[peak] - event_t (same as training script)
            offset_s = float(t_rel[peak_idx] - event_t)
            offsets_ms.append(offset_s * 1000.0)
            offsets_samples.append(offset_s * 200.0)  # 200 Hz sampling

        n_total = len(events)
        n_aligned = len(peak_indices_list)
        n_skipped = n_total - n_aligned

        if offsets_ms:
            arr_ms = np.asarray(offsets_ms)
            arr_samp = np.asarray(offsets_samples)
            stats = OffsetStats(
                stamp=rec["stamp"],
                user=rec["user"],
                label=rec["label"],
                total_events=n_total,
                aligned_peaks=n_aligned,
                skipped=n_skipped,
                duration_s=duration_s,
                frames=n_frames,
                offset_ms_mean=float(arr_ms.mean()),
                offset_ms_std=float(arr_ms.std()),
                offset_ms_min=float(arr_ms.min()),
                offset_ms_max=float(arr_ms.max()),
                offset_samples_mean=float(arr_samp.mean()),
                offset_samples_std=float(arr_samp.std()),
                offset_samples_min=float(arr_samp.min()),
                offset_samples_max=float(arr_samp.max()),
                all_offsets_ms=offsets_ms,
            )
        else:
            stats = OffsetStats(
                stamp=rec["stamp"],
                user=rec["user"],
                label=rec["label"],
                total_events=n_total,
                aligned_peaks=0,
                skipped=n_total,
                duration_s=duration_s,
                frames=n_frames,
                offset_ms_mean=None,
                offset_ms_std=None,
                offset_ms_min=None,
                offset_ms_max=None,
                offset_samples_mean=None,
                offset_samples_std=None,
                offset_samples_min=None,
                offset_samples_max=None,
                all_offsets_ms=None,
            )
        results.append(stats)
    return results


def print_stats_table(stats_list: list[OffsetStats]) -> None:
    """Print a formatted table of per-recording peak offset statistics."""
    if not stats_list:
        print("No recordings to display.")
        return

    # Column widths
    header = (
        f"  {'User':<6} {'Label':>5} {'Stamp':<14} "
        f"{'Events':>7} {'Aligned':>7} {'Skip':>5} "
        f"{'|':>1} "
        f"{'Mean(ms)':>9} {'Std(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8} "
        f"{'|':>1} "
        f"{'Mean(samp)':>11} {'Std(samp)':>10} {'Min(samp)':>10} {'Max(samp)':>10}"
    )
    sep = "  " + "-" * (len(header) - 2)

    print()
    print(header)
    print(sep)

    for s in stats_list:
        if s.offset_ms_mean is not None:
            print(
                f"  {s.user:<6} {s.label:>5} {s.stamp:<14} "
                f"{s.total_events:>7} {s.aligned_peaks:>7} {s.skipped:>5} "
                f"{'|':>1} "
                f"{s.offset_ms_mean:>+9.1f} {s.offset_ms_std:>8.1f} "
                f"{s.offset_ms_min:>+8.1f} {s.offset_ms_max:>+8.1f} "
                f"{'|':>1} "
                f"{s.offset_samples_mean:>+11.1f} {s.offset_samples_std:>10.1f} "
                f"{s.offset_samples_min:>+10.1f} {s.offset_samples_max:>+10.1f}"
            )
        else:
            print(
                f"  {s.user:<6} {s.label:>5} {s.stamp:<14} "
                f"{s.total_events:>7} {s.aligned_peaks:>7} {s.skipped:>5} "
                f"{'|':>1} "
                f"{'--':>9} {'--':>8} {'--':>8} {'--':>8} "
                f"{'|':>1} "
                f"{'--':>11} {'--':>10} {'--':>10} {'--':>10}"
            )
    print(sep)

    # ── aggregate summary ──
    all_offsets = []
    for s in stats_list:
        if s.all_offsets_ms:
            all_offsets.extend(s.all_offsets_ms)
    if all_offsets:
        arr = np.asarray(all_offsets)
        arr_samp = arr / 1000.0 * 200.0  # convert ms → samples
        print(f"  {'ALL':<6} {'':>5} {'':>14} "
              f"{'':>7} {len(all_offsets):>7} {'':>5} "
              f"{'|':>1} "
              f"{arr.mean():>+9.1f} {arr.std():>8.1f} "
              f"{arr.min():>+8.1f} {arr.max():>+8.1f} "
              f"{'|':>1} "
              f"{arr_samp.mean():>+11.1f} {arr_samp.std():>10.1f} "
              f"{arr_samp.min():>+10.1f} {arr_samp.max():>+10.1f}")
        print(sep)
        print(f"  Aggregate over {len(stats_list)} recording(s), {len(all_offsets)} total peaks.")
        print()

        # Per-user breakdown
        users = sorted(set(s.user for s in stats_list))
        if len(users) > 1:
            print("  Per-user summary:")
            print(f"  {'User':<6} {'Recordings':>11} {'Peaks':>7} "
                  f"{'Mean(ms)':>9} {'Std(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8}")
            print(f"  {'-'*6} {'-'*11} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")
            for user in users:
                user_offsets = []
                for s in stats_list:
                    if s.user == user and s.all_offsets_ms:
                        user_offsets.extend(s.all_offsets_ms)
                if user_offsets:
                    uarr = np.asarray(user_offsets)
                    n_recs = sum(1 for s in stats_list if s.user == user)
                    print(f"  {user:<6} {n_recs:>11} {len(user_offsets):>7} "
                          f"{uarr.mean():>+9.1f} {uarr.std():>8.1f} "
                          f"{uarr.min():>+8.1f} {uarr.max():>+8.1f}")
            print()

        # Per-label breakdown
        labels = sorted(set(s.label for s in stats_list))
        if len(labels) > 1:
            print("  Per-label summary:")
            print(f"  {'Label':<6} {'Recordings':>11} {'Peaks':>7} "
                  f"{'Mean(ms)':>9} {'Std(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8}")
            print(f"  {'-'*6} {'-'*11} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")
            for label in labels:
                label_offsets = []
                for s in stats_list:
                    if s.label == label and s.all_offsets_ms:
                        label_offsets.extend(s.all_offsets_ms)
                if label_offsets:
                    larr = np.asarray(label_offsets)
                    n_recs = sum(1 for s in stats_list if s.label == label)
                    print(f"  {label:<6} {n_recs:>11} {len(label_offsets):>7} "
                          f"{larr.mean():>+9.1f} {larr.std():>8.1f} "
                          f"{larr.min():>+8.1f} {larr.max():>+8.1f}")
            print()

    # ── warn about recordings with high skip counts ──
    skipped_recs = [s for s in stats_list if s.skipped > 0]
    if skipped_recs:
        print(f"  [{len(skipped_recs)} recording(s) had skipped events "
              f"(peak search window out of bounds)]")
        for s in skipped_recs:
            print(f"    user={s.user} label={s.label} stamp={s.stamp}: "
                  f"{s.skipped}/{s.total_events} skipped")
        print()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    if not data_dir.exists():
        print(f"[X] Data directory not found: {data_dir}")
        return 1

    decrypted_dir = decrypt_dataset(data_dir, args.keep_decrypted)

    if args.list:
        list_recordings(decrypted_dir)
        return 0

    recordings = collect_recordings(decrypted_dir)
    if not recordings:
        print("[X] No matching IMU+serial recordings found in decrypted data.")
        print("    Use --list to see available recordings.")
        return 1

    # ── filter recordings ──
    candidates = list(recordings.values())
    if args.stamp:
        candidates = [r for r in candidates if args.stamp in r["stamp"]]
    if args.user:
        candidates = [r for r in candidates if r["user"] == args.user]
    if args.label is not None:
        candidates = [r for r in candidates if r["label"] == args.label]

    if not candidates:
        print("[X] No recordings match the filter criteria.")
        print(f"    stamp='{args.stamp}' user='{args.user}' label={args.label}")
        return 1

    if len(candidates) > 1 and not args.stats:
        print(f"\n  Multiple matches ({len(candidates)}). Using the first one.")
        print(f"  To narrow: --stamp <substring> --user <name> --label <N>\n")
        for c in candidates:
            print(f"    stamp={c['stamp']}  user={c['user']}  label={c['label']}")

    # ── stats mode: process all matching recordings, no plot ──
    if args.stats:
        print(f"\n  Peak alignment statistics ({len(candidates)} recording(s) matched)")
        print(f"  Edge: {args.edge}  |  Peak search: +/-{args.peak_search} frames "
              f"(+/-{args.peak_search * 5}ms @200Hz)")
        print(f"  Algorithm: EXACT replica of train_local_drum_gestures.py peak_indices()")
        print(f"    → for each Arduino serial event, find nearest IMU timestamp")
        print(f"    → search ±{args.peak_search} frames around that index")
        print(f"    → pick frame with max ||linear_acc|| in search window")
        print(f"    → offset = imu_t[peak] - arduino_event_t")

        stats_list = compute_offset_stats(
            candidates, decrypted_dir, args.edge, args.peak_search,
        )
        print_stats_table(stats_list)
        return 0

    recording = candidates[0]

    # ── load and summarize ──
    raw_imu = np.load(recording["imu_path"])
    t_rel = imu_time_seconds(raw_imu)
    events = serial_event_times(recording["serial_path"], args.edge)
    linear_imu = estimate_linear_acceleration(raw_imu)
    peaks_idx, peak_times = align_peaks(linear_imu, t_rel, events, args.peak_search)

    print(f"\n  Recording: user={recording['user']}  label={recording['label']}  stamp={recording['stamp']}")
    print(f"  IMU file:  {recording['imu_path'].name}")
    print(f"  Serial file: {recording['serial_path'].name}")
    print(f"  Duration:  {t_rel[-1]:.1f}s  |  Frames: {len(t_rel):,}  |  Rate: ~{len(t_rel)/t_rel[-1]:.0f}Hz")
    print(f"  Serial events ({args.edge}): {len(events)}")
    print(f"  Aligned peaks:              {len(peaks_idx)}")

    if len(peaks_idx) > 0:
        offsets_ms = (np.array(peak_times) - events[:len(peaks_idx)]) * 1000
        print(f"  Peak offset (Arduino -> IMU peak):")
        print(f"    mean: {offsets_ms.mean():+.1f}ms  |  std: {offsets_ms.std():.1f}ms")
        print(f"    min:  {offsets_ms.min():+.1f}ms  |  max:  {offsets_ms.max():+.1f}ms")
        skipped = len(events) - len(peaks_idx)
        if skipped > 0:
            print(f"    ({skipped} event(s) skipped — peak search window out of bounds)")

        # ── per-click offset list ──
        print(f"\n  Per-click offsets ({args.edge}, search +/-{args.peak_search} frames):")
        # Apply click range filter for console output
        click_start = args.clicks[0] - 1 if args.clicks else 0  # 1-indexed → 0-indexed
        click_end = args.clicks[1] if args.clicks else len(peaks_idx)
        click_start = max(0, click_start)
        click_end = min(len(peaks_idx), click_end)

        # Table header
        print(f"  {'Click#':>6}  {'Arduino(s)':>11}  {'Peak(s)':>11}  "
              f"{'Offset(ms)':>10}  {'Offset(samp)':>12}  {'Bar':>8}")
        print(f"  {'-'*6}  {'-'*11}  {'-'*11}  {'-'*10}  {'-'*12}  {'-'*8}")

        max_abs_offset = float(max(abs(offsets_ms[click_start:click_end].min()),
                                    abs(offsets_ms[click_start:click_end].max())))
        bar_scale = 20.0 / max(max_abs_offset, 1.0)  # 20 chars max bar

        for i in range(click_start, click_end):
            off_ms = float(offsets_ms[i])
            off_samp = off_ms / 1000.0 * 200.0
            ard_t = events[i]
            peak_t = peak_times[i]
            # Bar: > for positive (peak after Arduino), < for negative (peak before)
            bar_len = int(abs(off_ms) * bar_scale)
            bar_char = ">" if off_ms >= 0 else "<"
            bar = bar_char * max(bar_len, 1)
            direction = "+" if off_ms >= 0 else "-"
            print(f"  {i + 1:>6}  {ard_t:>11.4f}  {peak_t:>11.4f}  "
                  f"{off_ms:>+10.1f}  {off_samp:>+12.1f}  {direction} {bar}")

        if args.clicks and click_end - click_start < len(peaks_idx):
            print(f"  ... ({len(peaks_idx) - click_end + click_start} more outside --clicks range)")

    print()

    # ── plot ──
    lead_offsets = tuple(int(x.strip()) for x in args.lead_offsets.split(","))
    plot_recording(
        recording,
        decrypted_dir,
        serial_edge=args.edge,
        peak_search_each_side=args.peak_search,
        lead_offsets=lead_offsets,
        t_range=tuple(args.range) if args.range else None,
        click_range=tuple(args.clicks) if args.clicks else None,
        hit_pre_ms=args.hit_pre,
        hit_post_ms=args.hit_post,
        title=args.title,
        save_path=args.save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
