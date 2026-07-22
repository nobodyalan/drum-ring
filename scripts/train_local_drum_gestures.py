from __future__ import annotations

"""End-to-end local drum gesture training.

Default reproduction command:

    python scripts/train_local_drum_gestures.py \
      --run-name local_drum_4gesture_lead6_gap_raw6

Pipeline:
1. decrypt encrypted recordings from data/*.enc into a temporary directory;
2. estimate online-compatible linear acceleration;
3. align serial edge events to nearby linear-acceleration peaks;
4. sample lead6 positive windows and same-recording hard negatives;
5. split by recording group, compute train-only acc/gyro z-score;
6. train RegularizedInceptionTime and save checkpoint/eval summaries.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastai.callback.schedule import fit_one_cycle, valley  # noqa: F401
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner
from fastai.losses import LabelSmoothingCrossEntropyFlat
from fastai.optimizer import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tsai.data.core import Categorize, TSDataLoaders, TSDatasets
from tsai.data.transforms import TSCutOut, TSGaussianNoise, TSInputDropout
from tsai.metrics import accuracy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize
from models.drum_gesture_recognizer import (  # noqa: E402
    OnlineLinearAccelerationEstimator,
    RegularizedInceptionTime,
)


WINDOW = 15
LEAD_END_OFFSETS = (18, 16, 14, 12)
PEAK_SEARCH_EACH_SIDE = 2 * WINDOW
EXCLUDE_BEFORE_PEAK = 30
EXCLUDE_AFTER_PEAK = 24
TRANSITION_FRAMES_BEFORE = 20  # 100ms @200Hz — region before positive windows forced as negative
TRANSITION_NEGATIVES_PER_PEAK = 4  # negative windows sampled from each peak's transition zone
HARD_NEGATIVES_PER_FILE = 380
SEED = 42
LABEL_IDS = (1, 2, 3, 4)
LABEL_LAST_DIGIT = False

_TEMP_DIR_HANDLES: list[tempfile.TemporaryDirectory] = []


@dataclass(frozen=True)
class Recording:
    stamp: str
    user: str
    label: int
    imu_path: Path
    serial_path: Path | None = None  # None → pure-negative recording (no Arduino)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 5-class local drum gesture model from encrypted data/*.enc recordings."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--work-dir", default=str(ROOT / "data_processed_local_drum_gesture"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoint" / "drum_gesture"))
    parser.add_argument("--analysis-dir", default=str(ROOT / "analysis_outputs"))
    parser.add_argument("--run-name", default="local_drum_4gesture_lead6_gap_raw6")
    parser.add_argument("--labels", type=parse_int_tuple, default=LABEL_IDS)
    parser.add_argument(
        "--source-filter",
        default="",
        help="Optional comma-separated user_label filters, e.g. lqw_1,lqw_2,wxt_1,wxt_2.",
    )
    parser.add_argument(
        "--latest-per-source",
        action="store_true",
        help="After source filtering, keep only the newest recording for each user+label.",
    )
    parser.add_argument(
        "--no-switch",
        action="store_true",
        help="Exclude switch recordings (user name contains 'switch').",
    )
    parser.add_argument(
        "--no-pure-negative",
        action="store_true",
        help="Exclude pure-negative recordings (label=0, no Arduino serial).",
    )
    parser.add_argument(
        "--label-last-digit",
        action="store_true",
        help="Interpret multi-digit suffixes such as lqw_22 as label 2.",
    )
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--lead-end-offsets", type=parse_int_tuple, default=LEAD_END_OFFSETS)
    parser.add_argument("--peak-search-samples", type=int, default=PEAK_SEARCH_EACH_SIDE)
    parser.add_argument("--exclude-before-peak", type=int, default=EXCLUDE_BEFORE_PEAK)
    parser.add_argument("--exclude-after-peak", type=int, default=EXCLUDE_AFTER_PEAK)
    parser.add_argument("--serial-edge", choices=["FALLING", "RISING"], default="FALLING")
    parser.add_argument(
        "--feature-mode",
        choices=["lin6", "rawlin12", "rawlin12_handcrafted"],
        default="lin6",
        help=(
            "lin6 uses linear_acc xyz + gyro xyz; rawlin12 prepends raw acc xyz + raw gyro xyz; "
            "rawlin12_handcrafted appends common per-frame IMU derived channels."
        ),
    )
    parser.add_argument(
        "--time-weight-pooling",
        action="store_true",
        help="Enable learnable time-weighted pooling instead of GAP. Default: off.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gaussian-noise", type=float, default=0.02)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--cutout", type=float, default=0.05)
    parser.add_argument(
        "--use-peak-exclusion",
        action="store_true",
        help="Exclude a guard zone around each peak from random negative sampling "
             "(peak-EXCLUDE_BEFORE_PEAK to peak+EXCLUDE_AFTER_PEAK+W-1).",
    )
    parser.add_argument("--hard-negatives-per-file", type=int, default=HARD_NEGATIVES_PER_FILE)
    parser.add_argument(
        "--transition-negatives-per-peak",
        type=int,
        default=TRANSITION_NEGATIVES_PER_PEAK,
        help="Number of forced negative windows sampled from the 100ms before each peak's positive region.",
    )
    parser.add_argument(
        "--transition-frames-before",
        type=int,
        default=TRANSITION_FRAMES_BEFORE,
        help="Duration (in frames @200Hz) of the forced-negative zone before each peak's positive region.",
    )
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--split-mode", choices=["group", "random"], default="group")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.20,
                        help="Validation fraction of train+val groups (default: 0.20)")
    parser.add_argument("--lr-find", action="store_true",
                        help="Run LR finder before training to pick optimal learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Training device (default: auto-detect GPU)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-windows", action="store_true")
    parser.add_argument(
        "--keep-decrypted",
        action="store_true",
        help="Keep decrypted files under work-dir/decrypted for inspection.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def decrypt_dataset(data_dir: Path, work_dir: Path, keep_decrypted: bool) -> Path:
    if keep_decrypted:
        decrypted_dir = work_dir / "decrypted"
        if decrypted_dir.exists():
            shutil.rmtree(decrypted_dir)
        decrypted_dir.mkdir(parents=True)
        for path in sorted(data_dir.glob("*.enc")):
            materialize(path, output_dir=decrypted_dir)
        return decrypted_dir

    tmp = tempfile.TemporaryDirectory(prefix="drum_gesture_decrypted_")
    decrypted_dir = Path(tmp.name)
    for path in sorted(data_dir.glob("*.enc")):
        materialize(path, output_dir=decrypted_dir)
    _TEMP_DIR_HANDLES.append(tmp)
    return decrypted_dir


def parse_recording_name(name: str) -> tuple[str, str, int] | None:
    # 允许用户名含下划线 (如 whr_fast, whrswitch12_2 等)
    # stamp 固定 10 位 Unix 时间戳, 避免 MAC 地址中短数字串(如 _72_)误匹配
    match = re.search(r"_(\d{10})_([A-Za-z0-9_]+?)_(\d+)(?:\.npy|\.csv)$", name)
    if not match:
        return None
    stamp, user, label_text = match.groups()
    label = int(label_text[-1]) if LABEL_LAST_DIGIT else int(label_text)
    return stamp, user, label


def parse_source_filter(text: str) -> set[tuple[str, int]]:
    sources = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"([A-Za-z0-9]+)_(\d+)", item)
        if not match:
            raise ValueError(f"Invalid source filter item: {item}")
        user, label_text = match.groups()
        sources.add((user, int(label_text)))
    return sources


def collect_recordings(
    decrypted_dir: Path,
    label_ids: tuple[int, ...],
    source_filter: set[tuple[str, int]] | None = None,
    latest_per_source: bool = False,
) -> tuple[list[Recording], list[dict[str, str]]]:
    imu_by_stamp = {}
    serial_by_stamp = {}
    skipped = []

    # Accept both positive labels and label=0 (pure-negative)
    accepted_labels = set(label_ids) | {0}

    for path in sorted(decrypted_dir.glob("*.npy")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            skipped.append({"file": path.name, "reason": "imu name has no configured label"})
            continue
        stamp, user, label = parsed
        if label not in accepted_labels:
            skipped.append({"file": path.name, "reason": f"imu label {label} not in {sorted(accepted_labels)}"})
            continue
        imu_by_stamp[stamp] = (path, user, label)

    for path in sorted(decrypted_dir.glob("serial_*.csv")):
        parsed = parse_recording_name(path.name)
        if parsed is None:
            skipped.append({"file": path.name, "reason": "serial name has no configured label"})
            continue
        stamp, user, label = parsed
        if label not in accepted_labels:
            skipped.append({"file": path.name, "reason": f"serial label {label} not in {sorted(accepted_labels)}"})
            continue
        serial_by_stamp[stamp] = (path, user, label)

    # All positive labels (exclude pure-negative label 0)
    positive_labels = set(label_ids)

    recordings: list[Recording] = []
    for stamp, (imu_path, imu_user, imu_label) in sorted(imu_by_stamp.items()):
        serial_item = serial_by_stamp.get(stamp)
        if serial_item is None:
            if imu_label not in positive_labels:
                # label=0 IMU without serial → pure-negative recording
                recordings.append(
                    Recording(
                        stamp=stamp,
                        user=imu_user,
                        label=0,
                        imu_path=imu_path,
                        serial_path=None,
                    )
                )
                continue
            skipped.append({"file": imu_path.name, "reason": "missing serial csv"})
            continue
        serial_path, serial_user, serial_label = serial_item
        if (imu_user, imu_label) != (serial_user, serial_label):
            skipped.append(
                {
                    "file": imu_path.name,
                    "serial": serial_path.name,
                    "reason": "imu and serial label/user mismatch",
                }
            )
            continue
        recordings.append(
            Recording(
                stamp=stamp,
                user=imu_user,
                label=imu_label,
                imu_path=imu_path,
                serial_path=serial_path,
            )
        )
    if source_filter:
        before = recordings
        recordings = [
            recording
            for recording in recordings
            if (recording.user, recording.label) in source_filter or recording.label == 0
        ]
        for recording in before:
            if (recording.user, recording.label) not in source_filter:
                skipped.append(
                    {
                        "file": recording.imu_path.name,
                        "serial": recording.serial_path.name,
                        "reason": "not selected by source filter",
                    }
                )
    if latest_per_source:
        latest: dict[tuple[str, int], Recording] = {}
        for recording in recordings:
            key = (recording.user, recording.label)
            if key not in latest or int(recording.stamp) > int(latest[key].stamp):
                latest[key] = recording
        recordings = sorted(latest.values(), key=lambda item: (item.user, item.label, item.stamp))
    return recordings, skipped


def imu_time_seconds(imu: np.ndarray) -> np.ndarray:
    if imu.shape[1] > 1 and np.nanmax(imu[:, 1]) > 0:
        t = imu[:, 1]
    else:
        t = imu[:, 0]
    return t.astype(np.float64) - float(t[0])


def estimate_linear_recording(imu: np.ndarray) -> np.ndarray:
    t = imu_time_seconds(imu)
    acc = imu[:, 2:5].astype(np.float32)
    gyro = imu[:, 5:8].astype(np.float32)
    estimator = OnlineLinearAccelerationEstimator()
    rows = np.zeros((imu.shape[0], 6), dtype=np.float32)
    last_t = float(t[0])
    for index in range(imu.shape[0]):
        current_t = float(t[index])
        dt = current_t - last_t if index else 1.0 / 200.0
        linear_acc, _gravity = estimator.update(acc[index], gyro[index], dt)
        rows[index, :3] = linear_acc
        rows[index, 3:] = gyro[index]
        last_t = current_t
    return rows


def serial_event_times(serial_path: Path, edge: str) -> np.ndarray:
    df = pd.read_csv(serial_path)
    mask = df["line"].astype(str).str.contains(edge, na=False)
    return df.loc[mask, "elapsed_s"].to_numpy(dtype=np.float64)


def peak_indices(linear_imu: np.ndarray, imu_t: np.ndarray, event_times: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for event_t in event_times:
        event_idx = int(np.argmin(np.abs(imu_t - event_t)))
        start = event_idx - PEAK_SEARCH_EACH_SIDE
        end = event_idx + PEAK_SEARCH_EACH_SIDE + 1
        if start < 0 or end > linear_imu.shape[0]:
            continue
        norm = np.linalg.norm(linear_imu[start:end, :3], axis=1)
        peaks.append(start + int(np.argmax(norm)))
    return peaks


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)


def _delta(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=0, prepend=values[:1]).astype(np.float32)


def handcrafted_channels(raw6: np.ndarray, linear_imu: np.ndarray) -> np.ndarray:
    raw_acc = raw6[:, :3]
    raw_gyro = raw6[:, 3:6]
    linear_acc = linear_imu[:, :3]
    gyro = linear_imu[:, 3:6]

    raw_acc_norm = _norm(raw_acc)
    linear_acc_norm = _norm(linear_acc)
    raw_gyro_norm = _norm(raw_gyro)
    gyro_norm = _norm(gyro)
    raw_acc_delta_norm = _norm(_delta(raw_acc))
    linear_acc_delta_norm = _norm(_delta(linear_acc))
    gyro_delta_norm = _norm(_delta(gyro))

    raw_linear_dot = np.sum(raw_acc * linear_acc, axis=1, keepdims=True).astype(np.float32)
    denom = np.maximum(raw_acc_norm * linear_acc_norm, 1e-6)
    raw_linear_cos = (raw_linear_dot / denom).astype(np.float32)

    return np.concatenate(
        [
            raw_acc_norm,
            linear_acc_norm,
            raw_gyro_norm,
            gyro_norm,
            raw_acc_delta_norm,
            linear_acc_delta_norm,
            gyro_delta_norm,
            raw_linear_dot,
            raw_linear_cos,
        ],
        axis=1,
    ).astype(np.float32)


def feature_recording(raw_imu: np.ndarray, linear_imu: np.ndarray, feature_mode: str) -> np.ndarray:
    raw6 = raw_imu[:, 2:8].astype(np.float32)
    if feature_mode == "lin6":
        return linear_imu.astype(np.float32)
    if feature_mode == "rawlin12":
        return np.concatenate([raw6, linear_imu], axis=1).astype(np.float32)
    if feature_mode == "rawlin12_handcrafted":
        base = np.concatenate([raw6, linear_imu], axis=1).astype(np.float32)
        return np.concatenate([base, handcrafted_channels(raw6, linear_imu)], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def positive_windows(feature_imu: np.ndarray, peaks: list[int]) -> np.ndarray:
    segments = []
    channel_count = int(feature_imu.shape[1])
    for peak in peaks:
        for lead in LEAD_END_OFFSETS:
            end_exclusive = peak - lead + 1
            start = end_exclusive - WINDOW
            if start < 0 or end_exclusive > feature_imu.shape[0]:
                continue
            segment = feature_imu[start:end_exclusive, :]
            if segment.shape == (WINDOW, channel_count):
                segments.append(segment)
    if not segments:
        return np.empty((0, channel_count, WINDOW), dtype=np.float32)
    return np.stack(segments, axis=0).transpose(0, 2, 1).astype(np.float32)


def _forbidden_end_inclusive_set(
    peaks: list[int],
    window: int,
    lead_offsets: tuple[int, ...],
    n_samples: int,
    transition_frames_before: int,
    use_peak_exclusion: bool = False,
    exclude_before: int = 30,
    exclude_after: int = 24,
) -> set[int]:
    """Collect end-inclusive indices that overlap with positive windows or transition zones."""
    forbidden: set[int] = set()
    for peak in peaks:
        # ── positive window region ──
        # Window for lead L covers [peak - window - L + 1, peak - L]
        # A negative window [e - window + 1, e] overlaps this positive window when:
        #   e - window + 1 <= peak - L  AND  e >= peak - window - L + 1
        # → e in [peak - window + 1,   peak - L + window - 1]
        min_lead = min(lead_offsets)
        max_lead = max(lead_offsets)
        pos_end_lo = peak - window + 1
        pos_end_hi = peak - min_lead + window - 1
        for e in range(max(0, pos_end_lo), min(n_samples, pos_end_hi + 1)):
            forbidden.add(e)

        # ── transition zone (100ms before earliest positive window) ──
        # Earliest positive window start: peak - window - max_lead + 1
        # Transition zone: [earliest_pos_start - transition_frames_before, earliest_pos_start - 1]
        # A negative window ending at e falls in the transition zone when its
        # end_inclusive is in [trans_start + window - 1,  trans_end - 1]
        earliest_pos_start = peak - window - max_lead + 1
        trans_start = earliest_pos_start - transition_frames_before
        trans_end = earliest_pos_start  # exclusive
        trans_e_lo = max(0, trans_start + window - 1)
        trans_e_hi = min(n_samples - 1, trans_end - 1)
        for e in range(trans_e_lo, trans_e_hi + 1):
            forbidden.add(e)

        # ── optional peak exclusion zone ──
        # Guard band around the peak itself: [peak - before, peak + after + window - 1]
        # Prevents random negatives from landing on the impact/reverb region
        if use_peak_exclusion:
            excl_lo = max(0, peak - exclude_before)
            excl_hi = min(n_samples - 1, peak + exclude_after + window - 1)
            for e in range(excl_lo, excl_hi + 1):
                forbidden.add(e)
    return forbidden


def safe_negative_end_indices(
    n_samples: int,
    peaks: list[int],
    window: int,
    lead_offsets: tuple[int, ...],
    transition_frames_before: int,
    use_peak_exclusion: bool = False,
    exclude_before: int = 30,
    exclude_after: int = 24,
) -> np.ndarray:
    valid = np.ones(n_samples, dtype=bool)
    valid[: window - 1] = False  # boundary protection
    forbidden = _forbidden_end_inclusive_set(
        peaks, window, lead_offsets, n_samples, transition_frames_before,
        use_peak_exclusion=use_peak_exclusion,
        exclude_before=exclude_before,
        exclude_after=exclude_after,
    )
    for e in forbidden:
        if 0 <= e < n_samples:
            valid[e] = False
    return np.where(valid)[0]


def transition_negative_windows(
    feature_imu: np.ndarray,
    peaks: list[int],
    window: int,
    lead_offsets: tuple[int, ...],
    transition_frames_before: int,
    negatives_per_peak: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample fixed negative windows from the transition zone (100ms) before each peak's
    earliest positive window.  These forced negatives teach the model that the pre-gesture
    preparation motion is *not* a gesture — reducing false triggers during gesture switching.
    """
    channel_count = int(feature_imu.shape[1])
    if channel_count <= 0 or not peaks or negatives_per_peak <= 0:
        return np.empty((0, max(channel_count, 1), window), dtype=np.float32)

    max_lead = max(lead_offsets)
    segments: list[np.ndarray] = []
    for peak in peaks:
        earliest_pos_start = peak - window - max_lead + 1
        trans_start = earliest_pos_start - transition_frames_before  # inclusive
        trans_end = earliest_pos_start  # exclusive

        # Valid end_inclusive range so the window fits entirely inside [trans_start, trans_end)
        min_end = trans_start + window - 1
        max_end = trans_end - 1
        if min_end > max_end or trans_start < 0:
            continue

        n_possible = max_end - min_end + 1
        n_sample = min(negatives_per_peak, n_possible)
        if n_sample <= 0:
            continue

        chosen = rng.choice(np.arange(min_end, max_end + 1), size=n_sample, replace=False)
        for end_inclusive in chosen:
            end_exclusive = int(end_inclusive) + 1
            start = end_exclusive - window
            segments.append(feature_imu[start:end_exclusive, :])

    if not segments:
        return np.empty((0, channel_count, window), dtype=np.float32)
    return np.stack(segments, axis=0).transpose(0, 2, 1).astype(np.float32)


def negative_windows(
    feature_imu: np.ndarray,
    peaks: list[int],
    rng: np.random.Generator,
    count: int,
    window: int,
    lead_offsets: tuple[int, ...],
    transition_frames_before: int,
    use_peak_exclusion: bool = False,
    exclude_before: int = 30,
    exclude_after: int = 24,
) -> tuple[np.ndarray, int]:
    channel_count = int(feature_imu.shape[1])
    end_indices = safe_negative_end_indices(
        feature_imu.shape[0], peaks, window, lead_offsets, transition_frames_before,
        use_peak_exclusion=use_peak_exclusion,
        exclude_before=exclude_before,
        exclude_after=exclude_after,
    )
    if end_indices.size == 0:
        return np.empty((0, channel_count, WINDOW), dtype=np.float32), 0
    chosen = rng.choice(end_indices, size=count, replace=end_indices.size < count)
    segments = []
    for end_inclusive in chosen:
        end_exclusive = int(end_inclusive) + 1
        start = end_exclusive - WINDOW
        segments.append(feature_imu[start:end_exclusive, :])
    return np.stack(segments, axis=0).transpose(0, 2, 1).astype(np.float32), int(end_indices.size)


def _detect_switch_pattern(user: str) -> tuple[int, int] | None:
    """Detect switch pattern from user name.

    Returns (label_a, label_b) if user contains 'switchAB', else None.
    e.g. whrswitch12 → (1, 2), whrswitch13 → (1, 3), whrswitch23 → (2, 3).
    """
    match = re.search(r"switch(\d)(\d)", user)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_switch_label_hint(user: str, a: int, b: int) -> int | None:
    """Try to extract intended start label from user name suffix.

    e.g. whrswitch12_2_ → hint=2, whrswitch12 → hint=None.
    """
    match = re.search(rf"switch{a}{b}_(\d)", user)
    if match:
        hint = int(match.group(1))
        if hint in (a, b):
            return hint
    return None


def _build_switch_labels(a: int, b: int, start_label: int, n_peaks: int) -> np.ndarray:
    """Generate alternating per-peak labels for switch recordings.

    Pattern: a-b-a-b-... or b-a-b-a-... depending on start_label.
    When start_label doesn't match a or b, default to starting with a.
    """
    labels = np.empty(n_peaks, dtype=np.int64)
    if start_label == b:
        use_a = False  # start with b
    else:
        use_a = True   # start with a (explicit match or default)
    for i in range(n_peaks):
        labels[i] = a if use_a else b
        use_a = not use_a
    return labels


def build_dataset(
    recordings: list[Recording],
    work_dir: Path,
    seed: int,
    hard_negatives_per_file: int,
    serial_edge: str,
    feature_mode: str,
    transition_negatives_per_peak: int = TRANSITION_NEGATIVES_PER_PEAK,
    transition_frames_before: int = TRANSITION_FRAMES_BEFORE,
    use_peak_exclusion: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    linear_dir = work_dir / "linear_acc"
    linear_dir.mkdir(parents=True, exist_ok=True)

    x_parts = []
    y_parts = []
    groups: list[str] = []
    summary = {
        "window": WINDOW,
        "lead_end_offsets": list(LEAD_END_OFFSETS),
        "peak_search_each_side": PEAK_SEARCH_EACH_SIDE,
        "serial_edge": serial_edge,
        "feature_mode": feature_mode,
        "transition_negatives": {
            "enabled": True,
            "frames_before_positive": transition_frames_before,
            "ms_before_positive": transition_frames_before * 5,  # 5 ms per frame @ 200 Hz
            "negatives_per_peak": transition_negatives_per_peak,
        },
        "negative_exclusion": {
            "use_peak_exclusion": use_peak_exclusion,
            "exclude_before_peak": EXCLUDE_BEFORE_PEAK,
            "exclude_after_peak": EXCLUDE_AFTER_PEAK,
            "window_overlap_extra_after": EXCLUDE_AFTER_PEAK + WINDOW - 1,
        },
        "recordings": [],
    }

    for recording in recordings:
        raw_imu = np.load(recording.imu_path)
        if raw_imu.ndim != 2 or raw_imu.shape[1] < 8:
            raise ValueError(f"Unsupported IMU shape for {recording.imu_path}: {raw_imu.shape}")

        linear_imu = estimate_linear_recording(raw_imu)
        np.save(linear_dir / f"{recording.stamp}_{recording.user}_{recording.label}_linear.npy", linear_imu)

        imu_t = imu_time_seconds(raw_imu)
        feature_imu = feature_recording(raw_imu, linear_imu, feature_mode)

        # ── pure-negative recording (no Arduino serial) ──
        if recording.serial_path is None:
            n_total = feature_imu.shape[0]
            # Sample extra negatives since the entire recording is background
            neg_count = hard_negatives_per_file * 3
            end_indices = np.arange(WINDOW - 1, n_total)
            if end_indices.size > 0:
                chosen = rng.choice(end_indices, size=min(neg_count, end_indices.size), replace=False)
                segments = []
                for end_inclusive in chosen:
                    end_exclusive = int(end_inclusive) + 1
                    start = end_exclusive - WINDOW
                    segments.append(feature_imu[start:end_exclusive, :])
                neg = np.stack(segments, axis=0).transpose(0, 2, 1).astype(np.float32)
            else:
                neg = np.empty((0, feature_imu.shape[1], WINDOW), dtype=np.float32)

            group = f"{recording.user}_{recording.stamp}_0"
            x_parts.append(neg)
            y_parts.append(np.zeros(neg.shape[0], dtype=np.int64))
            groups.extend([group] * neg.shape[0])

            rec_summary = {
                "stamp": recording.stamp,
                "user": recording.user,
                "label": recording.label,
                "imu_file": recording.imu_path.name,
                "serial_file": None,
                "raw_samples": int(raw_imu.shape[0]),
                "serial_events": 0,
                "usable_peaks": 0,
                "positive_windows": 0,
                "transition_negative_windows": 0,
                "random_negative_windows": int(neg.shape[0]),
                "negative_windows": int(neg.shape[0]),
                "safe_negative_end_candidates": n_total - WINDOW + 1,
                "peak_minus_event_samples_mean": None,
                "pure_negative": True,
            }
            summary["recordings"].append(rec_summary)
            continue

        # ── normal recording (with Arduino serial) ──
        events = serial_event_times(recording.serial_path, serial_edge)
        peaks = peak_indices(linear_imu, imu_t, events)

        pos = positive_windows(feature_imu, peaks)
        if pos.shape[0] > 0:
            # ── switch 标签交替逻辑 ──
            switch_pattern = _detect_switch_pattern(recording.user)
            if switch_pattern is not None:
                a, b = switch_pattern
                start_label = recording.label
                # 尝试从用户名提取真实的起始标签 (如 whrswitch12_2_ → 2)
                hint = _extract_switch_label_hint(recording.user, a, b)
                if hint is not None:
                    start_label = hint
                peak_labels = _build_switch_labels(a, b, start_label, len(peaks))
                # 每个 peak 生成 len(LEAD_END_OFFSETS) 个窗口 → 展开为逐窗口标签
                pos_labels = np.repeat(peak_labels, len(LEAD_END_OFFSETS))[: pos.shape[0]]
                pos_label_for_group = f"switch{a}{b}"
            else:
                pos_labels = np.full(pos.shape[0], recording.label, dtype=np.int64)
                pos_label_for_group = str(recording.label)

            group = f"{recording.user}_{recording.stamp}_{pos_label_for_group}"
            x_parts.append(pos)
            y_parts.append(pos_labels)
            groups.extend([group] * pos.shape[0])

        neg, safe_candidates = negative_windows(
            feature_imu,
            peaks,
            rng,
            hard_negatives_per_file,
            WINDOW,
            LEAD_END_OFFSETS,
            transition_frames_before,
            use_peak_exclusion=use_peak_exclusion,
            exclude_before=EXCLUDE_BEFORE_PEAK,
            exclude_after=EXCLUDE_AFTER_PEAK,
        )

        # ── transition negatives: forced sampling from 100ms before each peak's positive region ──
        trans_neg = transition_negative_windows(
            feature_imu,
            peaks,
            WINDOW,
            LEAD_END_OFFSETS,
            transition_frames_before,
            transition_negatives_per_peak,
            rng,
        )
        if trans_neg.shape[0] > 0:
            group = f"{recording.user}_{recording.stamp}_{recording.label}"
            x_parts.append(trans_neg)
            y_parts.append(np.zeros(trans_neg.shape[0], dtype=np.int64))
            groups.extend([group] * trans_neg.shape[0])

        if neg.shape[0] > 0:
            group = f"{recording.user}_{recording.stamp}_{recording.label}"
            x_parts.append(neg)
            y_parts.append(np.zeros(neg.shape[0], dtype=np.int64))
            groups.extend([group] * neg.shape[0])

        peak_minus_event = []
        for event_t, peak in zip(events[: len(peaks)], peaks):
            peak_minus_event.append(float(imu_t[peak] - event_t))
        rec_summary = {
            "stamp": recording.stamp,
            "user": recording.user,
            "label": recording.label,
            "imu_file": recording.imu_path.name,
            "serial_file": recording.serial_path.name if recording.serial_path else None,
            "raw_samples": int(raw_imu.shape[0]),
            "serial_events": int(events.size),
            "usable_peaks": int(len(peaks)),
            "positive_windows": int(pos.shape[0]),
            "transition_negative_windows": int(trans_neg.shape[0]),
            "random_negative_windows": int(neg.shape[0]),
            "negative_windows": int(neg.shape[0]) + int(trans_neg.shape[0]),
            "safe_negative_end_candidates": safe_candidates,
            "peak_minus_event_samples_mean": (
                float(np.mean(np.asarray(peak_minus_event) * 200.0))
                if peak_minus_event
                else None
            ),
            "pure_negative": False,
        }
        switch_pattern = _detect_switch_pattern(recording.user)
        if switch_pattern is not None:
            a, b = switch_pattern
            start_label = recording.label
            hint = _extract_switch_label_hint(recording.user, a, b)
            if hint is not None:
                start_label = hint
            peak_labels = _build_switch_labels(a, b, start_label, len(peaks))
            rec_summary["switch_pattern"] = f"{a}-{b}-{a}-{b}..."
            rec_summary["switch_start_label"] = int(start_label)
            rec_summary["per_peak_labels"] = peak_labels.tolist()
        summary["recordings"].append(rec_summary)

    if not x_parts:
        raise ValueError("No training windows were built.")
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)
    group_arr = np.asarray(groups)
    summary["x_shape"] = list(x.shape)
    summary["label_counts"] = {
        str(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))
    }
    return x, y, group_arr, summary


def split_by_group(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    group_labels = []
    for idx in range(len(unique_groups)):
        labels = y[group_indices == idx]
        positives = labels[labels > 0]
        group_labels.append(int(positives[0]) if positives.size else 0)
    group_labels = np.asarray(group_labels, dtype=np.int64)
    train_val_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        stratify=group_labels,
        random_state=seed,
    )
    train_val_labels = np.array(
        [group_labels[np.where(unique_groups == group)[0][0]] for group in train_val_groups]
    )
    label_counts = {int(label): int(count) for label, count in zip(*np.unique(train_val_labels, return_counts=True))}
    min_groups_per_class = min(label_counts.values())
    n_classes = len(label_counts)
    n_val = max(n_classes, int(val_size * len(train_val_groups)))

    try:
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=n_val,
            stratify=train_val_labels,
            random_state=seed,
        )
    except ValueError:
        # 分层拆分失败 → 手动确保每类至少 1 个 group 进验证集
        rng = np.random.default_rng(seed)
        val_groups_list = []
        train_groups_list = list(train_val_groups)
        for label in sorted(label_counts.keys()):
            # 找该类别的所有 group
            class_mask = train_val_labels == label
            class_groups = train_val_groups[class_mask].tolist()
            if len(class_groups) >= 2:
                # 随机选 1 个进验证集
                chosen = rng.choice(class_groups, size=1, replace=False)[0]
                val_groups_list.append(chosen)
                train_groups_list.remove(chosen)
        # 如果还没达到目标验证集大小，从 train 随机补充
        n_extra = n_val - len(val_groups_list)
        if n_extra > 0 and len(train_groups_list) > 0:
            extra = rng.choice(train_groups_list, size=min(n_extra, len(train_groups_list)), replace=False).tolist()
            val_groups_list.extend(extra)
            for g in extra:
                train_groups_list.remove(g)
        train_groups = np.array(train_groups_list)
        val_groups = np.array(val_groups_list)
        val_labels = [label_counts[l] for l in [int(group_labels[np.where(unique_groups == g)[0][0]]) for g in val_groups]]
        val_covered = len(set(val_labels))
        print(f"  Val split: manual (min {min_groups_per_class}/class), {len(val_groups)} groups covering {val_covered}/{n_classes} classes")
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)
    test_mask = np.isin(groups, test_groups)
    split_info = {
        "mode": "group_stratified",
        "test_size": test_size,
        "train_groups": sorted(set(groups[train_mask].astype(str).tolist())),
        "val_groups": sorted(set(groups[val_mask].astype(str).tolist())),
        "test_groups": sorted(set(groups[test_mask].astype(str).tolist())),
    }
    return x[train_mask], y[train_mask], x[val_mask], y[val_mask], x[test_mask], y[test_mask], split_info


def split_random(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    indices = np.arange(y.shape[0])
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=test_size,
        stratify=y[train_val_idx],
        random_state=seed,
    )
    split_info = {
        "mode": "random_stratified",
        "test_size": test_size,
        "train_groups": sorted(set(groups[train_idx].astype(str).tolist())),
        "val_groups": sorted(set(groups[val_idx].astype(str).tolist())),
        "test_groups": sorted(set(groups[test_idx].astype(str).tolist())),
    }
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx], split_info


def balance_classes(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    target = int(counts.min())
    selected_parts = []
    for label in labels:
        label_indices = np.where(y == label)[0]
        selected = rng.choice(label_indices, size=target, replace=False)
        selected_parts.append(selected)
    selected_indices = np.concatenate(selected_parts)
    rng.shuffle(selected_indices)
    info = {
        "enabled": True,
        "target_per_class": target,
        "before": {str(label): int(count) for label, count in zip(labels, counts)},
        "after": {str(label): target for label in labels},
    }
    return x[selected_indices], y[selected_indices], groups[selected_indices], info


def standardize_channel_groups(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if feature_mode == "lin6":
        groups = {
            "linear_acc": slice(0, 3),
            "gyro": slice(3, 6),
        }
    elif feature_mode in {"rawlin12", "rawlin12_handcrafted"}:
        groups = {
            "raw_acc": slice(0, 3),
            "raw_gyro": slice(3, 6),
            "linear_acc": slice(6, 9),
            "gyro": slice(9, 12),
        }
        if feature_mode == "rawlin12_handcrafted":
            handcrafted_names = [
                "raw_acc_norm",
                "linear_acc_norm",
                "raw_gyro_norm",
                "gyro_norm",
                "raw_acc_delta_norm",
                "linear_acc_delta_norm",
                "gyro_delta_norm",
                "raw_linear_dot",
                "raw_linear_cos",
            ]
            for offset, name in enumerate(handcrafted_names, start=12):
                groups[name] = slice(offset, offset + 1)
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    stats: dict[str, float | str] = {"mode": "channel_groups", "feature_mode": feature_mode}
    for name, channel_slice in groups.items():
        stats[f"{name}_mean"] = float(x_train[:, channel_slice, :].mean())
        stats[f"{name}_std"] = float(max(x_train[:, channel_slice, :].std(), 1e-6))

    def apply(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for name, channel_slice in groups.items():
            mean = float(stats[f"{name}_mean"])
            std = float(stats[f"{name}_std"])
            out[:, channel_slice, :] = (out[:, channel_slice, :] - mean) / std
        return out.astype(np.float32)

    return apply(x_train), apply(x_val), apply(x_test), stats


@torch.inference_mode()
def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    device = next(model.parameters()).device
    logits = model(torch.from_numpy(x).float().to(device))
    pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
    labels = sorted(np.unique(y).tolist())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
    }


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    run_name: str,
    epochs: int,
    seed: int,
    checkpoint_dir: Path,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    gaussian_noise: float,
    input_dropout: float,
    cutout: float,
    test_size: float,
    val_size: float,
    feature_mode: str,
    split_mode: str,
    lr_find: bool = False,
    device_str: str = "auto",
    time_weight_pooling: bool = False,
) -> dict:
    set_seed(seed)

    # GPU auto-detect
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"  Device: {device}")

    if split_mode == "random":
        x_train, y_train, x_val, y_val, x_test, y_test, split_info = split_random(
            x,
            y,
            groups,
            seed,
            test_size,
        )
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, split_info = split_by_group(
            x,
            y,
            groups,
            seed,
            test_size,
            val_size,
        )
    x_train, x_val, x_test, stats = standardize_channel_groups(
        x_train,
        x_val,
        x_test,
        feature_mode,
    )

    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    train_idx = list(range(len(y_train)))
    valid_idx = list(range(len(y_train), len(y_all)))
    dsets = TSDatasets(
        x_all,
        y_all,
        tfms=[None, [Categorize()]],
        splits=(train_idx, valid_idx),
        inplace=True,
    )
    batch_tfms = []
    if gaussian_noise > 0:
        batch_tfms.append(TSGaussianNoise(gaussian_noise))
    if input_dropout > 0:
        batch_tfms.append(TSInputDropout(input_dropout))
    if cutout > 0:
        batch_tfms.append(TSCutOut(cutout))
    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        bs=batch_size,
        batch_tfms=batch_tfms,
        num_workers=0,
        device=device,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        fastai_model_dir = checkpoint_dir.relative_to(ROOT)
    except ValueError:
        fastai_model_dir = checkpoint_dir
    model_head = "weighted" if time_weight_pooling else "gap"
    if time_weight_pooling:
        print("  [模型] 启用可学习时间加权池化 (weighted head)")
    model = RegularizedInceptionTime(
        c_in=dls.vars,
        c_out=dls.c,
        seq_len=WINDOW,
        head=model_head,
    ).to(device)
    learn = Learner(
        dls,
        model,
        loss_func=LabelSmoothingCrossEntropyFlat(eps=label_smoothing),
        metrics=accuracy,
        wd=weight_decay,
        opt_func=Adam,  # fastai Adam with decoupled weight decay (equivalent to AdamW)
        path=ROOT,
        model_dir=fastai_model_dir,
    )
    best_name = f"{run_name}_best"

    # LR finder (valley = loss 下降最快点, fastai 标准推荐)
    if lr_find:
        print("  Running LR finder...")
        suggested = learn.lr_find(suggest_funcs=(valley,))
        suggest_lr = float(getattr(suggested, "valley", lr))
        if suggest_lr < 1e-5:
            suggest_lr = lr
        print(f"  LR finder suggest: {suggest_lr:.2e}  (default: {lr:.2e})")
        lr = suggest_lr

    learn.fit_one_cycle(
        epochs,
        lr_max=lr,
        cbs=[SaveModelCallback(monitor="valid_loss", fname=best_name)],
    )
    learn.load(best_name)

    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    torch.save({"model": learn.model.state_dict()}, checkpoint_path)

    summary = {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "classes": {"0": "negative", **{str(label): f"gesture_{label}" for label in LABEL_IDS}},
        "split_info": split_info,
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
        "train_label_counts": {
            str(label): int(count) for label, count in zip(*np.unique(y_train, return_counts=True))
        },
        "val_label_counts": {
            str(label): int(count) for label, count in zip(*np.unique(y_val, return_counts=True))
        },
        "test_label_counts": {
            str(label): int(count) for label, count in zip(*np.unique(y_test, return_counts=True))
        },
        "standardize_mode": "channel_groups",
        "standardize_stats": stats,
        "feature_mode": feature_mode,
        "model": "RegularizedInceptionTime",
        "model_head": model_head,
        "input_shape": [int(x_train.shape[1]), WINDOW],
        "loss": "label_smoothing",
        "label_smoothing": label_smoothing,
        "wd": weight_decay,
        "gaussian_noise": gaussian_noise,
        "input_dropout": input_dropout,
        "cutout": cutout,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "best_valid_loss": float(min(row[1] for row in learn.recorder.values)),
        "best_val_accuracy": float(max(row[2] for row in learn.recorder.values)),
        "test_metrics": evaluate(learn.model.eval(), x_test, y_test),
    }
    eval_path = checkpoint_dir / f"{run_name}_eval.json"
    eval_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    global WINDOW
    global LEAD_END_OFFSETS
    global PEAK_SEARCH_EACH_SIDE
    global EXCLUDE_BEFORE_PEAK
    global EXCLUDE_AFTER_PEAK
    global LABEL_IDS
    global LABEL_LAST_DIGIT

    WINDOW = int(args.window)
    LEAD_END_OFFSETS = tuple(args.lead_end_offsets)
    PEAK_SEARCH_EACH_SIDE = int(args.peak_search_samples)
    EXCLUDE_BEFORE_PEAK = int(args.exclude_before_peak)
    EXCLUDE_AFTER_PEAK = int(args.exclude_after_peak)
    LABEL_IDS = tuple(args.labels)
    LABEL_LAST_DIGIT = bool(args.label_last_digit)

    data_dir = Path(args.data_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    decrypted_dir = decrypt_dataset(data_dir, work_dir, args.keep_decrypted)
    source_filter = parse_source_filter(args.source_filter) if args.source_filter else None
    recordings, skipped = collect_recordings(
        decrypted_dir,
        LABEL_IDS,
        source_filter=source_filter,
        latest_per_source=args.latest_per_source,
    )
    if args.no_switch:
        before = len(recordings)
        recordings = [r for r in recordings if "switch" not in r.user.lower()]
        skipped.append({"reason": f"--no-switch excluded {before - len(recordings)} switch recording(s)"})

    if args.no_pure_negative:
        before = len(recordings)
        recordings = [r for r in recordings if r.serial_path is not None]
        skipped.append({"reason": f"--no-pure-negative excluded {before - len(recordings)} pure-negative recording(s)"})

    if len(recordings) < max(2, len(LABEL_IDS)):
        raise SystemExit(f"Too few labeled recordings found: {len(recordings)}")

    x, y, groups, dataset_summary = build_dataset(
        recordings,
        work_dir,
        seed=args.seed,
        hard_negatives_per_file=args.hard_negatives_per_file,
        serial_edge=args.serial_edge,
        feature_mode=args.feature_mode,
        transition_negatives_per_peak=args.transition_negatives_per_peak,
        transition_frames_before=args.transition_frames_before,
        use_peak_exclusion=args.use_peak_exclusion,
    )
    balance_info = {"enabled": False}
    if args.balance_classes:
        x, y, groups, balance_info = balance_classes(x, y, groups, args.seed)
    dataset_summary["skipped_files"] = skipped
    dataset_summary["recording_count"] = len(recordings)
    dataset_summary["selected_recordings"] = [
        {
            "stamp": recording.stamp,
            "user": recording.user,
            "label": recording.label,
            "imu_file": recording.imu_path.name,
            "serial_file": recording.serial_path.name if recording.serial_path else None,
        }
        for recording in recordings
    ]
    dataset_summary["hard_negatives_per_file"] = args.hard_negatives_per_file
    dataset_summary["label_ids"] = list(LABEL_IDS)
    dataset_summary["source_filter"] = sorted([f"{user}_{label}" for user, label in source_filter]) if source_filter else []
    dataset_summary["latest_per_source"] = bool(args.latest_per_source)
    dataset_summary["label_last_digit"] = bool(args.label_last_digit)
    dataset_summary["balance_classes"] = balance_info
    dataset_summary["balanced_x_shape"] = list(x.shape)
    dataset_summary["balanced_label_counts"] = {
        str(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))
    }

    if args.save_windows:
        windows_dir = work_dir / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        np.save(windows_dir / f"{args.run_name}_x.npy", x)
        np.save(windows_dir / f"{args.run_name}_y.npy", y)
        np.save(windows_dir / f"{args.run_name}_groups.npy", groups)
        dataset_summary["windows"] = {
            "x": str(windows_dir / f"{args.run_name}_x.npy"),
            "y": str(windows_dir / f"{args.run_name}_y.npy"),
            "groups": str(windows_dir / f"{args.run_name}_groups.npy"),
        }

    analysis_dir.mkdir(exist_ok=True)
    dataset_path = analysis_dir / f"{args.run_name}_dataset_summary.json"
    dataset_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))

    train_summary = train_model(
        x,
        y,
        groups,
        args.run_name,
        args.epochs,
        args.seed,
        checkpoint_dir,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.label_smoothing,
        args.gaussian_noise,
        args.input_dropout,
        args.cutout,
        args.test_size,
        args.val_size,
        args.feature_mode,
        args.split_mode,
        lr_find=args.lr_find,
        device_str=args.device,
        time_weight_pooling=bool(args.time_weight_pooling),
    )
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
