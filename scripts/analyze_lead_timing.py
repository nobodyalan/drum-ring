"""
================================================================================
 Lead Timing Analyzer — 模型发声延迟 vs 数据标注对比
================================================================================

 对每个 Arduino 标注的 peak，在峰值周围滑动模型窗口，逐帧推理，
 找到模型"首次自信预测正确手势"的帧位置，
 与 peak 时间对比得出每种手势的发声延迟。

 同时记录：误判（预测错误手势）、漏判（完全不发声）。

 输出：
   - 每种手势的检测延迟统计 (mean/std/min/max ms)
   - Lead 取值建议（过大/过小/合适）
   - 漏判和误判详情

 用法:
   python scripts/analyze_lead_timing.py \
     --model checkpoint/drum_gesture/local_drum_all_w15_lead22201816_rawlin12hc_weighted_balanced_random.pth \
     --window 15 \
     --lead-offsets 22,20,18,16

   # 只分析某个用户
   python scripts/analyze_lead_timing.py --model ... --user whr

   # 只看某个手势
   python scripts/analyze_lead_timing.py --model ... --label 3
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.drum_gesture_recognizer import (
    DRUM_CLASS_LABELS,
    DrumGestureRecognizer,
    OnlineLinearAccelerationEstimator,
    build_rawlin12_handcrafted_window,
)

# ═══════════════════════════════════════════════════════════════════
# Decryption (shared with visualize_drum_gesture.py)
# ═══════════════════════════════════════════════════════════════════

_INDEX_PATH = ROOT / "apps" / "shared" / ".ui_index.json"


def _load_index() -> dict:
    return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))


def _derive_key(seed: str, salt: bytes, rounds: int) -> bytes:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=rounds,
    )
    return base64.urlsafe_b64encode(kdf.derive(seed.encode("utf-8")))


def _decrypt_file(enc_path: Path) -> tuple[bytes, str]:
    blob = enc_path.read_bytes()
    idx = _load_index()
    phrase = "".join(chr(a ^ b) for a, b in zip(idx["left"], idx["right"]))
    stamp_b64 = idx["stamp"]
    stamp_raw = base64.b64decode(stamp_b64)

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
        raise ValueError(f"File {enc_path.name} has unexpected header: {blob[:20]!r}")

    payload = json.loads(payload_bytes.decode("utf-8"))
    salt = base64.b64decode(payload["salt"])
    token = str(payload["blob"]).encode("ascii")
    key = _derive_key(phrase, salt, int(payload["rounds"]))
    try:
        data = Fernet(key).decrypt(token)
    except InvalidToken as exc:
        raise ValueError(f"Decryption failed for {enc_path.name}") from exc
    return data, str(payload["name"])


_TEMP_DIR_HANDLES: list[tempfile.TemporaryDirectory] = []


def decrypt_dataset(data_dir: Path) -> Path:
    tmp = tempfile.TemporaryDirectory(prefix="drum_lead_analysis_")
    decrypted_dir = Path(tmp.name)
    for path in sorted(data_dir.glob("*.enc")):
        data, original_name = _decrypt_file(path)
        out_name = Path(str(original_name).replace(":", "_")).name
        (decrypted_dir / out_name).write_bytes(data)
    _TEMP_DIR_HANDLES.append(tmp)
    return decrypted_dir


# ═══════════════════════════════════════════════════════════════════
# Data loading (same as visualize_drum_gesture.py)
# ═══════════════════════════════════════════════════════════════════

def parse_recording_name(name: str) -> tuple[str, str, int] | None:
    match = re.search(r"_(\d{10})_([A-Za-z0-9_]+?)_(\d+)(?:\.npy|\.csv)$", name)
    if not match:
        return None
    return match.groups()[0], match.groups()[1], int(match.groups()[2])


def imu_time_seconds(imu: np.ndarray) -> np.ndarray:
    if imu.shape[1] > 1 and np.nanmax(imu[:, 1]) > 0:
        t = imu[:, 1]
    else:
        t = imu[:, 0]
    return t.astype(np.float64) - float(t[0])


def estimate_linear_acceleration(imu: np.ndarray) -> np.ndarray:
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
    df = pd.read_csv(serial_path)
    mask = df["line"].astype(str).str.contains(edge, na=False)
    return df.loc[mask, "elapsed_s"].to_numpy(dtype=np.float64)


def align_peaks(
    linear_imu: np.ndarray,
    imu_t: np.ndarray,
    event_times: np.ndarray,
    search_each_side: int = 30,
) -> tuple[list[int], list[float]]:
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


def collect_recordings(decrypted_dir: Path) -> dict[str, dict]:
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
# Core analysis
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PeakAnalysis:
    """Per-peak analysis result."""
    peak_index: int           # global frame index of aligned peak
    peak_time_s: float        # peak time in seconds
    arduino_time_s: float     # original Arduino trigger time
    true_label: int           # ground truth label (1-4)

    # Detection
    detected: bool            # did model ever predict true_label with confidence > threshold?
    detection_frame: int      # first frame where true_label prob > threshold (0 if not detected)
    detection_time_s: float   # time of detection frame
    detection_gap_ms: float   # (detection_time - peak_time) in ms. Negative = before peak.

    # Per-frame probabilities around peak
    prob_curve: list[dict] = field(default_factory=list)  # [{frame, pred_label, prob_per_class}]

    # Error analysis
    first_wrong_label: int = 0       # first non-zero, non-correct prediction (if any)
    first_wrong_confidence: float = 0.0
    false_negative: bool = False     # true_label never detected
    false_positive_class: int = 0    # wrong gesture detected first (before correct)


@dataclass
class GestureStats:
    """Aggregated statistics for one gesture class."""
    label: int
    label_name: str
    total_peaks: int
    detected: int                    # correctly detected
    missed: int                      # false negative (never detected)
    confused: int                    # wrong gesture detected first

    gap_mean_ms: float
    gap_std_ms: float
    gap_min_ms: float
    gap_max_ms: float
    gap_median_ms: float
    gap_p25_ms: float
    gap_p75_ms: float

    # Lead recommendation
    lead_min: int                    # min lead offset used in training
    lead_max: int                    # max lead offset used in training
    recommended_min_lead: int        # suggested min lead (frames)
    recommended_max_lead: int        # suggested max lead (frames)
    lead_verdict: str                # "too small" / "too large" / "just right"

    # Raw gaps for per-peak detail
    all_gaps_ms: list[float] = field(default_factory=list)
    missed_peaks: list[int] = field(default_factory=list)       # peak indices
    confused_peaks: list[dict] = field(default_factory=list)    # [{peak_idx, predicted_label, confidence}]


def _build_feature_window(
    raw_imu: np.ndarray,
    linear_imu: np.ndarray,
    start: int,
    window: int,
) -> np.ndarray:
    """Build (window, 12) raw+linear feature window ending at end_exclusive-1."""
    end_exclusive = start + window
    raw6 = raw_imu[start:end_exclusive, 2:8].astype(np.float32)  # (W, 6)
    lin6 = linear_imu[start:end_exclusive, :].astype(np.float32)  # (W, 6)
    return np.concatenate([raw6, lin6], axis=1)  # (W, 12)


def _prepare_batch(
    recognizer: DrumGestureRecognizer,
    raw_linear_windows: np.ndarray,  # (N, window, 12)
) -> np.ndarray:
    """Prepare a batch of raw+linear windows for model inference. Returns (N, C, T)."""
    batch = []
    for i in range(raw_linear_windows.shape[0]):
        prepared = recognizer.prepare(raw_linear_windows[i])
        batch.append(prepared)
    return np.stack(batch, axis=0)  # (N, C, T)


def analyze_recording_batched(
    recognizer: DrumGestureRecognizer,
    raw_imu: np.ndarray,
    linear_imu: np.ndarray,
    t_rel: np.ndarray,
    peaks_idx: list[int],
    true_label: int,
    window: int,
    lead_offsets: tuple[int, ...],
    search_range: tuple[int, int],
    conf_threshold: float,
) -> list[PeakAnalysis]:
    """Batch-analyze all peaks in a recording."""
    min_lead = min(lead_offsets)
    max_lead = max(lead_offsets)
    n_frames = raw_imu.shape[0]

    # ── Step 1: Build all windows for all peaks ──
    all_windows_1212: list[np.ndarray] = []
    peak_window_ranges: list[tuple[int, int, int]] = []  # (peak_idx, start_frame, end_frame)

    for peak_idx in peaks_idx:
        end_start = peak_idx - max_lead - search_range[0]
        end_stop = peak_idx - min_lead + search_range[1]
        end_start = max(window - 1, end_start)
        end_stop = min(n_frames - 1, end_stop)

        for end_inclusive in range(end_start, end_stop + 1):
            start = end_inclusive - window + 1
            raw_linear_win = _build_feature_window(raw_imu, linear_imu, start, window)
            all_windows_1212.append(raw_linear_win)
            peak_window_ranges.append((peak_idx, end_inclusive, int(raw_imu.shape[0])))

    if not all_windows_1212:
        return []

    all_windows = np.stack(all_windows_1212, axis=0)  # (N, window, 12)

    # ── Step 2: Batch inference ──
    model = recognizer.model
    model.eval()
    device = recognizer.device

    batch_size = 512  # avoid OOM
    all_probs_list = []

    for b_start in range(0, len(all_windows), batch_size):
        b_end = min(b_start + batch_size, len(all_windows))
        chunk = all_windows[b_start:b_end]
        prepared = _prepare_batch(recognizer, chunk)
        inputs = torch.from_numpy(prepared).to(device)
        with torch.inference_mode():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        all_probs_list.append(probs)

    all_probs = np.concatenate(all_probs_list, axis=0)  # (N, num_classes)

    # ── Step 3: Split results back to per-peak analysis ──
    peak_groups: dict[int, list[tuple[int, np.ndarray]]] = {}  # peak_idx -> [(end_inclusive, probs)]
    for i, (peak_idx, end_inclusive, _) in enumerate(peak_window_ranges):
        peak_groups.setdefault(peak_idx, []).append((end_inclusive, all_probs[i]))

    results: list[PeakAnalysis] = []
    for peak_idx in peaks_idx:
        group = peak_groups.get(peak_idx, [])
        if not group:
            continue

        peak_time_s = float(t_rel[peak_idx])

        detected = False
        detection_frame = 0
        detection_time_s = 0.0
        first_wrong_label = 0
        first_wrong_confidence = 0.0
        false_positive_class = 0

        for end_inclusive, probs in sorted(group, key=lambda x: x[0]):
            class_id = int(np.argmax(probs))
            conf = float(probs[class_id])

            if not detected and class_id == true_label and conf >= conf_threshold:
                detected = True
                detection_frame = int(end_inclusive)
                detection_time_s = float(t_rel[end_inclusive])

            if first_wrong_label == 0 and class_id != 0 and class_id != true_label:
                if conf >= conf_threshold:
                    first_wrong_label = class_id
                    first_wrong_confidence = conf
                    if not detected:
                        false_positive_class = class_id

        if detected:
            detection_gap_ms = (detection_time_s - peak_time_s) * 1000.0
        else:
            detection_gap_ms = float("nan")

        results.append(PeakAnalysis(
            peak_index=int(peak_idx),
            peak_time_s=peak_time_s,
            arduino_time_s=0.0,
            true_label=int(true_label),
            detected=detected,
            detection_frame=int(detection_frame),
            detection_time_s=detection_time_s,
            detection_gap_ms=detection_gap_ms,
            prob_curve=[],
            first_wrong_label=first_wrong_label,
            first_wrong_confidence=first_wrong_confidence,
            false_negative=not detected,
            false_positive_class=false_positive_class,
        ))

    return results


def compute_gesture_stats(
    analyses: list[PeakAnalysis],
    lead_offsets: tuple[int, ...],
) -> dict[int, GestureStats]:
    """Aggregate per-gesture statistics from peak analyses."""
    min_lead = min(lead_offsets)
    max_lead = max(lead_offsets)
    fps = 200.0

    # Group by true_label
    by_label: dict[int, list[PeakAnalysis]] = {}
    for a in analyses:
        by_label.setdefault(a.true_label, []).append(a)

    result: dict[int, GestureStats] = {}
    for label, items in sorted(by_label.items()):
        detected_gaps = [a.detection_gap_ms for a in items if a.detected]
        missed = [a for a in items if a.false_negative]
        confused = [a for a in items if a.false_positive_class > 0]

        if detected_gaps:
            arr = np.array(detected_gaps)
            gap_mean = float(arr.mean())
            gap_std = float(arr.std())
            gap_min = float(arr.min())
            gap_max = float(arr.max())
            gap_median = float(np.median(arr))
            gap_p25 = float(np.percentile(arr, 25))
            gap_p75 = float(np.percentile(arr, 75))
        else:
            gap_mean = gap_std = gap_min = gap_max = gap_median = gap_p25 = gap_p75 = float("nan")

        # Lead recommendation
        # The training window ends at (peak - lead_offset).
        # If model detects at frame D, the model's "natural lead" = peak - D.
        # In ms: gap_ms = (D - peak) * 5 → natural_lead_frames = -gap_ms / 5
        # We compare natural_lead_frames to [min_lead, max_lead]
        if detected_gaps:
            natural_leads_frames = -np.array(detected_gaps) / (1000.0 / fps)
            # The model uses the END of the window for prediction.
            # If it detects at frame D, the gesture was visible in window ending at D.
            # The full gesture motion happened D - peak earlier.
            mean_natural_lead = float(natural_leads_frames.mean())

            if mean_natural_lead < min_lead:
                verdict = "LEAD TOO SMALL → 建议增大 lead（手势特征出现在窗口太靠后）"
                rec_min = max(4, int(mean_natural_lead - natural_leads_frames.std()))
                rec_max = int(mean_natural_lead + natural_leads_frames.std())
            elif mean_natural_lead > max_lead:
                verdict = "LEAD TOO LARGE → 建议减小 lead（模型提前太多就检测到了）"
                rec_min = max(4, int(mean_natural_lead - natural_leads_frames.std()))
                rec_max = int(mean_natural_lead + natural_leads_frames.std())
            else:
                verdict = "✓ JUST RIGHT — lead 范围覆盖了模型检测窗口"
                rec_min = int(np.percentile(natural_leads_frames, 20))
                rec_max = int(np.percentile(natural_leads_frames, 80))
        else:
            verdict = "无法判断（没有成功检测到任何该手势）"
            rec_min = min_lead
            rec_max = max_lead

        result[label] = GestureStats(
            label=label,
            label_name=DRUM_CLASS_LABELS.get(label, str(label)),
            total_peaks=len(items),
            detected=len(detected_gaps),
            missed=len(missed),
            confused=len(confused),
            gap_mean_ms=gap_mean,
            gap_std_ms=gap_std,
            gap_min_ms=gap_min,
            gap_max_ms=gap_max,
            gap_median_ms=gap_median,
            gap_p25_ms=gap_p25,
            gap_p75_ms=gap_p75,
            lead_min=min_lead,
            lead_max=max_lead,
            recommended_min_lead=rec_min,
            recommended_max_lead=rec_max,
            lead_verdict=verdict,
            all_gaps_ms=detected_gaps,
            missed_peaks=[a.peak_index for a in missed],
            confused_peaks=[
                {"peak_idx": a.peak_index, "predicted_label": a.false_positive_class,
                 "predicted_name": DRUM_CLASS_LABELS.get(a.false_positive_class, "?"),
                 "confidence": a.first_wrong_confidence}
                for a in confused
            ],
        )

    return result


# ═══════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════

def print_results(
    stats: dict[int, GestureStats],
    model_path: str,
    window: int,
    lead_offsets: tuple[int, ...],
    conf_threshold: float,
) -> None:
    """Pretty-print the analysis results."""
    print()
    print("=" * 80)
    print("  LEAD TIMING ANALYSIS RESULTS")
    print("=" * 80)
    print(f"  Model:     {model_path}")
    print(f"  Window:    {window} frames ({window * 5}ms @200Hz)")
    print(f"  Lead:      {lead_offsets} (range: {min(lead_offsets)*5}-{max(lead_offsets)*5}ms)")
    print(f"  Threshold: {conf_threshold}")
    print()
    sys.stdout.flush()

    # ── Summary table ──
    header = (
        f"  {'手势':<10} {'总数':>5} {'检出':>5} {'漏判':>5} {'误判':>5} "
        f"{'延迟均值':>9} {'标准差':>8} {'中位数':>8} {'P25':>8} {'P75':>8} "
        f"{'Min':>8} {'Max':>8} {'|':>2} {'判定'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for label, s in sorted(stats.items()):
        if s.detected > 0:
            print(
                f"  {s.label_name:<10} {s.total_peaks:>5} {s.detected:>5} {s.missed:>5} {s.confused:>5} "
                f"{s.gap_mean_ms:>+9.1f} {s.gap_std_ms:>8.1f} {s.gap_median_ms:>+8.1f} "
                f"{s.gap_p25_ms:>+8.1f} {s.gap_p75_ms:>+8.1f} "
                f"{s.gap_min_ms:>+8.1f} {s.gap_max_ms:>+8.1f} "
                f"{'|':>2} {s.lead_verdict[:50]}"
            )
        else:
            print(
                f"  {s.label_name:<10} {s.total_peaks:>5} {s.detected:>5} {s.missed:>5} {s.confused:>5} "
                f"{'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} "
                f"{'|':>2} {s.lead_verdict[:50]}"
            )
    print(sep)
    print()

    # ── Per-class detail ──
    for label, s in sorted(stats.items()):
        print(f"  ── {s.label_name} (label={label}) ──")
        print(f"  Lead 判定: {s.lead_verdict}")
        if s.detected > 0:
            print(f"  建议 lead 范围: [{s.recommended_min_lead}, {s.recommended_max_lead}] "
                  f"({s.recommended_min_lead*5}-{s.recommended_max_lead*5}ms)")
        if s.missed > 0:
            print(f"  ⚠ 漏判: {s.missed} 个 peak（模型未检测到该手势）")
        if s.confused > 0:
            print(f"  ⚠ 误判: {s.confused} 个 peak 被误判为其他手势:")
            for cp in s.confused_peaks[:10]:
                print(f"     peak@{cp['peak_idx']} → 误判为 {cp['predicted_name']} "
                      f"(conf={cp['confidence']:.2f})")
        if s.detected > 0:
            arr = np.array(s.all_gaps_ms)
            ms_5 = float(np.percentile(arr, 5))
            ms_95 = float(np.percentile(arr, 95))
            print(f"  延迟分布: 90% 的检测落在 [{ms_5:+.1f}, {ms_95:+.1f}] ms 内")
            print(f"  (负值 = 早于 peak 检测到, 正值 = 晚于 peak 检测到)")
        print()

    # ── Cross-gesture confusion summary ──
    print("  ── 误判交叉表 ──")
    labels = sorted(stats.keys())
    confusion = {l: {} for l in labels}
    for l in labels:
        for cp in stats[l].confused_peaks:
            pred_label = cp["predicted_label"]
            confusion[l][pred_label] = confusion[l].get(pred_label, 0) + 1

    header2 = f"  {'真实↓\\预测→':<12}"
    for l in labels:
        header2 += f" {DRUM_CLASS_LABELS.get(l, str(l)):>8}"
    print(header2)
    for l in labels:
        row = f"  {stats[l].label_name:<12}"
        for pl in labels:
            count = confusion[l].get(pl, 0)
            row += f" {count:>8}"
        print(row)
    print()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析模型发声延迟 vs Arduino 标注，评估 lead 取值是否合适",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_lead_timing.py \\
    --model checkpoint/drum_gesture/local_drum_all_w15_lead22201816_rawlin12hc_weighted_balanced_random.pth \\
    --window 15 --lead-offsets 22,20,18,16

  python scripts/analyze_lead_timing.py --model ... --user whr --label 3
  python scripts/analyze_lead_timing.py --model ... --threshold 0.6
        """,
    )
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint .pth file")
    parser.add_argument("--data-dir", default=str(ROOT / "data"),
                        help="Directory with encrypted .enc files")
    parser.add_argument("--window", type=int, default=15,
                        help="Model window size in frames (default: 15)")
    parser.add_argument("--lead-offsets", type=str, default="22,20,18,16",
                        help="Comma-separated lead offsets used in training (default: 22,20,18,16)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--search-before", type=int, default=40,
                        help="Frames to search before expected detection zone (default: 40)")
    parser.add_argument("--search-after", type=int, default=20,
                        help="Frames to search after expected detection zone (default: 20)")

    # Recording filters
    parser.add_argument("--stamp", default="",
                        help="Substring match on recording timestamp")
    parser.add_argument("--user", default="",
                        help="Filter by user name")
    parser.add_argument("--label", type=int, default=None,
                        help="Filter by gesture label (1-4)")

    # Detail flags
    parser.add_argument("--per-peak", action="store_true",
                        help="Print per-peak detail (very verbose)")
    parser.add_argument("--keep-decrypted", action="store_true",
                        help="Keep decrypted files for inspection")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        print(f"[X] Model not found: {model_path}")
        return 1

    lead_offsets = tuple(int(x.strip()) for x in args.lead_offsets.split(","))
    window = int(args.window)

    print("=" * 80)
    print("  LEAD TIMING ANALYZER")
    print(f"  Model: {model_path}")
    print(f"  Window: {window}  |  Lead offsets: {lead_offsets}")
    print(f"  Confidence threshold: {args.threshold}")
    print("=" * 80, flush=True)

    # ── Load model ──
    print(f"\n  Loading model...", flush=True)
    recognizer = DrumGestureRecognizer(str(model_path), device="cpu")
    print(f"  Loaded: {recognizer.input_channels} channels, {recognizer.window_size} frames", flush=True)

    # ── Load data ──
    print(f"  Decrypting data from {data_dir}...", flush=True)
    decrypted_dir = decrypt_dataset(data_dir)
    recordings = collect_recordings(decrypted_dir)

    # ── Filter ──
    candidates = list(recordings.values())
    if args.stamp:
        candidates = [r for r in candidates if args.stamp in r["stamp"]]
    if args.user:
        candidates = [r for r in candidates if r["user"] == args.user]
    if args.label is not None:
        candidates = [r for r in candidates if r["label"] == args.label]

    if not candidates:
        print("[X] No matching recordings found.")
        return 1

    print(f"\n  Analyzing {len(candidates)} recording(s)...")

    all_analyses: list[PeakAnalysis] = []

    for rec in candidates:
        print(f"\n  ── {rec['user']}_{rec['label']} (stamp={rec['stamp']}) ──", flush=True)

        raw_imu = np.load(rec["imu_path"])
        if raw_imu.ndim != 2 or raw_imu.shape[1] < 8:
            print(f"    [SKIP] Unsupported IMU shape: {raw_imu.shape}", flush=True)
            continue

        t_rel = imu_time_seconds(raw_imu)
        print(f"    Estimating linear acceleration ({raw_imu.shape[0]} frames)...", flush=True)
        linear_imu = estimate_linear_acceleration(raw_imu)
        events = serial_event_times(rec["serial_path"], "FALLING")
        peaks_idx, peak_times = align_peaks(linear_imu, t_rel, events)

        print(f"    {len(events)} Arduino events → {len(peaks_idx)} aligned peaks", flush=True)
        print(f"    Batch-analyzing {len(peaks_idx)} peaks...", flush=True)

        analyses = analyze_recording_batched(
            recognizer=recognizer,
            raw_imu=raw_imu,
            linear_imu=linear_imu,
            t_rel=t_rel,
            peaks_idx=peaks_idx,
            true_label=rec["label"],
            window=window,
            lead_offsets=lead_offsets,
            search_range=(args.search_before, args.search_after),
            conf_threshold=args.threshold,
        )
        all_analyses.extend(analyses)

        if args.per_peak:
            for a in analyses:
                if a.false_negative or a.false_positive_class:
                    status = "MISSED" if a.false_negative else f"CONFUSED→{a.false_positive_class}"
                    print(f"    peak @frame {a.peak_index:5d}: {status}", flush=True)

        n_detected = sum(1 for a in analyses if a.detected)
        n_fn = sum(1 for a in analyses if a.false_negative)
        n_fp = sum(1 for a in analyses if a.false_positive_class)
        print(f"    Result: {n_detected} correct, {n_fn} missed, {n_fp} confused", flush=True)

    # ── Aggregate ──
    stats = compute_gesture_stats(all_analyses, lead_offsets)
    print_results(stats, str(model_path), window, lead_offsets, args.threshold)

    # ── Overall summary ──
    total = len(all_analyses)
    total_detected = sum(s.detected for s in stats.values())
    total_missed = sum(s.missed for s in stats.values())
    total_confused = sum(s.confused for s in stats.values())
    print(f"  总计: {total} peaks | 检出: {total_detected} "
          f"({total_detected/total*100:.1f}%) | "
          f"漏判: {total_missed} ({total_missed/total*100:.1f}%) | "
          f"误判: {total_confused} ({total_confused/total*100:.1f}%)")

    # ── Cleanup ──
    if not args.keep_decrypted:
        for tmp in _TEMP_DIR_HANDLES:
            tmp.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
