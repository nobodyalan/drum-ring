"""Diagnose switch recording predictions — check if model trained with
switch data correctly classifies each peak's gesture.

Usage:
  python scripts/diagnose_switch_labels.py \
      --model checkpoint/drum_gesture/local_drum_switch_w20_rawlin12hc.pth \
      --stamp 1784192304
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize
from models.drum_gesture_recognizer import (
    DrumGestureRecognizer,
    OnlineLinearAccelerationEstimator,
)


def parse_recording_name(name: str) -> tuple[str, str, int] | None:
    match = re.search(r"_(\d{10})_([A-Za-z0-9_]+?)_(\d+)(?:\.npy|\.csv)$", name)
    if not match:
        return None
    stamp, user, label_text = match.groups()
    return stamp, user, int(label_text)


def _detect_switch_pattern(user: str) -> tuple[int, int] | None:
    match = re.search(r"switch(\d)(\d)", user)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_switch_label_hint(user: str, a: int, b: int) -> int | None:
    match = re.search(rf"switch{a}{b}_(\d)", user)
    if match:
        hint = int(match.group(1))
        if hint in (a, b):
            return hint
    return None


def _build_switch_labels(a, b, start_label, n_peaks):
    labels = np.empty(n_peaks, dtype=np.int64)
    if start_label == b:
        use_a = False
    else:
        use_a = True
    for i in range(n_peaks):
        labels[i] = a if use_a else b
        use_a = not use_a
    return labels


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


def serial_event_times(serial_path: Path, edge: str = "FALLING") -> np.ndarray:
    df = pd.read_csv(serial_path)
    mask = df["line"].astype(str).str.contains(edge, na=False)
    return df.loc[mask, "elapsed_s"].to_numpy(dtype=np.float64)


def peak_indices(
    linear_imu: np.ndarray,
    imu_t: np.ndarray,
    event_times: np.ndarray,
    search_each_side: int = 30,
) -> list[int]:
    peaks: list[int] = []
    for event_t in event_times:
        event_idx = int(np.argmin(np.abs(imu_t - event_t)))
        start = event_idx - search_each_side
        end = event_idx + search_each_side + 1
        if start < 0 or end > linear_imu.shape[0]:
            continue
        norm = np.linalg.norm(linear_imu[start:end, :3], axis=1)
        peaks.append(start + int(np.argmax(norm)))
    return peaks


def build_window(linear_imu, raw6, peak, window=15, lead=6):
    """Build a single window ending at (peak - lead + 1)."""
    end_exclusive = peak - lead + 1
    start = end_exclusive - window
    if start < 0 or end_exclusive > linear_imu.shape[0]:
        return None
    # rawlin12: raw_acc(3) + raw_gyro(3) + linear_acc(3) + gyro(3) = 12
    lin_window = linear_imu[start:end_exclusive, :]  # (window, 6): linear_acc+gyro
    raw_window = raw6[start:end_exclusive, :]  # (window, 6): raw_acc+raw_gyro
    return np.concatenate([raw_window, lin_window], axis=1)  # (window, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose switch label predictions")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--stamp", default="", help="Filter by stamp substring")
    parser.add_argument("--user", default="", help="Filter by user substring")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--lead", type=int, default=6)
    parser.add_argument("--peak-search", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    # ── 加载模型 ──
    print(f"Loading model: {args.model}")
    recognizer = DrumGestureRecognizer(Path(args.model))

    # ── 解密 ──
    tmp = tempfile.TemporaryDirectory(prefix="switch_diag_")
    decrypted_dir = Path(tmp.name)
    for path in sorted(data_dir.glob("*.enc")):
        materialize(path, output_dir=decrypted_dir)

    # ── 收集录制 ──
    imu_by_stamp = {}
    serial_by_stamp = {}
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

    # ── 筛选 switch 录制 ──
    matched = []
    for stamp, (imu_path, imu_user, imu_label) in sorted(imu_by_stamp.items()):
        serial_item = serial_by_stamp.get(stamp)
        if serial_item is None:
            continue
        serial_path, serial_user, serial_label = serial_item
        if imu_user != serial_user or imu_label != serial_label:
            continue
        sw = _detect_switch_pattern(imu_user)
        if sw is None:
            continue
        if args.stamp and args.stamp not in stamp:
            continue
        if args.user and args.user not in imu_user:
            continue
        matched.append((stamp, imu_user, imu_label, imu_path, serial_path, sw))

    if not matched:
        print("No switch recordings found matching filters.")
        return 1

    print(f"\n{'='*80}")
    print(f"  Diagnosing {len(matched)} switch recording(s)")
    print(f"  Window={args.window}  Lead={args.lead}  PeakSearch=+/-{args.peak_search}")
    print(f"{'='*80}\n")

    grand_correct = 0
    grand_total = 0

    for stamp, user, file_label, imu_path, serial_path, (a, b) in matched:
        raw_imu = np.load(imu_path)
        linear_imu = estimate_linear_acceleration(raw_imu)
        raw6 = raw_imu[:, 2:8].astype(np.float32)
        imu_t = imu_time_seconds(raw_imu)
        events = serial_event_times(serial_path, "FALLING")
        peaks = peak_indices(linear_imu, imu_t, events, args.peak_search)

        # 确定 ground-truth 标签序列
        start_label = file_label
        hint = _extract_switch_label_hint(user, a, b)
        if hint is not None:
            start_label = hint
        gt_labels = _build_switch_labels(a, b, start_label, len(peaks))

        print(f"  Stamp={stamp}  User={user}  Pattern={a}-{b}-{a}-{b}  StartLabel={start_label}")
        print(f"  Events={len(events)}  Peaks={len(peaks)}")
        print(f"  {'Peak#':>6}  {'GT':>3}  {'Pred':>4}  {'Conf':>6}  {'Probs (0-4)'}")
        print(f"  {'-'*6}  {'-'*3}  {'-'*4}  {'-'*6}  {'-'*40}")

        correct = 0
        total = 0
        for i, peak in enumerate(peaks):
            window = build_window(linear_imu, raw6, peak, args.window, args.lead)
            if window is None:
                continue
            prediction = recognizer.predict(window)
            gt = gt_labels[i]
            pred = prediction.class_id
            conf = prediction.confidence
            probs_str = " ".join(f"{p:.2f}" for p in prediction.probabilities)
            marker = "✓" if pred == gt else "✗"
            if pred == gt:
                correct += 1
            total += 1
            if total <= 50 or pred != gt:  # 显示前 50 个 + 所有错误的
                print(f"  {i+1:>6}  {gt:>3}  {pred:>4}  {conf:>5.2f}  [{probs_str}] {marker}")

        if total > 50:
            print(f"  ... ({total - 50} more, errors shown above)")

        acc = correct / total * 100 if total else 0
        print(f"  ── Accuracy: {correct}/{total} = {acc:.1f}%")
        grand_correct += correct
        grand_total += total
        print()

    grand_acc = grand_correct / grand_total * 100 if grand_total else 0
    print(f"  OVERALL Switch Accuracy: {grand_correct}/{grand_total} = {grand_acc:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
