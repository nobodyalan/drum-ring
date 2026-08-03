"""
LCY 鼓击手势数据切片工具

处理 data/newdatas/lcy 格式的录制数据，自动从 recording_meta 提取手势标签。

用法:
  python scripts/slice_lcy_gestures.py --ring BCL60371EA --run-name lcy_v1

流程:
  1. 解密 .enc 文件 → 临时目录
  2. 从 recording_meta_*.json 提取 stamp→{user, label, is_switch} 映射
  3. 筛选手部数据（排除中文文件名 = 腰/腿/手部负例）
  4. 按 --ring 指定戒指过滤 BCL 文件
  5. 匹配 IMU + serial + meta → Recording 列表
  6. 构建训练窗口 + 保存
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.shared.file_saver import materialize
from scripts.train_local_drum_gestures import (
    EXCLUDE_AFTER_PEAK,
    EXCLUDE_BEFORE_PEAK,
    HARD_NEGATIVES_PER_FILE,
    LABEL_IDS,
    PEAK_SEARCH_EACH_SIDE,
    POSITIVE_ZONE_END_FRAMES,
    Recording,
    TRANSITION_NEGATIVES_PER_PEAK,
    WINDOW,
    _detect_switch_pattern,
    build_dataset,
)


def _has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' for ch in text)


def _extract_stamp(filename: str) -> str | None:
    """Extract 10-digit Unix timestamp from a filename."""
    match = re.search(r'_(\d{10})_', filename)
    return match.group(1) if match else None


def _detect_switch_from_gesture(gesture_str: str) -> tuple[int, int] | None:
    """Parse switch pattern from gesture field like '1-2', '3-4', etc."""
    match = re.fullmatch(r'(\d+)-(\d+)', gesture_str)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def decrypt_lcy(data_dir: Path, work_dir: Path, keep_decrypted: bool = False) -> Path:
    """Decrypt .enc files, excluding phone_imu and Chinese-named files."""
    if keep_decrypted:
        decrypted_dir = work_dir / "decrypted_lcy"
        if decrypted_dir.exists():
            shutil.rmtree(decrypted_dir)
        decrypted_dir.mkdir(parents=True)
        for path in sorted(data_dir.glob("*.enc")):
            # Skip phone IMU and Chinese-named files
            if path.name.startswith("phone_imu_"):
                continue
            if _has_chinese(path.name):
                continue
            materialize(path, output_dir=decrypted_dir)
        return decrypted_dir

    tmp = tempfile.TemporaryDirectory(prefix="lcy_gesture_decrypted_")
    decrypted_dir = Path(tmp.name)
    for path in sorted(data_dir.glob("*.enc")):
        if path.name.startswith("phone_imu_"):
            continue
        if _has_chinese(path.name):
            continue
        materialize(path, output_dir=decrypted_dir)
    import scripts.train_local_drum_gestures as _t

    # Store reference so temp dir isn't garbage collected
    if not hasattr(_t, "_TEMP_DIR_HANDLES"):
        _t._TEMP_DIR_HANDLES = []
    _t._TEMP_DIR_HANDLES.append(tmp)
    return decrypted_dir


def collect_lcy_recordings(
    decrypted_dir: Path,
    ring_filter: str,
    label_ids: tuple[int, ...],
) -> tuple[list[Recording], list[dict]]:
    """Collect recordings from lcy-format decrypted directory.

    Uses recording_meta_*.json for label and switch detection.
    Filters by --ring to select one ring's data.
    """
    # ── Step 1: build meta_by_stamp from recording_meta ──
    meta_by_stamp: dict[str, dict] = {}
    for meta_path in sorted(decrypted_dir.glob("recording_meta_*.json")):
        stamp = _extract_stamp(meta_path.name)
        if stamp is None:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tags = meta.get("recording_tags", {})
        gesture_str = str(tags.get("gesture", ""))
        is_switch = "-" in gesture_str
        if is_switch:
            switch_parsed = _detect_switch_from_gesture(gesture_str)
            label = switch_parsed[0] if switch_parsed else 0  # default to first gesture
        else:
            switch_parsed = None
            label = int(gesture_str) if gesture_str.isdigit() else 0

        meta_by_stamp[stamp] = {
            "user": tags.get("user_id", "unknown"),
            "label": label,
            "is_switch": is_switch,
            "switch_pattern": switch_parsed,
            "gesture_str": gesture_str,
        }

    # ── Step 2: collect BCL files filtered by ring ──
    imu_files: dict[str, Path] = {}  # stamp → path
    for imu_path in sorted(decrypted_dir.glob("*.npy")):
        if not imu_path.name.startswith(ring_filter):
            continue
        stamp = _extract_stamp(imu_path.name)
        if stamp is None:
            continue
        imu_files[stamp] = imu_path

    # ── Step 3: collect serial files ──
    serial_files: dict[str, Path] = {}
    for ser_path in sorted(decrypted_dir.glob("serial_*.csv")):
        stamp = _extract_stamp(ser_path.name)
        if stamp is None:
            continue
        serial_files[stamp] = ser_path

    # ── Step 4: match and build Recording list ──
    accepted_labels = set(label_ids) | {0}
    recordings: list[Recording] = []
    skipped: list[dict] = []

    for stamp, imu_path in sorted(imu_files.items()):
        meta = meta_by_stamp.get(stamp)
        if meta is None:
            skipped.append({"stamp": stamp, "file": imu_path.name, "reason": "no recording_meta"})
            continue

        label = meta["label"]
        if label not in accepted_labels:
            skipped.append({"stamp": stamp, "file": imu_path.name, "reason": f"label {label} not in {sorted(accepted_labels)}"})
            continue

        serial_path = serial_files.get(stamp)
        if serial_path is None and label != 0:
            skipped.append({"stamp": stamp, "file": imu_path.name, "reason": "missing serial csv"})
            continue

        recordings.append(
            Recording(
                stamp=stamp,
                user=meta["user"],
                label=label,
                imu_path=imu_path,
                serial_path=serial_path,  # None for pure negatives
                switch_pattern=meta.get("switch_pattern"),
            )
        )

    return recordings, skipped


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="LCY 鼓击手势数据切片")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "newdatas" / "lcy"))
    parser.add_argument("--work-dir", default=str(ROOT / "data_processed_local_drum_gesture"))
    parser.add_argument("--output-dir", default=None,
                        help="切片输出目录 (默认: work-dir/windows)")
    parser.add_argument("--run-name", default=None,
                        help="切片名称 (默认: lcy_w{WINDOW}_{feature_mode})")
    parser.add_argument("--feature-mode", choices=["lin6", "rawlin12", "rawlin12_handcrafted"],
                        default="rawlin12_handcrafted")
    parser.add_argument("--window", type=int, default=WINDOW, help="窗口大小 (帧)")

    # ── 戒指选择 ──
    parser.add_argument("--ring", default="BCL60371EA",
                        choices=["BCL60371EA", "BCL60359F0"],
                        help="选择哪个戒指的 BCL 数据 (默认: BCL60371EA)")

    # ── 正例 zone ──
    parser.add_argument("--zone-first", type=int, default=14,
                        help="最早正例末帧 offset (默认: 14, 即 peak-14)")
    parser.add_argument("--zone-last", type=int, default=4,
                        help="最晚正例末帧 offset (默认: 4, 即 peak-4)")
    parser.add_argument("--zone-stride", type=int, default=2,
                        help="正例采样步长 (帧, 默认: 2)")

    # ── 过渡区 ──
    parser.add_argument("--transition-start", type=int, default=50,
                        help="过渡区起始 offset (默认: 50, 即 peak-50)")
    parser.add_argument("--transition-end", type=int, default=15,
                        help="过渡区截止 offset (默认: 15, 即 peak-15)")
    parser.add_argument("--transition-negatives-per-peak", type=int,
                        default=TRANSITION_NEGATIVES_PER_PEAK)

    # ── 标签 ──
    parser.add_argument("--labels", type=parse_int_tuple, default=(1, 2, 3, 4, 5),
                        help="手势标签 (默认: 1,2,3,4,5 含新手势 5)")
    parser.add_argument("--serial-edge", choices=["FALLING", "RISING"], default="FALLING")
    parser.add_argument("--hard-negatives-per-file", type=int, default=HARD_NEGATIVES_PER_FILE)
    parser.add_argument("--use-peak-exclusion", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-decrypted", action="store_true",
                        help="保留解密文件用于检查")
    parser.add_argument("--no-pure-negative", action="store_true",
                        help="排除纯负样本 (label=0)")

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"lcy_w{args.window}_{args.feature_mode}"

    # ── 全局常量覆写 ──
    import scripts.train_local_drum_gestures as _t

    _t.WINDOW = int(args.window)
    _t.LABEL_IDS = tuple(args.labels)
    if args.zone_first is not None:
        _t.POS_ZONE_END_FIRST = int(args.zone_first)
    if args.zone_last is not None:
        _t.POS_ZONE_END_LAST = int(args.zone_last)
    if args.zone_stride is not None:
        _t.POS_ZONE_STRIDE = int(args.zone_stride)
    _t.POSITIVE_ZONE_END_FRAMES = tuple(
        range(_t.POS_ZONE_END_FIRST, _t.POS_ZONE_END_LAST - 1, -_t.POS_ZONE_STRIDE)
    )
    if args.transition_start is not None:
        _t.TRANSITION_START_OFFSET = int(args.transition_start)
    if args.transition_end is not None:
        _t.TRANSITION_END_OFFSET = int(args.transition_end)

    print(f"  戒指: {args.ring}")
    print(f"  窗口: {_t.WINDOW} 帧 ({_t.WINDOW * 5}ms)")
    print(f"  正例 zone: peak-{_t.POS_ZONE_END_FIRST} ~ peak-{_t.POS_ZONE_END_LAST} "
          f"(stride={_t.POS_ZONE_STRIDE}, {len(_t.POSITIVE_ZONE_END_FRAMES)} 窗口/峰)")
    print(f"  过渡区: peak-{_t.TRANSITION_START_OFFSET} ~ peak-{_t.TRANSITION_END_OFFSET} "
          f"({_t.TRANSITION_START_OFFSET - _t.TRANSITION_END_OFFSET} 帧)")
    print(f"  标签: {_t.LABEL_IDS}")

    data_dir = Path(args.data_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 解密 ──
    print("解密录制文件...")
    print(f"  (排除 phone_imu 和中文文件名)")
    decrypted_dir = decrypt_lcy(data_dir, work_dir, args.keep_decrypted)

    # ── 收集录制 ──
    recordings, skipped = collect_lcy_recordings(
        decrypted_dir,
        args.ring,
        _t.LABEL_IDS,
    )
    if args.no_pure_negative:
        before = len(recordings)
        recordings = [r for r in recordings if r.serial_path is not None]
        skipped.append({"reason": f"--no-pure-negative excluded {before - len(recordings)} pure negative(s)"})

    if not recordings:
        raise SystemExit("没有找到可用录制")

    # Show summary
    print(f"\n录制统计 (戒指={args.ring}):")
    from collections import Counter
    label_counts = Counter(r.label for r in recordings)
    switch_count = sum(1 for r in recordings if r.switch_pattern is not None)
    pure_neg = sum(1 for r in recordings if r.serial_path is None)
    print(f"  总计: {len(recordings)} 个录制")
    print(f"  含 switch: {switch_count} 个")
    print(f"  纯负样本: {pure_neg} 个")
    for lbl in sorted(label_counts):
        label_name = {0: "背景", 1: "踩镲", 2: "通鼓2", 3: "军鼓", 4: "通鼓1", 5: "新手势5"}.get(lbl, str(lbl))
        print(f"    class {lbl} ({label_name}): {label_counts[lbl]} 个")
    if skipped:
        print(f"\n跳过: {len(skipped)} 个")
        for s in skipped[:5]:
            print(f"  {s}")

    # ── 构建数据集 ──
    print("\n处理 + 切片...")
    x, y, groups, dataset_summary = build_dataset(
        recordings,
        work_dir,
        seed=args.seed,
        hard_negatives_per_file=args.hard_negatives_per_file,
        serial_edge=args.serial_edge,
        feature_mode=args.feature_mode,
        transition_negatives_per_peak=args.transition_negatives_per_peak,
        use_peak_exclusion=args.use_peak_exclusion,
    )
    dataset_summary["run_name"] = args.run_name
    dataset_summary["skipped_files"] = skipped
    dataset_summary["recording_count"] = len(recordings)
    dataset_summary["label_ids"] = list(_t.LABEL_IDS)
    dataset_summary["ring"] = args.ring

    print(f"  X: {x.shape}, y: {y.shape}, 录制组: {len(set(groups))}")

    # ── 保存 ──
    if args.output_dir:
        windows_dir = Path(args.output_dir).expanduser().resolve()
    else:
        windows_dir = work_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    np.save(windows_dir / f"{args.run_name}_x.npy", x)
    np.save(windows_dir / f"{args.run_name}_y.npy", y)
    np.save(windows_dir / f"{args.run_name}_groups.npy", groups)
    print(f"\n切片已保存: {windows_dir}")

    summary_path = windows_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"摘要已保存: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
