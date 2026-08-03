"""
鼓击手势数据切片工具

将加密的 IMU 录制解密、对齐峰值、切分为训练窗口 (X, y, groups) 并保存。

用法:
  python scripts/slice_drum_gestures.py --run-name w15-17ch-sliced --feature-mode rawlin12_handcrafted

流程:
  1. 解密 data/*.enc → 临时目录
  2. 在线线性加速度估计
  3. Arduino 串口事件 → 峰值对齐
  4. 正例窗口 (连续 zone) + 过渡负例 + 随机负例
  5. 保存 X.npy, y.npy, groups.npy → data_processed_local_drum_gesture/windows/

python scripts/slice_drum_gestures.py --feature-mode rawlin12_handcrafted --window 15 --zone-first 22 --zone-last 16 --zone-stride 2 --transition-start 58 --transition-end 30 --transition-negatives-per-peak 6 --hard-negatives-per-file 80 --output-dir my_sliced_data --no-pure-negative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用训练脚本中的全部处理函数
from scripts.train_local_drum_gestures import (
    EXCLUDE_AFTER_PEAK,
    EXCLUDE_BEFORE_PEAK,
    HARD_NEGATIVES_PER_FILE,
    LABEL_IDS,
    LABEL_LAST_DIGIT,
    PEAK_SEARCH_EACH_SIDE,
    POSITIVE_ZONE_END_FRAMES,
    TRANSITION_NEGATIVES_PER_PEAK,
    WINDOW,
    build_dataset,
    collect_recordings,
    decrypt_dataset,
    parse_source_filter,
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="鼓击手势数据切片")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--work-dir", default=str(ROOT / "data_processed_local_drum_gesture"))
    parser.add_argument("--output-dir", default=None,
                        help="切片输出目录 (默认: work-dir/windows)")
    parser.add_argument("--run-name", default=None,
                        help="切片名称 (默认: sliced_w{WINDOW}_{feature_mode})")
    parser.add_argument("--feature-mode", choices=["lin6", "rawlin12", "rawlin12_handcrafted"],
                        default="rawlin12_handcrafted")
    parser.add_argument("--window", type=int, default=WINDOW, help="窗口大小 (帧)")
    # ── 正例 zone ──
    parser.add_argument("--zone-first", type=int, default=14,
                        help="最早正例末帧 offset (默认: 14, 即 peak-14)")
    parser.add_argument("--zone-last", type=int, default=4,
                        help="最晚正例末帧 offset (默认: 4, 即 peak-4)")
    parser.add_argument("--zone-stride", type=int, default=None,
                        help="正例采样步长 (帧, 默认: 2)")
    # ── 过渡区 ──
    parser.add_argument("--transition-start", type=int, default=50,
                        help="过渡区起始 offset (默认: 50, 即 peak-50)")
    parser.add_argument("--transition-end", type=int, default=15,
                        help="过渡区截止 offset (默认: 15, 即 peak-15)")
    parser.add_argument("--transition-negatives-per-peak", type=int, default=TRANSITION_NEGATIVES_PER_PEAK)
    # ── 其他 ──
    parser.add_argument("--labels", type=parse_int_tuple, default=LABEL_IDS)
    parser.add_argument("--label-last-digit", action="store_true")
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--latest-per-source", action="store_true")
    parser.add_argument("--no-switch", action="store_true")
    parser.add_argument("--no-pure-negative", action="store_true")
    parser.add_argument("--serial-edge", choices=["FALLING", "RISING"], default="FALLING")
    parser.add_argument("--hard-negatives-per-file", type=int, default=HARD_NEGATIVES_PER_FILE)
    parser.add_argument("--use-peak-exclusion", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-decrypted", action="store_true",
                        help="保留解密文件用于检查")
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"sliced_w{args.window}_{args.feature_mode}"

    # ── 全局常量覆写 ──
    import scripts.train_local_drum_gestures as _t
    _t.WINDOW = int(args.window)
    _t.LABEL_IDS = tuple(args.labels)
    _t.LABEL_LAST_DIGIT = bool(args.label_last_digit)
    # Zone 参数
    if args.zone_first is not None:
        _t.POS_ZONE_END_FIRST = int(args.zone_first)
    if args.zone_last is not None:
        _t.POS_ZONE_END_LAST = int(args.zone_last)
    if args.zone_stride is not None:
        _t.POS_ZONE_STRIDE = int(args.zone_stride)
    _t.POSITIVE_ZONE_END_FRAMES = tuple(
        range(_t.POS_ZONE_END_FIRST, _t.POS_ZONE_END_LAST - 1, -_t.POS_ZONE_STRIDE)
    )
    # 过渡区参数
    if args.transition_start is not None:
        _t.TRANSITION_START_OFFSET = int(args.transition_start)
    if args.transition_end is not None:
        _t.TRANSITION_END_OFFSET = int(args.transition_end)

    print(f"  窗口: {_t.WINDOW} 帧 ({_t.WINDOW*5}ms)")
    print(f"  正例 zone: peak-{_t.POS_ZONE_END_FIRST} ~ peak-{_t.POS_ZONE_END_LAST} "
          f"(stride={_t.POS_ZONE_STRIDE}, {len(_t.POSITIVE_ZONE_END_FRAMES)} 窗口/峰)")
    print(f"  过渡区: peak-{_t.TRANSITION_START_OFFSET} ~ peak-{_t.TRANSITION_END_OFFSET} "
          f"({_t.TRANSITION_START_OFFSET-_t.TRANSITION_END_OFFSET} 帧)")

    data_dir = Path(args.data_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 解密 ──
    print("解密录制文件...")
    decrypted_dir = decrypt_dataset(data_dir, work_dir, args.keep_decrypted)

    # ── 收集录制 ──
    source_filter = parse_source_filter(args.source_filter) if args.source_filter else None
    recordings, skipped = collect_recordings(
        decrypted_dir,
        _t.LABEL_IDS,
        source_filter=source_filter,
        latest_per_source=args.latest_per_source,
    )
    if args.no_switch:
        before = len(recordings)
        recordings = [r for r in recordings if "switch" not in r.user.lower()]
        skipped.append({"reason": f"--no-switch excluded {before - len(recordings)}"})
    if args.no_pure_negative:
        before = len(recordings)
        recordings = [r for r in recordings if r.serial_path is not None]
        skipped.append({"reason": f"--no-pure-negative excluded {before - len(recordings)}"})
    if len(recordings) < max(2, len(_t.LABEL_IDS)):
        raise SystemExit(f"录制不足: {len(recordings)}")

    print(f"录制数: {len(recordings)}")

    # ── 构建数据集 ──
    print("处理 + 切片...")
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
    print(f"切片已保存: {windows_dir}")

    summary_path = windows_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print(f"摘要已保存: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
