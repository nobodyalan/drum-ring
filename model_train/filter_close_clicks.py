"""
================================================================================
 过滤过近的连续踩踏标注
================================================================================

 场景: 录制训练数据时，偶尔会出现连续两个 Arduino 触发靠得太近的情况
       （比如传感器抖动、误触发），导致标签质量下降。

 策略: 扫描 session 的所有 click timestamp，如果连续两个踩踏的时间间隔
       小于 min_gap_sec 秒，则删除靠后的那个。反复迭代直到全部满足。

 用法:
   # 预览模式（不修改文件，只报告会删除多少）
   python filter_close_clicks.py --data_dir train_data --min-gap 0.12

   # 预览 new_train_data
   python filter_close_clicks.py --data_dir new_train_data

   # 实际执行删除（会备份原文件为 .bak）
   python filter_close_clicks.py --data_dir train_data --min-gap 0.12 --apply

   # 同时处理两个目录
   python filter_close_clicks.py --data_dir train_data --data_dir new_train_data --apply
"""

import os
import sys
import argparse
import shutil
from glob import glob

import numpy as np


def filter_clicks(timestamps: np.ndarray, min_gap_sec: float) -> tuple[np.ndarray, int]:
    """
    过滤掉与前一个保留踩踏间隔 < min_gap_sec 的踩踏。

    参数:
        timestamps: 已排序的踩踏时间戳数组 (Unix 秒)
        min_gap_sec: 最小间隔 (秒)

    返回:
        (filtered_timestamps, removed_count)
    """
    if len(timestamps) <= 1:
        return timestamps, 0

    # 贪心：保留第一个，依次检查后续
    kept = [timestamps[0]]
    removed = 0

    for ts in timestamps[1:]:
        if ts - kept[-1] < min_gap_sec:
            removed += 1
        else:
            kept.append(ts)

    return np.array(kept, dtype=timestamps.dtype), removed


def process_session(session_dir: str, min_gap_sec: float, apply: bool) -> dict | None:
    """
    处理单个 session 目录。

    返回:
        None 表示跳过（无 click 文件）
        dict: {name, path, total, removed, kept, min_gap, gaps_before, gaps_after}
    """
    # 查找 click_timestamps 文件
    click_files = sorted(glob(os.path.join(session_dir, "*_click_timestamps_60.npy")))
    if not click_files:
        return None

    click_path = click_files[0]
    timestamps = np.load(click_path)

    if len(timestamps) <= 1:
        return None

    # 统计过滤前的间隔
    gaps_before = np.diff(timestamps) * 1000  # ms
    n_too_close = int((gaps_before < min_gap_sec * 1000).sum())

    if n_too_close == 0:
        return {
            "name": os.path.basename(session_dir),
            "path": click_path,
            "total": len(timestamps),
            "removed": 0,
            "kept": len(timestamps),
            "min_gap_ms": gaps_before.min(),
            "pairs_too_close": 0,
        }

    # 过滤
    filtered, removed = filter_clicks(timestamps, min_gap_sec)

    if apply:
        # 备份原文件
        bak_path = click_path + ".bak"
        if not os.path.exists(bak_path):
            shutil.copy2(click_path, bak_path)

        # 覆写
        np.save(click_path, filtered)
        action = "APPLIED"
    else:
        action = "PREVIEW"

    gaps_after = np.diff(filtered) * 1000

    return {
        "name": os.path.basename(session_dir),
        "path": click_path,
        "total": len(timestamps),
        "removed": removed,
        "kept": len(filtered),
        "min_gap_ms": gaps_before.min(),
        "min_gap_after_ms": gaps_after.min() if len(gaps_after) > 0 else float("nan"),
        "pairs_too_close": n_too_close,
        "action": action,
    }


def scan_sessions(data_dirs: list[str]) -> list[str]:
    """扫描所有 session 子目录（包含 _click_timestamps_60.npy 的目录）"""
    sessions = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"  [!] 目录不存在: {data_dir}，跳过")
            continue
        for entry in sorted(os.listdir(data_dir)):
            full = os.path.join(data_dir, entry)
            if not os.path.isdir(full):
                continue
            if glob(os.path.join(full, "*_click_timestamps_60.npy")):
                sessions.append(full)
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description="过滤训练数据中时间间隔过近的连续踩踏标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览
  python filter_close_clicks.py --data_dir train_data --min-gap 0.12

  # 实际删除
  python filter_close_clicks.py --data_dir train_data --apply

  # 同时处理多个目录
  python filter_close_clicks.py --data_dir train_data --data_dir new_train_data --min-gap 0.15 --apply
        """,
    )
    parser.add_argument("--data_dir", type=str, action="append", default=[],
                        help="数据目录（可多次指定），如 train_data、new_train_data")
    parser.add_argument("--min-gap", type=float, default=0.12,
                        help="两个连续踩踏之间的最小间隔 (秒)，默认 0.12")
    parser.add_argument("--apply", action="store_true",
                        help="实际执行删除 + 备份（不加则仅预览）")

    args = parser.parse_args()

    if not args.data_dir:
        # 默认：自动扫描 train_data 和 new_train_data
        default_dirs = []
        for d in ["train_data", "new_train_data"]:
            if os.path.isdir(d):
                default_dirs.append(d)
        if not default_dirs:
            print("  [X] 未找到 train_data 或 new_train_data 目录，请用 --data_dir 指定")
            sys.exit(1)
        args.data_dir = default_dirs

    print("=" * 55)
    print("  过滤过近的连续踩踏标注")
    print(f"  目录: {args.data_dir}")
    print(f"  最小间隔: {args.min_gap * 1000:.0f}ms ({args.min_gap}s)")
    print(f"  模式: {'APPLY (会备份原文件)' if args.apply else 'PREVIEW (仅预览)'}")
    print("=" * 55)

    sessions = scan_sessions(args.data_dir)
    if not sessions:
        print("\n  [X] 未找到任何 session 目录")
        return

    print(f"\n  找到 {len(sessions)} 个 session\n")
    print(f"  {'Session':<25s} {'总数':>5s} {'删除':>5s} {'保留':>5s} "
          f"{'最小间隔':>8s} {'过近对':>6s}")
    print(f"  {'─' * 25} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 8} {'─' * 6}")

    total_removed = 0
    total_clicks = 0
    affected_sessions = 0

    for session_dir in sessions:
        result = process_session(session_dir, args.min_gap, args.apply)
        if result is None:
            continue

        total_clicks += result["total"]
        total_removed += result["removed"]
        if result["removed"] > 0:
            affected_sessions += 1

        min_gap_str = f"{result['min_gap_ms']:.0f}ms"
        pairs_str = str(result["pairs_too_close"]) if result["pairs_too_close"] > 0 else "-"

        if result["removed"] > 0:
            tag = " [DEL]" if args.apply else " [PREVIEW]"
            print(f"  {result['name']:<25s} {result['total']:5d} {result['removed']:5d} "
                  f"{result['kept']:5d} {min_gap_str:>8s} {pairs_str:>6s}{tag}")
        else:
            print(f"  {result['name']:<25s} {result['total']:5d} {result['removed']:5d} "
                  f"{result['kept']:5d} {min_gap_str:>8s} {pairs_str:>6s}")

    print(f"  {'─' * 25} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 8} {'─' * 6}")
    print(f"  {'合计':25s} {total_clicks:5d} {total_removed:5d} "
          f"{total_clicks - total_removed:5d}")

    print(f"\n  Summary:")
    print(f"    Session 总数: {len(sessions)}")
    print(f"    有删除的 session: {affected_sessions}")
    print(f"    总踩踏标点数: {total_clicks}")
    print(f"    总删除数: {total_removed}")
    print(f"    保留率: {(1 - total_removed / max(total_clicks, 1)) * 100:.1f}%")
    print(f"    最小间隔阈值: {args.min_gap * 1000:.0f}ms")

    if not args.apply and total_removed > 0:
        print(f"\n  [提示] 这是预览模式，未修改任何文件。")
        print(f"         确认无误后，加 --apply 执行删除。")
        print(f"         原文件会备份为 .bak。")
    elif args.apply and total_removed > 0:
        print(f"\n  [OK] 已删除 {total_removed} 个标点。原文件备份为 *_click_timestamps_60.npy.bak")
    elif total_removed == 0:
        print(f"\n  [OK] 所有 session 的踩踏间隔都 >= {args.min_gap * 1000:.0f}ms，无需删除。")


if __name__ == "__main__":
    main()
