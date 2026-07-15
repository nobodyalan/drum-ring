"""
================================================================================
 踩踏标注清理工具 — 过滤过近/偏移异常/手动删除
================================================================================

 三种功能:
   1. 过滤间隔过近的连续踩踏 (min-gap)
   2. 删除峰值偏移异常的踩踏 (remove-low-shift)
   3. 手动删除指定编号的踩踏 (remove-clicks)

 所有删除操作默认预览模式, 加 --apply 才实际执行并自动备份 .bak。

 ────────────────────────────────────────────────────────────────────────────
 参数:
   --data_dir DIR         数据目录 (可多次指定)
   --session SESSION      只处理指定 session (可多次指定)
   --min-gap SEC          最小间隔秒数 (默认: 0.12)
   --apply                实际执行删除 (不加则仅预览)
   --remove-clicks S:I1,I2  删除指定编号: session名:编号1,编号2 (1-based)
   --remove-low-shift MS  删除峰值偏移 < MS 的踩踏 (如 20)
   --list-clicks SESSION  列出 session 的所有踩踏及峰值偏移

 用法:
   # 预览间隔过滤
   python filter_close_clicks.py --data_dir train_data --min-gap 0.12

   # 只看指定 session
   python filter_close_clicks.py --data_dir train_data --session slow_0 --min-gap 0.12

   # 实际执行
   python filter_close_clicks.py --data_dir train_data --min-gap 0.12 --apply

   # 列出踩踏及峰值偏移
   python filter_close_clicks.py --data_dir train_data --list-clicks slow_0

   # 预览删除指定编号
   python filter_close_clicks.py --data_dir train_data --remove-clicks slow_0:3,7,12

   # 执行删除指定编号
   python filter_close_clicks.py --data_dir train_data --remove-clicks slow_0:3,7,12 --apply

   # 预览删除偏移<20ms的踩踏
   python filter_close_clicks.py --data_dir train_data --remove-low-shift 20

   # 执行删除
   python filter_close_clicks.py --data_dir train_data --remove-low-shift 20 --apply
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


def remove_clicks_from_session(session_dir: str, indices_1based: list[int],
                               apply: bool = False) -> dict:
    """
    从 session 的 click_timestamps 中删除指定编号 (1-based) 的踩踏.

    返回: {name, path, total, removed, kept, removed_timestamps}
    """
    click_files = sorted(glob(os.path.join(session_dir, "*_click_timestamps_60.npy")))
    if not click_files:
        return {"name": os.path.basename(session_dir), "total": 0,
                "removed": 0, "kept": 0, "removed_timestamps": []}

    click_path = click_files[0]
    timestamps = np.load(click_path)
    name = os.path.basename(session_dir)

    # 转为 0-based
    indices_0based = [i - 1 for i in indices_1based if 1 <= i <= len(timestamps)]
    invalid = [i for i in indices_1based if i < 1 or i > len(timestamps)]
    if invalid:
        print(f"  [!] {name}: 无效编号 {invalid} (共 {len(timestamps)} 次)")

    if not indices_0based:
        return {"name": name, "path": click_path, "total": len(timestamps),
                "removed": 0, "kept": len(timestamps), "removed_timestamps": []}

    keep_mask = np.ones(len(timestamps), dtype=bool)
    keep_mask[indices_0based] = False
    filtered = timestamps[keep_mask]
    removed_ts = timestamps[~keep_mask]

    if apply:
        bak_path = click_path + ".bak"
        if not os.path.exists(bak_path):
            shutil.copy2(click_path, bak_path)
        np.save(click_path, filtered)

    return {"name": name, "path": click_path, "total": len(timestamps),
            "removed": len(indices_0based), "kept": len(filtered),
            "removed_timestamps": removed_ts.tolist()}


def list_clicks_in_session(session_dir: str):
    """列出 session 的所有踩踏时间戳及峰值对齐偏移."""
    # 加载 IMU 数据用于峰值对齐
    imu_files = sorted(glob(os.path.join(session_dir, "*_imu_60.npy")))
    t0_files = sorted(glob(os.path.join(session_dir, "*_t0_60.npy")))
    if not imu_files or not t0_files:
        print(f"  session 中无 IMU 数据, 无法计算峰值偏移")
        return

    imu = np.load(imu_files[0])
    t0 = np.load(t0_files[0]).item()
    imu_time = imu[:, -1] / 1_000_000
    imu_data = imu[:, :6].astype(np.float32)

    click_files = sorted(glob(os.path.join(session_dir, "*_click_timestamps_60.npy")))
    if not click_files:
        print(f"  session 中无 click 文件")
        return

    timestamps = np.load(click_files[0])
    click_rel = timestamps - t0

    # 峰值对齐: [0, +100ms]
    ax = imu_data[:, 3].astype(np.float64)
    ay = imu_data[:, 4].astype(np.float64)
    az = imu_data[:, 5].astype(np.float64)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    shifts_ms = []
    for c in click_rel:
        mask = (imu_time >= c) & (imu_time <= c + 0.100)
        if mask.any():
            peak_time = imu_time[mask][np.argmax(accel_mag[mask])]
            shifts_ms.append((peak_time - c) * 1000)
        else:
            shifts_ms.append(float('nan'))

    shifts = np.array(shifts_ms)
    valid = ~np.isnan(shifts)
    mean_s = shifts[valid].mean() if valid.any() else float('nan')
    std_s = shifts[valid].std() if valid.any() else float('nan')

    print(f"\n  {os.path.basename(session_dir)}: {len(timestamps)} 次踩踏")
    print(f"  峰值对齐窗口: [0, +100ms] | 偏移 mean={mean_s:.0f}ms, std={std_s:.0f}ms\n")
    print(f"  {'#':>4s}  {'timestamp':>15s}  {'gap(ms)':>8s}  {'shift(ms)':>10s}  {'flag':>6s}")
    print(f"  {'─' * 4}  {'─' * 15}  {'─' * 8}  {'─' * 10}  {'─' * 6}")
    for i, ts in enumerate(timestamps):
        gap = (timestamps[i] - timestamps[i-1]) * 1000 if i > 0 else float('nan')
        gap_str = f"{gap:6.0f}" if not np.isnan(gap) else "     -"
        s = shifts[i]
        shift_str = f"{s:7.0f}" if not np.isnan(s) else "     N/A"
        # 标记异常偏移
        if np.isnan(s):
            flag = "N/A"
        elif s > mean_s + 2 * std_s:
            flag = "!HIGH"
        elif s < mean_s - 2 * std_s:
            flag = "!LOW"
        elif s > 99:  # 接近100ms边界, 可能选错峰
            flag = "!EDGE"
        else:
            flag = ""
        print(f"  {i+1:4d}  {ts:15.3f}  {gap_str}  {shift_str}  {flag}")


def remove_low_shift_clicks(session_dir: str, thresh_ms: float = 20.0,
                           apply: bool = False) -> dict:
    """
    删除峰值偏移 < thresh_ms 的踩踏.
    偏移过小说明 Arduino 触发后 IMU 没有明显的冲击上升过程 → 可能是误触发.

    返回: {name, total, removed, kept, removed_indices, shifts}
    """
    imu_files = sorted(glob(os.path.join(session_dir, "*_imu_60.npy")))
    click_files = sorted(glob(os.path.join(session_dir, "*_click_timestamps_60.npy")))
    t0_files = sorted(glob(os.path.join(session_dir, "*_t0_60.npy")))
    name = os.path.basename(session_dir)

    if not imu_files or not click_files or not t0_files:
        return {"name": name, "total": 0, "removed": 0, "kept": 0,
                "removed_indices": [], "shifts": []}

    imu = np.load(imu_files[0])
    t0 = np.load(t0_files[0]).item()
    imu_time = imu[:, -1] / 1_000_000
    imu_data = imu[:, :6].astype(np.float32)
    timestamps = np.load(click_files[0])
    click_rel = timestamps - t0

    ax = imu_data[:, 3].astype(np.float64)
    ay = imu_data[:, 4].astype(np.float64)
    az = imu_data[:, 5].astype(np.float64)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    shifts = []
    for c in click_rel:
        mask = (imu_time >= c) & (imu_time <= c + 0.100)
        if mask.any():
            peak_time = imu_time[mask][np.argmax(accel_mag[mask])]
            shifts.append((peak_time - c) * 1000)
        else:
            shifts.append(float('nan'))

    shifts = np.array(shifts)
    low_mask = (~np.isnan(shifts)) & (shifts < thresh_ms)
    removed_indices = np.where(low_mask)[0] + 1  # 1-based
    keep_mask = ~low_mask
    keep_mask[np.isnan(shifts)] = True  # NaN 的不删

    if apply and low_mask.any():
        click_path = click_files[0]
        bak_path = click_path + ".bak"
        if not os.path.exists(bak_path):
            shutil.copy2(click_path, bak_path)
        np.save(click_path, timestamps[keep_mask])

    return {"name": name, "total": len(timestamps), "removed": int(low_mask.sum()),
            "kept": int(keep_mask.sum()), "removed_indices": removed_indices.tolist(),
            "shifts": shifts.tolist()}


def main():
    parser = argparse.ArgumentParser(
        description="过滤训练数据中时间间隔过近的连续踩踏标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览
  python filter_close_clicks.py --data_dir train_data --min-gap 0.12

  # 只预览某几个 session
  python filter_close_clicks.py --data_dir train_data --session wxt_2 --session lqw_1

  # 实际删除
  python filter_close_clicks.py --data_dir train_data --apply

  # 同时处理多个目录
  python filter_close_clicks.py --data_dir train_data --data_dir new_train_data --min-gap 0.15 --apply

  # 只对某几个 session 执行
  python filter_close_clicks.py --data_dir train_data --session lqw_1 --session wxt_2 --apply
        """,
    )
    parser.add_argument("--data_dir", type=str, action="append", default=[],
                        help="数据目录（可多次指定），如 train_data、new_train_data")
    parser.add_argument("--session", type=str, action="append", default=[],
                        help="只处理指定 session（可多次指定），如 --session wxt_2 --session lqw_1")
    parser.add_argument("--min-gap", type=float, default=0.12,
                        help="两个连续踩踏之间的最小间隔 (秒)，默认 0.12")
    parser.add_argument("--apply", action="store_true",
                        help="实际执行删除 + 备份（不加则仅预览）")
    parser.add_argument("--remove-clicks", type=str, default=None,
                        metavar="SESSION:IDX,...",
                        help="删除指定 session 的特定踩踏, 如 'slow_0:3,7,12' (1-based). "
                             "可多次指定.")
    parser.add_argument("--remove-low-shift", type=float, default=None,
                        metavar="THRESH_MS",
                        help="删除峰值偏移 < THRESH_MS 的踩踏 (默认阈值 20ms). "
                             "如 --remove-low-shift 20 表示偏移<20ms的全部删除")
    parser.add_argument("--list-clicks", type=str, default=None,
                        metavar="SESSION",
                        help="列出指定 session 的所有踩踏时间戳及编号")

    args = parser.parse_args()

    # ── --list-clicks: 列出踩踏编号 ──
    if args.list_clicks:
        if not args.data_dir:
            args.data_dir = ["train_data"]
        for d in args.data_dir:
            session_path = os.path.join(d, args.list_clicks)
            if os.path.isdir(session_path):
                list_clicks_in_session(session_path)
            else:
                print(f"  [X] session 不存在: {session_path}")
        return

    # ── --remove-clicks: 删除指定编号的踩踏 ──
    if args.remove_clicks:
        if not args.data_dir:
            args.data_dir = ["train_data"]
        # 解析 "session:1,3,7" 格式
        parts = args.remove_clicks.split(":")
        if len(parts) != 2:
            print("  [X] 格式错误, 应为 SESSION:IDX,IDX,...  如 slow_0:3,7,12")
            return
        session_name, idx_str = parts
        indices = [int(x.strip()) for x in idx_str.split(",") if x.strip().isdigit()]
        if not indices:
            print("  [X] 未指定有效的踩踏编号")
            return

        found = False
        for d in args.data_dir:
            session_path = os.path.join(d, session_name)
            if os.path.isdir(session_path):
                found = True
                result = remove_clicks_from_session(session_path, indices, apply=args.apply)
                mode = "APPLIED" if args.apply else "PREVIEW"
                print(f"  [{mode}] {result['name']}: 总数={result['total']}, "
                      f"删除={result['removed']}次 (编号: {indices}), 保留={result['kept']}次")

                # 预览时也计算峰值偏移
                if not args.apply and result['total'] > 0:
                    imu_files = sorted(glob(os.path.join(session_path, "*_imu_60.npy")))
                    t0_files = sorted(glob(os.path.join(session_path, "*_t0_60.npy")))
                    if imu_files and t0_files:
                        imu = np.load(imu_files[0])
                        t0 = np.load(t0_files[0]).item()
                        imu_time = imu[:, -1] / 1_000_000
                        imu_data = imu[:, :6].astype(np.float32)
                        ax = imu_data[:, 3].astype(np.float64)
                        ay = imu_data[:, 4].astype(np.float64)
                        az = imu_data[:, 5].astype(np.float64)
                        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
                        timestamps = np.load(result['path'])
                        click_rel = timestamps - t0
                        print(f"  峰值对齐偏移 [0,+100ms]:")
                        for idx in indices:
                            i = idx - 1  # 0-based
                            if 0 <= i < len(click_rel):
                                c = click_rel[i]
                                mask = (imu_time >= c) & (imu_time <= c + 0.100)
                                if mask.any():
                                    peak_t = imu_time[mask][np.argmax(accel_mag[mask])]
                                    shift = (peak_t - c) * 1000
                                    print(f"    #{idx}: shift={shift:.0f}ms")
                                else:
                                    print(f"    #{idx}: 搜索窗口内无峰值!")
                if not args.apply:
                    print(f"  [提示] 加 --apply 执行删除")
        if not found:
            print(f"  [X] 未找到 session: {session_name}")
        return

    # ── --remove-low-shift: 删除偏移过小的踩踏 ──
    if args.remove_low_shift is not None:
        if not args.data_dir:
            args.data_dir = ["train_data", "new_train_data"]
        thresh = args.remove_low_shift
        all_sessions = scan_sessions(args.data_dir)
        if args.session:
            all_sessions = [s for s in all_sessions if os.path.basename(s) in args.session]

        if not all_sessions:
            print("  [X] 未找到任何 session")
            return

        print(f"\n  偏移阈值: <{thresh:.0f}ms 的踩踏将被删除\n")
        total_removed = 0
        for session_dir in all_sessions:
            result = remove_low_shift_clicks(session_dir, thresh, apply=args.apply)
            if result["total"] == 0:
                continue
            removed = result["removed"]
            total_removed += removed
            if removed > 0:
                mode = "APPLIED" if args.apply else "PREVIEW"
                idx_str = ",".join(str(i) for i in result["removed_indices"])
                shifts_low = [s for s in result["shifts"]
                              if not np.isnan(s) and s < thresh]
                avg_shift = sum(shifts_low) / len(shifts_low) if shifts_low else 0
                print(f"  [{mode}] {result['name']}: "
                      f"删除 {removed}/{result['total']} 次 (偏移<{thresh:.0f}ms, "
                      f"均值{avg_shift:.0f}ms)")
                print(f"         编号: {idx_str}")
        if total_removed == 0:
            print(f"  [OK] 没有偏移 <{thresh:.0f}ms 的踩踏")
        elif not args.apply:
            print(f"\n  共 {total_removed} 次将被删除. 加 --apply 执行.")
        else:
            print(f"\n  已删除 {total_removed} 次. 原文件备份为 .bak")
        return

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
    if args.session:
        sessions = [s for s in sessions if os.path.basename(s) in args.session]
        if not sessions:
            print(f"\n  [X] 未找到匹配的 session: {args.session}")
            return
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
