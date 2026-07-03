"""
修复旧数据: 为没有 _t0_ 文件的 session 补写正确的 t0
=====================================================
旧数据的 click 时间对齐假设"第一次踩踏 = IMU 第 0 帧", 导致踩踏标注整体偏移。

本脚本通过搜索法找正确 t0:
  - 尝试不同的 offset (第一个 click 在 IMU 时间轴上的位置)
  - 对每个 offset, 计算 click 位置附近 accel_mag 的峰值评分
  - 取评分最高的 offset → 反算 t0
  - 补写 _t0_ 文件, 之后 visualize / slice / train 都会自动使用

用法:
  # 预览模式 (不写文件, 只显示建议的修正量)
  python fix_old_data.py --data_dir train_data

  # 修复模式 (写入 _t0_ 文件)
  python fix_old_data.py --data_dir train_data --fix

  # 也包括 new_data
  python fix_old_data.py --data_dir new_data --fix

  # 修复单个 session
  python fix_old_data.py --data_dir train_data/w_0 --fix
"""

import os
import sys
import argparse
import numpy as np

# ==================== 核心: 搜索最佳 offset ====================

def find_t0(imu: np.ndarray, clicks: np.ndarray, imu_time: np.ndarray) -> tuple[float, float]:
    """
    通过搜索法找到正确的 t0 (第一帧 IMU 的 PC 绝对时间).

    原理: clicks 和 IMU 都使用 time.time() 作为时钟,
         t0 = click_pc_time - click_imu_relative_position
         搜索 click_imu_relative_position (即第一个 click 在 IMU 流中的时间).

    返回: (t0, offset_seconds) 其中 offset_seconds 是第一个 click 的正确 IMU 位置
    """
    # 合加速度幅值 (踩踏主要体现为 accel 波动)
    accel_mag = np.sqrt(imu[:, 3] ** 2 + imu[:, 4] ** 2 + imu[:, 5] ** 2)

    # 搜索范围: 第一个 click 可能从 IMU 开头到末尾区间
    # clicks 的时间跨度 = clicks[-1] - clicks[0], 必须全部落在 IMU 内
    clicks_span = clicks[-1] - clicks[0]
    max_offset = imu_time[-1] - clicks_span
    if max_offset <= 0:
        # click 跨度超出 IMU 长度, 搜索到头
        max_offset = imu_time[-1] * 0.8

    best_score = -np.inf
    best_offset = 0.0

    # 粗搜索: 100ms 步长
    for offset in np.arange(0, max_offset, 0.1):
        score = 0
        valid_count = 0
        for c in clicks:
            # click 在 IMU 时间轴上的正确位置 (假设 offset 为第一个 click 的位置)
            pos = (c - clicks[0]) + offset
            idx = np.searchsorted(imu_time, pos)
            if 0 <= idx < len(imu_time):
                # 取 click 附近 ±3 帧 (~50ms) 的峰值
                lo = max(0, idx - 3)
                hi = min(len(accel_mag), idx + 4)
                score += np.max(accel_mag[lo:hi])
                valid_count += 1
        if valid_count >= len(clicks) * 0.8:  # 至少 80% click 在范围内
            # 归一化: 平均每 click 的峰值响度
            score = score / valid_count
            if score > best_score:
                best_score = score
                best_offset = offset

    # 精搜索: 在最佳 offset 附近 10ms 步长
    fine_start = max(0, best_offset - 0.15)
    fine_end = min(max_offset, best_offset + 0.15)
    for offset in np.arange(fine_start, fine_end, 0.01):
        score = 0
        valid_count = 0
        for c in clicks:
            pos = (c - clicks[0]) + offset
            idx = np.searchsorted(imu_time, pos)
            if 0 <= idx < len(imu_time):
                lo = max(0, idx - 3)
                hi = min(len(accel_mag), idx + 4)
                score += np.max(accel_mag[lo:hi])
                valid_count += 1
        if valid_count >= len(clicks) * 0.8:
            score = score / valid_count
            if score > best_score:
                best_score = score
                best_offset = offset

    # 反算 t0: click_rel[0] = clicks[0] - t0 = best_offset
    # → t0 = clicks[0] - best_offset
    t0 = clicks[0] - best_offset
    return t0, best_offset


# ==================== Session 处理 ====================

def process_session(session_dir: str, fix: bool = False) -> dict | None:
    """处理单个 session, 返回诊断信息"""
    files = os.listdir(session_dir)
    imu_files = [f for f in files if "_imu_" in f and f.endswith(".npy")]
    click_files = [f for f in files if "click_timestamps" in f and f.endswith(".npy")]
    t0_files = [f for f in files if "_t0_" in f and f.endswith(".npy")]

    if not imu_files or not click_files:
        return None

    name = os.path.basename(session_dir)

    # 已有 t0 文件, 跳过
    if t0_files:
        existing_t0 = np.load(os.path.join(session_dir, t0_files[0])).item()
        return {"name": name, "status": "ok", "t0": existing_t0, "offset": None}

    imu = np.load(os.path.join(session_dir, imu_files[0]))
    clicks = np.load(os.path.join(session_dir, click_files[0]))
    imu_time = imu[:, 6] / 1_000_000

    # 没有 click 的 session (如 no_move, shaking), 跳过
    if len(clicks) == 0:
        return {"name": name, "status": "no_clicks", "t0": 0.0, "offset": 0.0}

    # 旧估算 t0 (错误的)
    t0_old_estimates = []
    for c in clicks[:min(10, len(clicks))]:
        approx = c - clicks[0] + imu_time[0]
        idx = np.argmin(np.abs(imu_time - approx))
        lo, hi = max(0, idx - 300), min(len(imu_time), idx + 300)
        best = lo + np.argmin(np.abs(imu_time[lo:hi] - approx))
        t0_old_estimates.append(c - imu_time[best])
    t0_old = np.median(t0_old_estimates)

    # 搜索正确 t0
    t0_new, offset = find_t0(imu, clicks, imu_time)

    correction_s = t0_old - t0_new  # 旧 click_rel 需要加这个值才是正确位置

    result = {
        "name": name,
        "status": "fixed" if fix else "dry_run",
        "t0_old": t0_old,
        "t0_new": t0_new,
        "offset": offset,
        "correction_s": correction_s,
    }

    if fix:
        # 写入 _t0_ 文件 (命名匹配现有的 imu 文件)
        # 从 imu 文件名提取前缀
        imu_name = imu_files[0]
        t0_name = imu_name.replace("_imu_", "_t0_")
        t0_path = os.path.join(session_dir, t0_name)
        np.save(t0_path, np.array([t0_new]))
        print(f"  [OK] {name}: 修正 {correction_s:+.1f}s | "
              f"t0_old={t0_old:.3f} -> t0_new={t0_new:.3f} | "
              f"已写入 {t0_name}")
    else:
        print(f"  [>>] {name}: 修正量={correction_s:+.1f}s | "
              f"第一个click从 IMU {offset:.1f}s 处开始 | "
              f"(预览模式, 用 --fix 写入)")

    return result


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="修复旧数据: 补写正确的 _t0_ 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python fix_old_data.py --data_dir train_data           # 预览, 不写文件
  python fix_old_data.py --data_dir train_data --fix     # 写入 _t0_ 文件
  python fix_old_data.py --data_dir new_data --fix
  python fix_old_data.py --data_dir train_data/w_0 --fix # 修复单个
        """,
    )
    parser.add_argument("--data_dir", required=True,
                        help="数据目录 (train_data, new_data, 或单个 session)")
    parser.add_argument("--fix", action="store_true",
                        help="实际写入 _t0_ 文件 (不加则仅预览)")

    args = parser.parse_args()

    data_dir = args.data_dir

    # 判断是单个 session 还是包含多个子目录的父目录
    if os.path.isdir(data_dir):
        has_imu = any("_imu_" in f for f in os.listdir(data_dir))
    else:
        print(f"[X] 目录不存在: {data_dir}")
        return

    if has_imu:
        # 单个 session
        sessions = [data_dir]
    else:
        # 父目录: 扫描子目录
        sessions = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

    if not sessions:
        print("[X] 未找到 session 目录")
        return

    mode = "修复模式" if args.fix else "预览模式 (加 --fix 写入)"
    print(f"\n{'='*60}")
    print(f"  旧数据 t0 修复 — {mode}")
    print(f"  扫描 {len(sessions)} 个 session...")
    print(f"{'='*60}\n")

    results = []
    corrected = 0
    already_ok = 0
    for s in sessions:
        r = process_session(s, fix=args.fix)
        if r is None:
            print(f"  [!] {os.path.basename(s)}: 跳过 (缺少文件)")
            continue
        if r["status"] == "ok":
            already_ok += 1
        elif r["status"] == "fixed":
            corrected += 1
        results.append(r)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  汇总:")
    print(f"    已有 t0: {already_ok}")
    print(f"    已修正: {corrected}")
    dry_runs = sum(1 for r in results if r.get("status") == "dry_run")
    if dry_runs > 0:
        print(f"    待修正: {dry_runs} (加 --fix 写入)")
        corrections = [r["correction_s"] for r in results if r.get("correction_s")]
        if corrections:
            print(f"    建议修正量: {np.min(corrections):+.1f}s ~ {np.max(corrections):+.1f}s")
    print(f"{'='*60}")

    if not args.fix and dry_runs > 0:
        print(f"\n  确认修正量合理后运行:")
        print(f"    python fix_old_data.py --data_dir {data_dir} --fix")


if __name__ == "__main__":
    main()
