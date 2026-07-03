"""
全数据滑动窗口切片 + 标签生成
==============================
从连续 IMU 录制中生成 (N, C, T) 训练样本，适配 tsai。

标签策略 (实时预测):
  - 每个 click 前后扩展为 hit_zone: [click - PRE_MS, click + POST_MS]
  - 窗口末尾落在 hit_zone 内 → label = 击打类别 (训练/推理方向一致)
  - 窗口末尾落在 hit_zone 外 → label = 0 (背景)

训练 vs 推理:
  训练: 窗口 [t-300ms, t] → 预测 t 时刻是否击打 (只见过往)
  推理: 每帧都取过去 300ms → 实时输出概率 (与训练完全一致)

用法:
  # 自动扫描 (从目录名推断 weak/strong)
  python slice_dataset.py --data_dir train_data --save sliced_data

  # 自定义窗口 (训练用 stride=2帧, 推理用 stride=1帧)
  python slice_dataset.py --data_dir train_data --window 0.3 --stride 0.033 --save sliced_data
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# ===================== 切片参数 (可调) =====================
WINDOW_SEC = 0.25       # 窗口长度 (秒) — 250ms 足够看到完整波动
STRIDE_SEC = 1/60        # 默认步长 1帧
TARGET_FS = 60           # 目标采样率
PRE_MS = 10              # 击打前扩展 (ms) — 允许微弱预判
POST_MS = 190            # 击打后扩展 (ms) — 覆盖P90, 标注绝大部分可见波动

WINDOW_FRAMES = int(WINDOW_SEC * TARGET_FS)
STRIDE_FRAMES = int(STRIDE_SEC * TARGET_FS)

# ===================== 核心函数 =====================
def load_session(session_dir: str, label: int) -> dict | None:
    """
    加载一个 session 的数据。
    返回 {"imu": (T, 6), "t_rel": (T,) 相对秒, "click_rel": (N,) 相对秒, "label": int}
    """
    # 找文件
    files = os.listdir(session_dir)
    imu_file = [f for f in files if "_imu_" in f and f.endswith(".npy")]
    click_file = [f for f in files if "click_timestamps" in f and f.endswith(".npy")]

    if not imu_file or not click_file:
        print(f"  [!] {session_dir}: 缺少文件, 跳过")
        return None

    imu = np.load(os.path.join(session_dir, imu_file[0]))   # (T, 7)
    clicks = np.load(os.path.join(session_dir, click_file[0]))  # (N,) Unix 秒

    # IMU 数据通道
    imu_data = imu[:, :6].astype(np.float32)  # (T, 6)
    imu_us = imu[:, 6]                        # 微秒 (相对录制起点)

    # 估算 t0: 取前10个 click, 找最近 IMU 帧, 算 t0 = click - imu_time_s
    imu_time_s = imu_us / 1_000_000  # 相对秒

    # 优先使用录制时保存的真实 t0
    t0_files = [f for f in files if "_t0_" in f and f.endswith(".npy")]
    if t0_files:
        t0 = np.load(os.path.join(session_dir, t0_files[0])).item()
    else:
        print(f"  [!] {os.path.basename(session_dir)}: 缺少 _t0_ 文件, 使用估算对齐 (旧数据, 可能偏差)")
        t0_estimates = []
        for c in clicks[:min(10, len(clicks))]:
            approx = c - clicks[0] + imu_time_s[0]
            idx = np.argmin(np.abs(imu_time_s - approx))
            search_start = max(0, idx - 300)
            search_end = min(len(imu_time_s), idx + 300)
            local_idx = search_start + np.argmin(
                np.abs(imu_time_s[search_start:search_end] - approx))
            t0_estimates.append(c - imu_time_s[local_idx])
        t0 = np.median(t0_estimates)

    # 将 click 转为相对秒
    click_rel = clicks - t0  # (N,) 相对录制起点的秒数

    # 过滤超出 IMU 范围的 click
    valid = (click_rel >= imu_time_s[0]) & (click_rel <= imu_time_s[-1])
    click_rel = click_rel[valid]
    print(f"  {os.path.basename(session_dir)}: IMU={imu_data.shape}, clicks={len(click_rel)} (有效={valid.sum()}/{len(valid)}), t0={t0:.3f}")

    return {
        "imu": imu_data,
        "t_rel": imu_time_s,
        "click_rel": click_rel,
        "label": label,
    }


MIN_PEAK_SNR = 1.5          # 最小峰值 SNR: IMU 峰值必须 > 基线 1.5x 才算有效击打

def filter_clicks_by_imu(t_rel: np.ndarray, imu: np.ndarray,
                         click_rel: np.ndarray,
                         min_snr: float = MIN_PEAK_SNR,
                         search_ms: float = 150) -> np.ndarray:
    """
    过滤 click: 只保留在 [click, click+search_ms] 区间内 IMU 有明显峰值的击打。
    无明显 IMU 响应的 click 不产生标签, 避免噪声污染训练数据。

    返回: 过滤后的 click_rel
    """
    accel_mag = np.sqrt(imu[:, 3]**2 + imu[:, 4]**2 + imu[:, 5]**2)
    keep = []
    for c in click_rel:
        # 搜索窗口
        sm = (t_rel >= c) & (t_rel <= c + search_ms / 1000)
        bm = (t_rel >= c - 0.15) & (t_rel <= c - 0.02)
        if sm.sum() < 2 or bm.sum() < 2:
            continue
        baseline = accel_mag[bm].mean()
        peak = accel_mag[sm].max()
        if peak / (baseline + 1e-8) >= min_snr:
            keep.append(c)
    return np.array(keep)


def build_labels(t_rel: np.ndarray, click_rel: np.ndarray,
                 label_val: int) -> np.ndarray:
    """
    根据 click 时间生成逐帧标签。
    hit_zone = [click - PRE_MS, click + POST_MS] → label = label_val
    其余 → label = 0
    多个 click 的 hit_zone 重叠时取最大值
    """
    y = np.zeros(len(t_rel), dtype=np.int32)
    pre_s = PRE_MS / 1000
    post_s = POST_MS / 1000

    for c in click_rel:
        lo = np.searchsorted(t_rel, c - pre_s, side="left")
        hi = np.searchsorted(t_rel, c + post_s, side="right")
        y[lo:hi] = max(y[lo:hi].max(), label_val) if len(y[lo:hi]) > 0 else label_val

    return y


def slice_windows(imu: np.ndarray, y_frame: np.ndarray, t_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    滑动窗口切片。
    标签取窗口末尾帧的标签 → 训练与推理方向一致
    (推理时只见过往数据, 预测 "此刻" 是否击打)

    返回 X (M, 6, W), y (M,)
    """
    T = len(imu)
    W = WINDOW_FRAMES
    S = STRIDE_FRAMES

    n_windows = max(0, (T - W) // S + 1)

    X_list = []
    y_list = []

    for i in range(n_windows):
        start = i * S
        end = start + W  # exclusive
        window = imu[start:end]  # (W, 6)
        X_list.append(window.T)  # → (6, W)  tsai 格式: (C, T)

        # 标签 = 窗口末尾帧 (即 "此刻")
        y_list.append(y_frame[end - 1])

    if not X_list:
        return np.empty((0, 6, W), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = np.stack(X_list, axis=0)  # (M, 6, W)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def slice_session(session: dict) -> tuple[np.ndarray, np.ndarray]:
    """对一个 session 生成 frame labels + 滑动窗口切片"""
    y_frame = build_labels(session["t_rel"], session["click_rel"], session["label"])
    X, y = slice_windows(session["imu"], y_frame, session["t_rel"])
    return X, y


# ===================== 主流程 =====================

def process_sessions(session_dirs: list[str], labels: list[int],
                     save_dir: str, balance: bool = True,
                     bg_ratio: float = 2.0):
    """
    处理多个 session, 合并并保存。

    balance: 是否对背景类下采样 (避免类别极度不平衡)
    bg_ratio: 背景:击打的最大比例
    """
    os.makedirs(save_dir, exist_ok=True)

    all_X, all_y = [], []
    session_ids = []
    class_counts = defaultdict(int)

    for session_dir, label in zip(session_dirs, labels):
        print(f"\n加载 {session_dir} (label={label})")
        session = load_session(session_dir, label)
        if session is None:
            continue

        X, y = slice_session(session)
        print(f"  切片: {X.shape[0]} 个窗口, "
              f"hit={np.sum(y > 0)}, bg={np.sum(y == 0)}")

        all_X.append(X)
        all_y.append(y)
        session_ids.extend([len(session_ids)] * len(y))
        for v in y:
            class_counts[v] += 1

    if not all_X:
        print("[X] 没有数据!")
        return

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # ── 类别平衡 (下采样背景) ──
    if balance:
        hit_mask = y_all > 0
        bg_mask = y_all == 0
        n_hit = hit_mask.sum()
        n_bg_max = int(n_hit * bg_ratio)

        print(f"\n类别分布 (平衡前): {dict(class_counts)}")
        print(f"  击打: {n_hit}, 背景: {bg_mask.sum()}")

        if bg_mask.sum() > n_bg_max:
            bg_indices = np.where(bg_mask)[0]
            keep_bg = np.random.RandomState(42).choice(
                bg_indices, size=n_bg_max, replace=False)
            keep = np.concatenate([np.where(hit_mask)[0], keep_bg])
            keep.sort()
            X_all = X_all[keep]
            y_all = y_all[keep]
            # 重新统计
            class_counts.clear()
            for v in y_all:
                class_counts[v] += 1

    print(f"\n类别分布 (最终): {dict(sorted(class_counts.items()))}")

    # ── 保存 ──
    np.save(os.path.join(save_dir, "X.npy"), X_all)
    np.save(os.path.join(save_dir, "y.npy"), y_all)
    np.savez_compressed(os.path.join(save_dir, "info.npz"),
        window_sec=WINDOW_SEC,
        stride_sec=STRIDE_SEC,
        target_fs=TARGET_FS,
        window_frames=WINDOW_FRAMES,
        stride_frames=STRIDE_FRAMES,
        pre_ms=PRE_MS,
        post_ms=POST_MS,
        class_distribution=np.array(list(class_counts.items())),
        ch_names=np.array(["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]),
    )

    print(f"\n✅ 已保存: {save_dir}/")
    print(f"   X.npy — {X_all.shape} (samples={X_all.shape[0]}, channels={X_all.shape[1]}, timesteps={X_all.shape[2]})")
    print(f"   y.npy — {y_all.shape}, classes={np.unique(y_all).tolist()}")
    print(f"\ntsai 加载示例:")
    print(f"   from tsai.all import *")
    print(f"   X, y, splits = get_classification_data('{save_dir}', split_data=False)")


# ===================== 自动发现 sessions =====================

def auto_find_sessions(data_dir: str) -> tuple[list[str], list[int]]:
    """
    自动扫描 train_data/ 下的所有 session 目录,
    从文件名推断 label。
    """
    sessions = []
    labels = []
    # label 映射: weak→1, strong→2 (0=背景)
    label_map = {"weak": 1, "strong": 2, "w": 1, "s": 2}

    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if not os.path.isdir(full):
            continue
        # 检查是否有 _imu_ 文件
        has_imu = any("_imu_" in f for f in os.listdir(full))
        if not has_imu:
            continue

        # 从目录名推断 label
        d_lower = d.lower()
        label = 0
        for key, val in label_map.items():
            if key in d_lower:
                label = val
                break
        if label == 0:
            print(f"  [!] {d}: 无法从目录名推断 label, 跳过 (请在目录名中包含 weak/strong)")
            continue

        sessions.append(full)
        labels.append(label)
        print(f"  发现: {d} → label={label}")

    return sessions, labels


# ===================== CLI =====================

def main():
    global WINDOW_SEC, STRIDE_SEC, PRE_MS, POST_MS, WINDOW_FRAMES, STRIDE_FRAMES

    parser = argparse.ArgumentParser(
        description="全数据滑动窗口切片 + 标签生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动扫描 train_data/ 下所有 session
  python slice_dataset.py --data_dir train_data --save sliced_data

  # 指定 session
  python slice_dataset.py --sessions train_data/w_0 train_data/s_0 --labels 1 2 --save sliced_data

  # 自定义窗口参数
  python slice_dataset.py --data_dir train_data --window 0.5 --stride 0.05 --pre 150 --post 300 --save sliced_data
        """
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="自动扫描目录")
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="手动指定 session 目录列表")
    parser.add_argument("--labels", nargs="+", type=int, default=None,
                        help="对应的标签列表 (需与 --sessions 一一对应)")
    parser.add_argument("--save", "-s", type=str, default="sliced_data",
                        help="输出目录")
    parser.add_argument("--window", type=float, default=WINDOW_SEC,
                        help=f"窗口长度秒 (默认 {WINDOW_SEC}s = {int(WINDOW_SEC*1000)}ms)")
    parser.add_argument("--stride", type=float, default=STRIDE_SEC,
                        help=f"滑动步长秒 (默认 {STRIDE_SEC:.3f}s = 1帧; 训练可用 0.033=2帧)")
    parser.add_argument("--pre", type=int, default=PRE_MS,
                        help=f"击打前扩展 ms (默认 {PRE_MS}ms, 让模型学会预判)")
    parser.add_argument("--post", type=int, default=POST_MS,
                        help=f"击打后扩展 ms (默认 {POST_MS}ms, 覆盖击打瞬间)")
    parser.add_argument("--no-balance", action="store_true",
                        help="不对背景类下采样")
    parser.add_argument("--bg-ratio", type=float, default=2.0,
                        help="背景:击打最大比例 (默认 2.0)")

    args = parser.parse_args()

    # 用命令行参数覆盖模块级默认值
    WINDOW_SEC = args.window
    STRIDE_SEC = args.stride
    PRE_MS = args.pre
    POST_MS = args.post
    WINDOW_FRAMES = int(WINDOW_SEC * TARGET_FS)
    STRIDE_FRAMES = int(STRIDE_SEC * TARGET_FS)

    # ── 确定 sessions 和 labels ──
    if args.sessions:
        sessions = args.sessions
        if args.labels and len(args.labels) == len(sessions):
            labels = args.labels
        else:
            print("[X] --sessions 和 --labels 数量不匹配")
            return
    elif args.data_dir:
        sessions, labels = auto_find_sessions(args.data_dir)
        if not sessions:
            print(f"[X] {args.data_dir} 下未找到有效 session")
            return
    else:
        print("[X] 请指定 --data_dir 或 --sessions")
        return

    print(f"\n{'='*60}")
    print(f"  滑动窗口切片")
    print(f"  窗口: {WINDOW_SEC}s ({WINDOW_FRAMES}帧)")
    print(f"  步长: {STRIDE_SEC}s ({STRIDE_FRAMES}帧)")
    print(f"  重叠: {WINDOW_SEC - STRIDE_SEC}s")
    print(f"  Hit zone: [-{PRE_MS}ms, +{POST_MS}ms]")
    print(f"  类别: 0=背景, 1=weak, 2=strong")
    print(f"{'='*60}")

    process_sessions(sessions, labels, args.save,
                     balance=not args.no_balance,
                     bg_ratio=args.bg_ratio)


if __name__ == "__main__":
    main()
