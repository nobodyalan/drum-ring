"""
================================================================================
 IMU 训练数据预处理 — 滑动窗口切片 + 标签生成 (按 session 独立保存)
================================================================================

 从原始 session 目录加载连续 IMU 录制，生成 (N, C, T) 训练样本。
 每个 session 独立保存，train_model.py 在训练时再决定 train/val 分割。

 标签策略 (实时预测方向):
   - 每个 Arduino click 通过峰值对齐找到 [click, click+100ms] 内 accel_mag 全局最大峰
   - hit_zone = [peak - PRE_MS, peak + POST_MS]
   - 窗口末尾帧落在 hit_zone 内 → label = 1 (击打)
   - 窗口末尾帧落在 hit_zone 外 → label = 0 (背景)

 数据流:
   session 目录 (连续 IMU + 敲击时间戳)
     → load_session() — t0 对齐
     → refine_clicks_to_peaks() — 峰值对齐 [0, +100ms]
     → make_frame_labels() — 逐帧二值标签
     → slice_windows() — 滑动窗口切片 (C, T) 格式
     → 每个 session 独立保存为 {name}_X.npy + {name}_y.npy
     → 全量数据计算 preproc.npz (mean/std)

 ────────────────────────────────────────────────────────────────────────────
 参数:
   --data_dir DIR        原始 session 目录 (必须)
   --save DIR            输出目录 (默认: sliced_data)
   --pre MS              击打前扩展 ms (默认: 80)
   --post MS             击打后扩展 ms (默认: 30)
   --search-pre-ms MS    峰值搜索前向窗口 ms (默认: 0)
   --search-post-ms MS   峰值搜索后向窗口 ms (默认: 100)
   --no-refine           不做峰值对齐，直接用原始 Arduino 时间戳
   --clean               切片前清空输出目录

 用法:
   python slice_dataset.py --data_dir train_data --save sliced_data
   python slice_dataset.py --data_dir train_data --save sliced_data --pre 120 --post 30
   python slice_dataset.py --data_dir train_data --save sliced_data --pre 150 --post -20  # 零延迟
   python slice_dataset.py --data_dir train_data --save sliced_data --no-refine           # 旧行为
   python slice_dataset.py --data_dir train_data --save sliced_data --search-post-ms 120 --clean
"""

import os
import argparse
import numpy as np

# ============================================================================
# 一、切片参数
# ============================================================================

WINDOW_SEC = 0.250          # 窗口长度: 250ms (15帧 @60Hz)
STRIDE_SEC = 0.033          # 滑动步长: ~33ms (训练用, 降低样本冗余)
TARGET_FS = 60              # 手机 IMU 采样率 (ZigSim 60Hz)
PRE_MS = 80                  # 击打前扩展: 80ms (预判零延迟)
POST_MS = 30                 # 击打后扩展: 30ms (落腿冲击训练信号)
WINDOW_FRAMES = int(WINDOW_SEC * TARGET_FS)    # 7帧
STRIDE_FRAMES = int(STRIDE_SEC * TARGET_FS)    # 1帧


# ============================================================================
# 二、数据加载: 从 session 目录读取连续 IMU + 敲击时间戳
# ============================================================================

def load_session(session_dir: str) -> dict | None:
    """
    加载一个采集 session。

    输入: train_data/w_0/ 目录, 包含:
      *_imu_60.npy           → (T, 7) float64
        列: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, timestamp_us]
      *_click_timestamps_60.npy → (N,) float64 (Unix 秒)
      *_t0_60.npy            → 录制起始 PC 绝对时间 (可选, 优先使用)

    返回: {"imu": (T,6), "t_rel": (T,), "click_rel": (N,)}
    """
    files = os.listdir(session_dir)
    imu_files = [f for f in files if "_imu_" in f and f.endswith(".npy")]
    click_files = [f for f in files if "click_timestamps" in f and f.endswith(".npy")]
    if not imu_files or not click_files:
        return None

    imu = np.load(os.path.join(session_dir, imu_files[0]))        # (T, 7) 旧 or (T, 14) 新
    clicks = np.load(os.path.join(session_dir, click_files[0]))   # (N,) Unix秒

    n_raw_cols = imu.shape[1]  # 7=旧格式, 14=新格式
    # 时间戳永远是最后一列
    imu_time = imu[:, -1] / 1_000_000

    if n_raw_cols >= 14:
        # ── 新格式: gyro(3) + accel(3) + quat(4) + grav(3) + ts(1) ──
        imu_data = imu[:, :13].astype(np.float32)  # (T, 13)
        ax, ay, az = imu_data[:, 3], imu_data[:, 4], imu_data[:, 5]
        gx, gy, gz = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]
        grav_x, grav_y, grav_z = imu_data[:, 10], imu_data[:, 11], imu_data[:, 12]

        # 基础衍生特征
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)
        accel_jerk = np.diff(accel_mag, prepend=accel_mag[0])
        gyro_jerk  = np.diff(gyro_mag, prepend=gyro_mag[0])

        # 垂直加速度: 将加速度投影到重力方向
        grav_norm = np.sqrt(grav_x**2 + grav_y**2 + grav_z**2) + 1e-8
        vertical_accel = (ax * grav_x + ay * grav_y + az * grav_z) / grav_norm

        imu_data = np.column_stack([
            imu_data,
            accel_mag.astype(np.float32),
            gyro_mag.astype(np.float32),
            accel_jerk.astype(np.float32),
            vertical_accel.astype(np.float32),
            gyro_jerk.astype(np.float32),
        ])  # (T, 18)
    else:
        # ── 旧格式: gyro(3) + accel(3) + ts(1) ──
        imu_data = imu[:, :6].astype(np.float32)  # (T, 6)
        ax, ay, az = imu_data[:, 3], imu_data[:, 4], imu_data[:, 5]

        # 基础衍生特征 (无重力数据, 无法计算 vertical_accel)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag  = np.sqrt(imu_data[:, 0]**2 + imu_data[:, 1]**2 + imu_data[:, 2]**2)
        accel_jerk = np.diff(accel_mag, prepend=accel_mag[0])

        imu_data = np.column_stack([
            imu_data,
            accel_mag.astype(np.float32),
            gyro_mag.astype(np.float32),
            accel_jerk.astype(np.float32),
        ])  # (T, 9)

    # ── 对齐两个时间轴: 优先 _t0_ 文件, 否则估算 ──
    t0_files = [f for f in files if "_t0_" in f and f.endswith(".npy")]
    if t0_files:
        t0 = np.load(os.path.join(session_dir, t0_files[0])).item()
    elif len(clicks) == 0:
        t0 = imu_time[0]
    else:
        print(f"  [!] {os.path.basename(session_dir)}: 缺少 _t0_ 文件, 使用估算对齐 (旧数据, 可能偏差)")
        t0_estimates = []
        for c in clicks[:min(10, len(clicks))]:
            approx = c - clicks[0] + imu_time[0]
            idx = np.argmin(np.abs(imu_time - approx))
            lo, hi = max(0, idx - 300), min(len(imu_time), idx + 300)
            best = lo + np.argmin(np.abs(imu_time[lo:hi] - approx))
            t0_estimates.append(c - imu_time[best])
        t0 = np.median(t0_estimates)

    click_rel = clicks - t0
    valid = (click_rel >= imu_time[0]) & (click_rel <= imu_time[-1])

    return {"imu": imu_data, "t_rel": imu_time, "click_rel": click_rel[valid]}


# ============================================================================
# 三、标签生成: 从敲击时刻生成逐帧二值标签
# ============================================================================

def refine_clicks_to_peaks(t_rel: np.ndarray, click_rel: np.ndarray,
                            imu_data: np.ndarray,
                            search_pre_ms: int = 0,
                            search_post_ms: int = 100) -> np.ndarray:
    """
    将 Arduino 点击时间戳对齐到 [click, click+120ms] 内的加速度模长全局峰值。

    理由: 真正的踩踏冲击峰总是在 Arduino 触发之后 0-120ms 内出现。
    反弹峰在 150ms+ 之后, 被 120ms 窗口自然排除。
    在这个窄窗内取全局最大是安全的。

    返回 refined_clicks + 打印偏移统计.
    """
    if len(click_rel) == 0:
        return click_rel.copy()

    ax = imu_data[:, 3].astype(np.float64)
    ay = imu_data[:, 4].astype(np.float64)
    az = imu_data[:, 5].astype(np.float64)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    refined = []
    for c in click_rel:
        # 用时间掩码精确控制搜索范围 (避免帧索引对齐误差)
        mask = (t_rel >= c - search_pre_ms / 1000.0) & \
               (t_rel <= c + search_post_ms / 1000.0)
        if mask.any():
            peak_offset = np.argmax(accel_mag[mask])
            refined.append(t_rel[mask][peak_offset])
        else:
            refined.append(c)

    refined = np.array(refined, dtype=click_rel.dtype)
    shifts_ms = (refined - click_rel) * 1000
    print(f"  [峰值对齐] 窗口 [0, +100ms] | "
          f"偏移: mean={shifts_ms.mean():.0f}ms, std={shifts_ms.std():.0f}ms, "
          f"min={shifts_ms.min():.0f}ms, max={shifts_ms.max():.0f}ms")

    return refined


def make_frame_labels(t_rel: np.ndarray, click_rel: np.ndarray) -> np.ndarray:
    """hit_zone = [click - PRE_MS, click + POST_MS], 重叠区域保持 1."""
    y = np.zeros(len(t_rel), dtype=np.float32)
    pre_s = PRE_MS / 1000
    post_s = POST_MS / 1000
    for c in click_rel:
        lo = np.searchsorted(t_rel, c - pre_s)
        hi = np.searchsorted(t_rel, c + post_s)
        if lo < hi:
            y[lo:hi] = 1.0
    return y


# ============================================================================
# 四、滑动窗口切片: 连续 IMU → (N, C, T) 训练样本
# ============================================================================

def slice_windows(imu: np.ndarray, y_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """标签取窗口末尾帧 (训练/推理方向一致). 返回 X (N, C, T), y (N,)."""
    T, W, S = len(imu), WINDOW_FRAMES, STRIDE_FRAMES
    n = max(0, (T - W) // S + 1)

    X_list, y_list = [], []
    for i in range(n):
        start = i * S
        end = start + W
        X_list.append(imu[start:end].T)   # (6, W) — tsai 格式 (C, T)
        y_list.append(y_frame[end - 1])

    if not X_list:
        return np.empty((0, 0, W)), np.empty((0,))
    return np.stack(X_list), np.array(y_list)


# ============================================================================
# 五、处理 & 保存: 每个 session 独立保存
# ============================================================================

def save_preproc_stats(X_all: np.ndarray, save_dir: str):
    """从全量数据计算 per-channel mean/std, 保存 preproc.npz."""
    mean = X_all.mean(axis=(0, 2)).astype(np.float32)
    std = (X_all.std(axis=(0, 2)) + 1e-8).astype(np.float32)

    np.savez_compressed(os.path.join(save_dir, "preproc.npz"),
        mean=mean, std=std,
        window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC,
        target_fs=TARGET_FS, window_frames=WINDOW_FRAMES,
        stride_frames=STRIDE_FRAMES,
        pre_ms=PRE_MS, post_ms=POST_MS,
        channels=X_all.shape[1], classes=2,
    )
    return mean, std


def process_sessions(data_dir: str, save_dir: str,
                     no_refine: bool = False,
                     search_pre_ms: int = 50,
                     search_post_ms: int = 200):
    """扫描所有 session, 逐个切片并独立保存。返回 session 名称列表。"""
    session_dirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    if not session_dirs:
        print("  [X] 未找到任何 session 目录!")
        return []

    os.makedirs(save_dir, exist_ok=True)
    all_X_for_stats = []
    session_names = []
    total_hit, total_bg = 0, 0

    print(f"\n  {'─' * 60}")
    for d in session_dirs:
        path = os.path.join(data_dir, d)
        s = load_session(path)
        if s is None:
            continue

        if not no_refine:
            refined_clicks = refine_clicks_to_peaks(
                s["t_rel"], s["click_rel"], s["imu"],
                search_pre_ms, search_post_ms)
        else:
            refined_clicks = s["click_rel"]
        y_frame = make_frame_labels(s["t_rel"], refined_clicks)
        X, y = slice_windows(s["imu"], y_frame)

        nhit = int(y.sum())
        nbg = X.shape[0] - nhit

        np.save(os.path.join(save_dir, f"{d}_X.npy"), X)
        np.save(os.path.join(save_dir, f"{d}_y.npy"), y.astype(np.int64))

        all_X_for_stats.append(X)
        session_names.append(d)
        total_hit += nhit
        total_bg += nbg

        print(f"  {d:25s}: {X.shape[0]:5d} 窗口, hit={nhit:4d}, bg={nbg:4d}")

    if not session_names:
        print("  [X] 未找到任何有效 session!")
        return []

    # ── 汇总文件: 扫描磁盘上所有已存在的 _X.npy, 确保增量添加不丢旧数据 ──
    all_names = sorted(set(
        f.replace("_X.npy", "")
        for f in os.listdir(save_dir)
        if f.endswith("_X.npy")
    ))
    # 合并本次新增 + 磁盘已有
    merged = sorted(set(session_names) | set(all_names))

    with open(os.path.join(save_dir, "sessions.txt"), "w", encoding="utf-8") as f:
        for name in merged:
            f.write(name + "\n")

    # 从磁盘所有 session 重新计算 preproc 统计量
    all_X = []
    for name in merged:
        X = np.load(os.path.join(save_dir, f"{name}_X.npy"))
        all_X.append(X)
    X_all = np.concatenate(all_X, axis=0)

    mean, std = save_preproc_stats(X_all, save_dir)

    new_count = len(session_names)
    old_count = len(merged) - new_count
    extra = f" (磁盘已有 {old_count} 个, 本次新增 {new_count} 个)" if old_count > 0 else ""
    print(f"\n  全量: {X_all.shape[0]} 窗口, {len(merged)} 个 session{extra}")
    print(f"  preproc.npz: 已更新 (mean/std 来自全部 {len(merged)} 个 session)")

    return session_names


# ============================================================================
# 六、命令行入口
# ============================================================================

def main():
    global PRE_MS, POST_MS

    parser = argparse.ArgumentParser(
        description="IMU 训练数据预处理 — 按 session 独立切片保存"
    )
    parser.add_argument("--data_dir", required=True,
                        help="包含原始 session 子目录的文件夹 (如 train_data/)")
    parser.add_argument("--save", default="sliced_data",
                        help="输出目录 (默认: sliced_data/)")
    parser.add_argument("--pre", type=int, default=80,
                        help="击打前扩展 ms (默认: 80ms)")
    parser.add_argument("--post", type=int, default=30,
                        help="击打后扩展 ms (默认: 30ms)")
    parser.add_argument("--search-pre-ms", type=int, default=0,
                        help="峰值搜索窗口前向 ms (默认: 0)")
    parser.add_argument("--search-post-ms", type=int, default=100,
                        help="峰值搜索窗口后向 ms (默认: 100)")
    parser.add_argument("--no-refine", action="store_true",
                        help="不做峰值对齐，直接用原始 Arduino 时间戳")
    parser.add_argument("--clean", action="store_true",
                        help="切片前清空输出目录 (默认: 增量添加/覆盖)")

    args = parser.parse_args()

    PRE_MS = args.pre
    POST_MS = args.post

    # --clean: 清空输出目录
    if args.clean and os.path.exists(args.save):
        import shutil
        shutil.rmtree(args.save)
        print(f"  已清空: {args.save}/\n")

    print("=" * 55)
    print("  IMU 击打检测 — 数据预处理 (按 session 独立切片)")
    print(f"  窗口: {WINDOW_SEC*1000:.0f}ms ({WINDOW_FRAMES} 帧 @{TARGET_FS}Hz)")
    print(f"  步长: {STRIDE_SEC*1000:.0f}ms ({STRIDE_FRAMES} 帧)")
    print(f"  击打区: [-{PRE_MS}ms, +{POST_MS}ms]")
    print(f"  峰值对齐: {'关闭 (--no-refine)' if args.no_refine else f'开启 (前{args.search_pre_ms}ms / 后{args.search_post_ms}ms)'}")
    print(f"  标签: 二值 (0=背景, 1=击打)")
    print("=" * 55)

    session_names = process_sessions(args.data_dir, args.save,
                                      no_refine=args.no_refine,
                                      search_pre_ms=args.search_pre_ms,
                                      search_post_ms=args.search_post_ms)

    if not session_names:
        print("  [X] 未找到任何训练数据!")
        return

    print(f"\n  预处理完成 -> {args.save}/")
    print(f"  {len(session_names)} 个 session 已独立切片")
    print(f"  下一步: python train_model.py --data_dir {args.save} --save_dir models")
    print(f"  可选参数: --val-sessions ... --neg-sessions ...")


if __name__ == "__main__":
    main()
