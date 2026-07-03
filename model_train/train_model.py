"""
================================================================================
 IMU 击打检测模型训练 — InceptionTime / FCN / LiteCNN / TST
================================================================================

 场景: 手机放在裤子口袋, 脚踩地面 → 手机 IMU 产生短暂脉冲
       模型学习从 IMU 6轴信号中检测 "是否击打" (二分类)

 数据流:
   session 目录 (连续 IMU + 敲击时间戳)
     → 逐帧生成二值标签 (hit_zone 内=1, 外=0)
     → 滑动窗口切片 (窗口末尾帧标签 = 样本标签)
     → 平衡正负样本 (背景:击打 ≤ 2:1)
     → InceptionTime 训练
     → EarlyStopping (valid_loss 10轮不降即停)
     → 保存最佳模型 + 预处理参数

 用法:
   D:/miniconda3/envs/drum/python.exe train_model.py --data_dir train_data
   D:/miniconda3/envs/drum/python.exe train_model.py --data_dir train_data --model fcn

 参考:
   InceptionTime: Fawaz et al. (2020) https://arxiv.org/abs/1909.04939
   FCN: Wang et al. (2017) "Time Series Classification from Scratch"
"""

import os
import argparse
import numpy as np
from collections import defaultdict

# ============================================================================
# 一、切片参数 (所有值均可通过命令行覆盖)
# ============================================================================

WINDOW_SEC = 0.25           # 窗口长度: 250ms (15帧, 足够看到完整波动)
STRIDE_SEC = 0.033          # 滑动步长: 2帧 ≈ 33ms (训练用, 降低样本冗余)
TARGET_FS = 60              # 手机 IMU 采样率 (ZigSim 60Hz)
PRE_MS = 10                 # 击打前扩展: 10ms (允许微弱预判)
POST_MS = 190               # 击打后扩展: 190ms (覆盖P90, 标注绝大部分可见波动)
WINDOW_FRAMES = int(WINDOW_SEC * TARGET_FS)    # 15帧
STRIDE_FRAMES = int(STRIDE_SEC * TARGET_FS)    # 2帧


# ============================================================================
# 二、数据加载: 从 session 目录读取连续 IMU + 敲击时间戳
# ============================================================================

def load_session(session_dir: str) -> dict | None:
    """
    加载一个采集 session。

    输入: train_data/w_0/ 目录, 包含:
      whr_gesture_0_weak_0_0_imu_60.npy           → (T, 7) float64
        列: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, timestamp_us]
        其中 timestamp_us 是相对于录制起点的微秒数
      whr_gesture_0_weak_0_0_click_timestamps_60.npy → (N,) float64
        Unix 秒级时间戳, 每次 Arduino 检测到敲击的时刻

    关键: IMU 的时间轴是相对微秒, click 是 Unix 绝对秒, 需要对齐。
          IMU[:,6] = (pc_time - t0) * 1e6
          click   = pc_time
          → t0 = click - imu_time_s
          → click_rel = click - t0  (转为相对秒, 与 IMU 时间轴对齐)

    返回: {"imu": (T,6), "t_rel": (T,), "click_rel": (N,)}
    """
    files = os.listdir(session_dir)
    imu_files = [f for f in files if "_imu_" in f and f.endswith(".npy")]
    click_files = [f for f in files if "click_timestamps" in f and f.endswith(".npy")]
    if not imu_files or not click_files:
        return None

    # 加载原始数据
    imu = np.load(os.path.join(session_dir, imu_files[0]))        # (T, 7)
    clicks = np.load(os.path.join(session_dir, click_files[0]))   # (N,) Unix秒

    imu_data = imu[:, :6].astype(np.float32)   # 前6列为 IMU 信号
    imu_time = imu[:, 6] / 1_000_000            # 第7列微秒 → 秒

    # ── 对齐两个时间轴 ──
    # 取前10个 click, 每个找到最近的 IMU 帧, 反算录制起始时间 t0
    # t0 = click_unix_time - imu_relative_time
    t0_estimates = []
    for c in clicks[:min(10, len(clicks))]:
        # 粗定位: 假设第一个 click 对应的 IMU 时间约为 imu_time[0]
        approx = c - clicks[0] + imu_time[0]
        idx = np.argmin(np.abs(imu_time - approx))
        # 在粗定位附近 ±5秒范围细搜索
        lo, hi = max(0, idx - 300), min(len(imu_time), idx + 300)
        best = lo + np.argmin(np.abs(imu_time[lo:hi] - approx))
        t0_estimates.append(c - imu_time[best])
    t0 = np.median(t0_estimates)  # 取中位数, 抗异常值

    # 将 Unix 时间戳转为相对于录制起点的秒数
    click_rel = clicks - t0
    # 过滤超出 IMU 录制范围的 click
    valid = (click_rel >= imu_time[0]) & (click_rel <= imu_time[-1])

    return {
        "imu": imu_data,
        "t_rel": imu_time,
        "click_rel": click_rel[valid],
    }


# ============================================================================
# 三、标签生成: 从敲击时刻生成逐帧二值标签
# ============================================================================

def make_frame_labels(t_rel: np.ndarray, click_rel: np.ndarray) -> np.ndarray:
    """
    生成逐帧二值标签。

    hit_zone = [click - PRE_MS, click + POST_MS]
    时间轴示意 (每格 ≈ 16.7ms @ 60Hz):
      帧:  0   1   2   3   4   5   6   7   8
                    ↑ click (Arduino 检测到敲击)
               [========== hit_zone =========]
           0   0   1   1   1   1   1   0   0  ← 逐帧标签

    多个 click 的 hit_zone 可能重叠, 保持 1。
    """
    y = np.zeros(len(t_rel), dtype=np.float32)
    pre_s = PRE_MS / 1000   # 15ms → 0.015s
    post_s = POST_MS / 1000  # 65ms → 0.065s

    for c in click_rel:
        lo = np.searchsorted(t_rel, c - pre_s)  # hit_zone 起始帧
        hi = np.searchsorted(t_rel, c + post_s)  # hit_zone 结束帧
        if lo < hi:
            y[lo:hi] = 1.0
    return y


# ============================================================================
# 四、滑动窗口切片: 连续 IMU → (N, C, T) 训练样本
# ============================================================================

def slice_windows(imu: np.ndarray, y_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    滑动窗口切片。

    关键设计: 标签取窗口末尾帧, 而非中心帧。
    原因: 推理时模型只见过往数据 (窗口 [t-150ms, t]), 需要预测 t 时刻的状态。
          如果训练时用窗口中心帧做标签, 模型会依赖 "未来" 数据, 与推理不一致。

    训练窗口: [t-150ms, t]  → 标签 = t 时刻的帧标签
    推理窗口: [now-150ms, now] → 预测 now 时刻是否击打
    """
    T, W, S = len(imu), WINDOW_FRAMES, STRIDE_FRAMES
    n = max(0, (T - W) // S + 1)  # 窗口总数

    X_list, y_list = [], []
    for i in range(n):
        start = i * S
        end = start + W
        # tsai 格式: (通道, 时间) = (C, T)
        X_list.append(imu[start:end].T)
        # 标签 = 窗口末尾帧 (即 "此刻")
        y_list.append(y_frame[end - 1])

    if not X_list:
        return np.empty((0, 6, W)), np.empty((0,))
    return np.stack(X_list), np.array(y_list)


# ============================================================================
# 五、构建数据集: 合并所有 session, 平衡正负样本
# ============================================================================

def build_dataset(data_dir: str, val_sessions: list[str] | None = None,
                  neg_sessions: list[str] | None = None):
    """
    加载训练数据，支持两种格式:

      1. 预切片格式: data_dir/ 下有 X.npy + y.npy (slice_dataset.py 的输出)
      2. 原始录制格式: data_dir/ 下有子目录, 每个含 _imu_ 文件

    当 val_sessions 提供时，按 session 级别分割训练/验证集（无窗口泄露）。
    否则返回全部数据，由调用方随机分割。

    当 neg_sessions 提供时，这些 session 的背景样本在平衡下采样时优先保留。

    平衡策略: 仅对训练集下采样背景至击打的 2 倍。验证集保持原始分布。
    """
    train_X, train_y = [], []
    train_neg_mask = []
    val_X, val_y = [], []
    train_hit, train_bg = 0, 0
    val_hit, val_bg = 0, 0

    # ── 格式1: 预切片数据 (X.npy + y.npy) ──
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"  加载预切片数据: {X.shape}, labels={np.unique(y).tolist()}")
        if val_sessions:
            print("  ⚠ 预切片数据不支持 --val-sessions, 全部作为训练集")
        return X, y

    # ── 格式2: 原始录制 (每个子目录是一个 session) ──
    session_dirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    if not session_dirs:
        print("  ❌ 未找到任何 session 目录!")
        return np.empty((0, 6, WINDOW_FRAMES)), np.empty((0,))

    neg_set = set(neg_sessions) if neg_sessions else set()
    print(f"\n  Session 分割:")
    if val_sessions:
        val_set = set(val_sessions)
        print(f"  验证集: {val_sessions}")
        print(f"  训练集: 其余 {len(session_dirs) - len(val_set)} 个 session")
    if neg_sessions:
        print(f"  负样本优先保留: {neg_sessions}")
    print(f"  {'─' * 60}")

    for d in session_dirs:
        path = os.path.join(data_dir, d)
        s = load_session(path)
        if s is None:
            continue

        # 生成逐帧标签 + 滑动窗口切片
        y_frame = make_frame_labels(s["t_rel"], s["click_rel"])
        X, y = slice_windows(s["imu"], y_frame)

        nhit = int(y.sum())
        nbg = X.shape[0] - nhit

        # 按 session 分配到训练集或验证集
        if val_sessions and d in val_set:
            val_X.append(X)
            val_y.append(y)
            val_hit += nhit
            val_bg += nbg
            tag = "[VAL]"
        else:
            train_X.append(X)
            train_y.append(y)
            train_hit += nhit
            train_bg += nbg
            is_neg = neg_sessions and d in neg_set
            train_neg_mask.append(np.full(len(y), is_neg, dtype=bool))
            tag = "[TRAIN-NEG]" if is_neg else "[TRAIN]"

        print(f"  {tag} {d:25s}: {X.shape[0]:5d} 窗口, hit={nhit:4d}, bg={nbg:4d}")

    if not train_X:
        print("  ❌ 未找到任何数据!")
        return np.empty((0, 6, WINDOW_FRAMES)), np.empty((0,))

    # ── 拼接训练集 ──
    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    neg_mask = np.concatenate(train_neg_mask) if train_neg_mask else np.zeros(len(y_train), dtype=bool)

    # ── 平衡训练集 (背景下采样, 优先保留 neg_sessions) ──
    hit_idx = np.where(y_train == 1)[0]
    bg_idx  = np.where(y_train == 0)[0]
    n_keep = min(len(bg_idx), len(hit_idx) * 2)

    neg_bg_idx = bg_idx[neg_mask[bg_idx]]
    other_bg_idx = bg_idx[~neg_mask[bg_idx]]

    print(f"\n  训练集 (平衡前): hit={train_hit}, bg={train_bg} "
          f"(neg={len(neg_bg_idx)}, 普通={len(other_bg_idx)})")
    if len(bg_idx) > n_keep:
        n_neg_keep = min(len(neg_bg_idx), n_keep)
        n_other_keep = n_keep - n_neg_keep
        rng = np.random.RandomState(42)
        keep_neg = rng.choice(neg_bg_idx, n_neg_keep, replace=False) if len(neg_bg_idx) > n_neg_keep else neg_bg_idx
        keep_other = rng.choice(other_bg_idx, n_other_keep, replace=False) if len(other_bg_idx) > n_other_keep else other_bg_idx
        keep_bg = np.concatenate([keep_neg, keep_other])
        keep = np.concatenate([hit_idx, keep_bg])
        keep.sort()
        X_train = X_train[keep]
        y_train = y_train[keep]
        print(f"  训练集 (平衡后): hit={len(hit_idx)}, bg={n_keep} "
              f"(neg={n_neg_keep}, 普通={n_other_keep})")

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

    # ── 验证集 (不采样, 保持原始分布) ──
    if val_X:
        X_val = np.concatenate(val_X)
        y_val = np.concatenate(val_y)
        print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}  "
              f"(hit={val_hit}, bg={val_bg})")
        return X_train, y_train, X_val, y_val

    print(f"  (无独立验证集, 将随机分割)")
    return X_train, y_train


# ============================================================================
# 六、模型定义
# ============================================================================

def build_inception_time(dls, nf: int = 32, depth: int = 4):
    """
    InceptionTime 定制版 — 针对 15帧短窗口优化
    =========================================================================
    相比 tsai 默认版的两大改动:

    1. 卷积核 [3,5,7] 替代 [39,19,9]
       - k=3 →  ~50ms 局部模式 (踩踏上升沿, ~3帧)
       - k=5 →  ~83ms 中等模式
       - k=7 → ~117ms 较长模式 (衰减过程)
       - 15帧窗口下所有核都 100% 用真实数据, 不做 padding 无效运算

    2. 指数加权池化 替代 GAP (全局平均池化)
       - 靠近窗口末尾的帧权重更大 (最后帧权重 ≈ 第一帧的 4.5x)
       - 模型学会"窗口末帧 = 此刻 = 预测目标"
       - 训练/推理方向一致

    参数:
      nf=32:   每支路滤波器数 (4路 = 128输出通道)
      depth=4: Inception 模块堆叠层数 (15帧用4层足够)
    """
    import torch
    import torch.nn as nn

    c_in, c_out = dls.vars, dls.c

    # ── Inception 模块 ──
    class InceptionModule(nn.Module):
        """单层 Inception: 1x1 bottleneck + 多尺度卷积 + maxpool"""
        def __init__(self, ni: int, nf: int, ks: list):
            super().__init__()
            self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False)
            self.convs = nn.ModuleList([
                nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in ks
            ])
            self.maxpool = nn.Sequential(
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(ni, nf, 1, bias=False),
            )
            self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))
            self.act = nn.ReLU()

        def forward(self, x):
            z = self.bottleneck(x)                          # (N, ni, T) → (N, nf, T)
            branches = [conv(z) for conv in self.convs]     # k-scale convs
            branches.append(self.maxpool(x))                 # maxpool branch
            out = torch.cat(branches, dim=1)                # (N, nf*4, T)
            return self.act(self.bn(out))

    # ── Inception Block (depth 层 + 残差连接) ──
    class InceptionBlock(nn.Module):
        def __init__(self, ni: int, nf: int, depth: int, ks: list):
            super().__init__()
            out_ch = nf * (len(ks) + 1)  # 4 branches → nf*4
            self.inceptions = nn.ModuleList([
                InceptionModule(ni if i == 0 else out_ch, nf, ks)
                for i in range(depth)
            ])
            self.shortcut = nn.Sequential(
                nn.Conv1d(ni, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            self.act = nn.ReLU()

        def forward(self, x):
            residual = self.shortcut(x)
            for inc in self.inceptions:
                x = inc(x)
            return self.act(x + residual)

    # ── 完整模型: Backbone + 加权池化 + 分类头 ──
    ks = [3, 5, 7]
    out_ch = nf * (len(ks) + 1)  # 128

    class WeightedInceptionTime(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = InceptionBlock(c_in, nf, depth, ks)
            # 可学习的时序权重 (初始化为指数递增)
            max_t = 15  # window_frames
            init_w = torch.exp(torch.linspace(0, 2.0, max_t))  # 1.0 → 7.4
            self.time_weights = nn.Parameter(init_w)
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(out_ch, c_out),
            )

        def forward(self, x):
            # x: (N, C, T)  e.g. (64, 6, 15)
            feat = self.backbone(x)                        # → (N, 128, T)
            T = feat.shape[-1]

            # 可学习加权池化: softmax 归一化, 模型自动学最优权重
            w = self.time_weights[:T]                      # (T,)
            w = torch.softmax(w, dim=0)                    # 归一化, sum=1

            pooled = (feat * w.view(1, 1, -1)).sum(dim=-1)  # (N, out_ch)
            return self.head(pooled)

    model = WeightedInceptionTime()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [模型] 定制 InceptionTime: ks={ks}, depth={depth}, nf={nf}")
    print(f"  [模型] 参数量: {param_count:,} (vs tsai默认 456K)")
    print(f"  [模型] 池化: 指数加权 (末尾帧 ~4.5x)")
    return model


def build_fcn(dls):
    """
    FCN (Fully Convolutional Network) — 简单 baseline
    =========================================================================
    三层 1D 卷积 + BatchNorm + ReLU + Global Average Pooling + Softmax.

    每层: Conv1d(kernel=3) → BN → ReLU
    参数量约 2 万, 适合快速实验和移动端部署。

    参考文献: Wang et al. (2017)
    """
    from tsai.all import FCN
    return FCN(dls.vars, dls.c)


def build_lite_cnn(dls):
    """
    LiteCNN — 专为 9帧超短窗口设计的轻量模型
    =========================================================================
    使用深度可分离卷积 (Depthwise Separable Conv) 大幅减少参数量。

    深度可分离卷积 = Depthwise Conv (逐通道空间滤波)
                    + Pointwise Conv (1×1 跨通道融合)

    2 层深度可分离卷积 + 全局平均池化 → 约 5,000 参数。
    推理极快 (<1ms), 适合部署到移动端或嵌入式设备。
    """
    import torch
    import torch.nn as nn

    class LiteCNN(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            # 第一层: 深度可分离卷积, 6通道 → 16通道
            self.conv1 = nn.Sequential(
                nn.Conv1d(c_in, c_in, kernel_size=3, groups=c_in, padding=1),  # Depthwise
                nn.Conv1d(c_in, 16, kernel_size=1),                             # Pointwise
                nn.BatchNorm1d(16),
                nn.ReLU(),
            )
            # 第二层: 深度可分离卷积, 16通道 → 32通道
            self.conv2 = nn.Sequential(
                nn.Conv1d(16, 16, kernel_size=3, groups=16, padding=1),  # Depthwise
                nn.Conv1d(16, 32, kernel_size=1),                        # Pointwise
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            # 全局平均池化: (32, 9) → (32, 1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            # 分类头: 32 → 2
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),          # 防过拟合
                nn.Linear(32, c_out),
            )

        def forward(self, x):
            x = self.conv1(x)   # (N, 6, 9)  → (N, 16, 9)
            x = self.conv2(x)   # (N, 16, 9) → (N, 32, 9)
            x = self.pool(x)    # (N, 32, 9) → (N, 32, 1)
            return self.head(x) # (N, 32)    → (N, 2)

    return LiteCNN(dls.vars, dls.c)


def build_tst(dls):
    """
    TST (Time Series Transformer) — 基于自注意力机制
    =========================================================================
    用 Multi-Head Self-Attention 代替卷积捕获时序依赖。

    注意: 9帧窗口太短, Transformer 的位置编码和长距离建模优势不明显,
          且参数量大 (~20万), 容易在小数据集上过拟合。
          此处仅作为对比实验。
    """
    from tsai.all import TST
    return TST(dls.vars, dls.c, d_model=64, n_heads=4, fc_dropout=0.3)


# ============================================================================
# 七、训练流程
# ============================================================================

def train(X: np.ndarray, y: np.ndarray, model_name: str = "inception",
          epochs: int = 70, batch_size: int = 64, lr: float = 1e-3,
          save_dir: str = "models", focal_gamma: float = 0,
          X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
    """
    完整训练流程:
      1. 标准化 (每通道 Z-score)
      2. 训练/验证分割:
         - 提供了 X_val/y_val → 直接使用 (session 级独立验证集)
         - 未提供 → 窗口级随机分割 (旧行为, 可能泄露)
      3. InceptionTime / FCN / LiteCNN / TST
      4. CrossEntropyLoss / AdditiveMarginLoss + AdamW (One-Cycle 学习率)
      5. EarlyStopping: valid_loss 连续 10 轮不降 → 停止
      6. 保存验证集 loss 最低的模型 + 预处理参数
      7. 评估: 分类报告 + 混淆矩阵 + 训练曲线
    """
    import torch
    import torch.nn as nn
    from tsai.all import (
        Learner, accuracy, RandomSplitter, IndexSplitter, get_ts_dls
    )
    from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback
    import fastai.torch_core

    # ── GPU 检测 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  [GPU] CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print(f"\n  [CPU] 使用 CPU 训练")
    fastai.torch_core.default_device(device)

    y = y.astype(np.int64)

    # ── 7.1 标准化 ──
    # 用训练集的 mean/std, 验证集也照此标准化
    mean = X.mean(axis=(0, 2), keepdims=True)   # (1, 6, 1)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    X = (X - mean) / std

    if X_val is not None and y_val is not None:
        y_val = y_val.astype(np.int64)
        X_val = (X_val - mean) / std

    # ── 7.2 训练/验证分割 ──
    if X_val is not None:
        # Session 级独立验证集: 拼接后按长度分
        X_all = np.concatenate([X, X_val], axis=0)
        y_all = np.concatenate([y, y_val], axis=0)
        n_train = len(X)
        train_idx = list(range(n_train))
        valid_idx = list(range(n_train, len(X_all)))
        splits = (train_idx, valid_idx)
        print(f"\n  验证方式: Session 级独立验证集 ({n_train} train / {len(valid_idx)} val)")
    else:
        X_all, y_all = X, y
        splits = RandomSplitter(valid_pct=0.2, seed=42)(y_all)
        print(f"\n  验证方式: 窗口级随机20%分割 (可能跨 session 泄露!)")
        print(f"  💡 建议用 --val-sessions 指定独立验证 session")

    # ── 7.3 构建 tsai DataLoaders ──
    dls = get_ts_dls(X_all, y_all, splits=splits, bs=batch_size)
    dls.c = 2  # 强制二分类 (tsai 1.0.1 默认将二值标签视为 1 类输出, 需显式设 2)

    # ── 7.4 选择模型 ──
    models = {
        "inception": build_inception_time,
        "fcn": build_fcn,
        "lite": build_lite_cnn,
        "tst": build_tst,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(models.keys())}")
    model = models[model_name](dls)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  模型: {model_name}, 参数量: {param_count:,}")
    print(f"  窗口: {WINDOW_SEC*1000:.0f}ms ({WINDOW_FRAMES}帧 @{TARGET_FS}Hz)")
    print(f"  训练集: {len(splits[0])}, 验证集: {len(splits[1])}")

    # ── 7.5 Learner (训练器) ──
    # 构建 loss 函数
    if focal_gamma > 0:
        from tsai.all import FocalLoss
        loss_func = FocalLoss(gamma=focal_gamma)
        print(f"\n  [Loss] tsai FocalLoss  (gamma={focal_gamma})")
    else:
        loss_func = nn.CrossEntropyLoss()
        print(f"\n  [Loss] CrossEntropyLoss")

    learn = Learner(
        dls, model,
        loss_func=loss_func,
        metrics=[accuracy],
        cbs=[
            SaveModelCallback(monitor="valid_loss"),
            EarlyStoppingCallback(monitor="valid_loss", patience=10),
        ],
    )
    learn.model.to(device)

    # ── 7.6 One-Cycle 学习率策略 ──
    # 从 lr/10 升至 lr, 再降至 lr/1000, 帮助模型快速收敛并逃离局部最优
    learn.fit_one_cycle(epochs, lr_max=lr)

    # ── 7.7 评估 ──
    # Best Model 的指标从 SaveModelCallback 加载后的 learner.validate() 获取
    # learn.final_record 顺序在 fastai 版本间不一致, 直接用验证集预测结果算
    valid_dl = dls.valid
    preds, targs = learn.get_preds(dl=valid_dl)
    pred_labels = preds.argmax(dim=1)
    valid_acc = (pred_labels == targs).float().mean().item()
    valid_loss = learn.final_record[0]  # 至少 loss 是正确的
    print(f"\n  最佳验证准确率: {valid_acc:.3f} ({valid_acc*100:.1f}%)")

    # 在验证集上计算分类报告

    from sklearn.metrics import classification_report, confusion_matrix
    print("\n  分类报告:")
    print(classification_report(targs, pred_labels, target_names=["背景", "击打"]))

    # 混淆矩阵: 每个格子 = 样本数
    cm = confusion_matrix(targs, pred_labels)
    print(f"  混淆矩阵:\n"
          f"    TN (正确识别为非击打) = {cm[0,0]:4d}\n"
          f"    FP (误报, 非击打被判为击打) = {cm[0,1]:4d}\n"
          f"    FN (漏检, 击打被判为非击打) = {cm[1,0]:4d}\n"
          f"    TP (正确检出击打) = {cm[1,1]:4d}")

    # ── 7.8 保存模型 ──
    # 同时保存预处理参数 (mean, std), 推理时需要用相同方式标准化
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"hit_detector_{model_name}.pth")
    torch.save({
        "model_state": model.state_dict(),    # 模型权重
        "model_name": model_name,             # 架构名 (推理时重建用)
        "mean": mean,                          # 标准化均值 (6,)
        "std": std,                            # 标准化标准差 (6,)
        "window_sec": WINDOW_SEC,              # 窗口长度
        "window_frames": WINDOW_FRAMES,        # 窗口帧数
        "target_fs": TARGET_FS,                # 采样率
        "channels": 6,                         # 输入通道数
        "classes": 2,                          # 输出类别数
    }, model_path)
    print(f"\n  模型已保存: {model_path}")

    # ── 7.9 绘制训练曲线 ──
    try:
        import matplotlib.pyplot as plt
        rec = learn.recorder

        # 用 fastai 内置方法拿到 loss 历史
        train_loss = [v[0] for v in rec.values]
        valid_loss_vals = [v[1] for v in rec.values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs_arr = range(1, len(train_loss) + 1)

        ax1.plot(epochs_arr, train_loss, "b-", label="Train Loss")
        ax1.plot(epochs_arr, valid_loss_vals, "r-", label="Valid Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)

        # accuracy — rec.values[i] 可能只有2个元素 (无metrics)
        if len(rec.values[0]) >= 3:
            acc = [v[2] for v in rec.values]
            ax2.plot(epochs_arr, acc, "g-", label="Accuracy")
        else:
            ax2.text(0.5, 0.5, "accuracy 不可用\n(loss 曲线正常)", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Training Curve — {model_name}")
        plt.tight_layout()
        path = os.path.join(save_dir, f"curve_{model_name}.png")
        plt.savefig(path, dpi=100)
        print(f"  训练曲线: {path}")
        plt.show()
    except Exception as e:
        print(f"  [WARN] 绘制曲线失败: {e}")

    return learn, model


# ============================================================================
# 八、命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IMU 击打检测模型训练 (InceptionTime/FCN/LiteCNN/TST)"
    )
    parser.add_argument("--data_dir", default="train_data",
                        help="包含 session 子目录的数据目录")
    parser.add_argument("--model", default="inception",
                        choices=["inception", "fcn", "lite", "tst"],
                        help="模型架构: inception (推荐), fcn (快速baseline), "
                             "lite (极轻量), tst (Transformer对比)")
    parser.add_argument("--epochs", type=int, default=70,
                        help="最大训练轮数 (早停机制会在 loss 不降时提前停止)")
    parser.add_argument("--bs", type=int, default=64,
                        help="批次大小 (显存不够可减小)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率 (One-Cycle 策略会自动调整)")
    parser.add_argument("--save_dir", default="models",
                        help="模型保存目录")
    parser.add_argument("--focal-loss", type=float, default=0,
                        help="Focal Loss gamma 参数, 0=标准 CE, 2=推荐值. "
                             "容易样本自动降权, 训练聚焦于模棱两可的边界样本")
    parser.add_argument("--val-sessions", type=str, default="",
                        help="逗号分隔的验证集 session 名, 例如 'w_5,f_3,w_7'. "
                             "这些 session 不参与训练, 用于评估泛化能力. "
                             "留空则随机窗口分割 (可能泄露).")
    parser.add_argument("--neg-sessions", type=str, default="",
                        help="逗号分隔的负样本 session 名, 例如 'no_move_0,shaking_0'. "
                             "背景样本在下采样时优先保留, 确保纯负样本不被随机丢弃.")

    args = parser.parse_args()

    val_sessions = [s.strip() for s in args.val_sessions.split(",") if s.strip()] \
                   if args.val_sessions else None
    neg_sessions = [s.strip() for s in args.neg_sessions.split(",") if s.strip()] \
                   if args.neg_sessions else None

    print("=" * 55)
    print("  IMU 击打检测 — 训练")
    print(f"  窗口: {WINDOW_SEC*1000:.0f}ms ({WINDOW_FRAMES}帧 @{TARGET_FS}Hz)")
    print(f"  击打区: [-{PRE_MS}ms, +{POST_MS}ms]")
    print(f"  模型: {args.model}")
    if args.focal_loss > 0:
        print(f"  Loss:  FocalLoss (gamma={args.focal_loss})")
    if val_sessions:
        print(f"  验证:  session 级独立 ({', '.join(val_sessions)})")
    if neg_sessions:
        print(f"  负样本优先: ({', '.join(neg_sessions)})")
    print("=" * 55)

    # 加载数据
    result = build_dataset(args.data_dir, val_sessions, neg_sessions)
    if isinstance(result, tuple) and len(result) == 4:
        X_train, y_train, X_val, y_val = result
    else:
        X_train, y_train = result
        X_val, y_val = None, None

    if len(X_train) == 0:
        print("  ❌ 无数据")
        return

    # 训练
    train(X_train, y_train, args.model, args.epochs, args.bs, args.lr, args.save_dir,
          focal_gamma=args.focal_loss, X_val=X_val, y_val=y_val)


if __name__ == "__main__":
    main()
