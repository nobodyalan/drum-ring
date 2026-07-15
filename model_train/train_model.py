"""
================================================================================
 IMU 击打检测模型训练 — InceptionTime / FCN / LiteCNN / TST
================================================================================

 场景: 手机放在裤子口袋, 脚踩地面 → 手机 IMU 产生短暂脉冲
       模型学习从 IMU 6轴信号中检测 "是否击打" (二分类)

 数据流:
   sliced_data/ → load_session_slices() → 平衡采样 → 训练 → 保存模型

 ────────────────────────────────────────────────────────────────────────────
 参数:
   --data_dir DIR        切片数据目录 (默认: sliced_data)
   --model MODEL         模型架构: inception/fcn/lite/tst (默认: inception)
   --epochs N            最大训练轮数 (默认: 50)
   --bs N                批次大小 (默认: 64)
   --lr LR               初始学习率 (默认: 5e-4)
   --save_dir DIR        模型保存目录 (默认: models)
   --focal-loss GAMMA    Focal Loss gamma, 0=标准CE (默认: 0)
   --val-sessions S1,S2   验证集 session 名 (逗号分隔, 不参与训练)
   --neg-sessions S1,S2   负样本 session 名 (背景采样时优先保留)
   --aug-rotate DEG      IMU 旋转增强角度 (默认: 0, 建议 10-15)
   --dropout P           Dropout 概率 (默认: 0.4)
   --weight-decay WD     L2 权重衰减 (默认: 1e-3)
   --no-augment          禁用数据增强
   --balance-mode MODE   负例平衡: posthit/random (默认: posthit)

 用法:
   python train_model.py --data_dir sliced_data
   python train_model.py --data_dir sliced_data --model fcn --epochs 100
   python train_model.py --data_dir sliced_data --neg-sessions neg_a_0,neg_b_0 --no-augment
   python train_model.py --data_dir sliced_data --val-sessions w_5,f_3 --aug-rotate 10
   python train_model.py --data_dir sliced_data --balance-mode random --dropout 0.5
   python train_model.py --data_dir sliced_data --focal-loss 2 --lr 1e-3

 参考:
   InceptionTime: Fawaz et al. (2020) https://arxiv.org/abs/1909.04939
   FCN: Wang et al. (2017) "Time Series Classification from Scratch"
"""

import os
import argparse
import numpy as np

# ============================================================================
# 一、数据加载: 从预处理目录加载独立 session 切片 + 汇总
# ============================================================================

def load_session_slices(data_dir: str,
                        val_sessions: list[str] | None = None,
                        neg_sessions: list[str] | None = None,
                        balance_mode: str = "posthit") -> dict:
    """
    加载 slice_dataset.py 按 session 独立切片的数据, 按需汇总 train/val。

    格式检测顺序:
      1. sessions.txt — 新格式: 每个 session 独立文件 ({name}_X.npy + {name}_y.npy)
      2. X_train.npy + y_train.npy — 旧新格式 (合并好的训练集)
      3. X.npy + y.npy — 旧旧格式 (无验证分割)
      4. 子目录含 _imu_ 文件 — 原始 session, 打印错误引导

    val_sessions: 指定哪些 session 划入验证集 (不参与训练, 不做平衡)
    neg_sessions: 纯背景 session, 平衡采样时给 2x 配额

    返回 dict: {X_train, y_train, X_val, y_val, mean, std, preproc, format}
    """
    sessions_path = os.path.join(data_dir, "sessions.txt")

    # ── 格式1: 按 session 独立切片 (sessions.txt + {name}_X.npy) ──
    if os.path.exists(sessions_path):
        with open(sessions_path, "r", encoding="utf-8") as f:
            all_sessions = [line.strip() for line in f if line.strip()]

        val_set = set(val_sessions) if val_sessions else set()
        neg_set = set(neg_sessions) if neg_sessions else set()
        all_set = set(all_sessions)

        # 校验: val/neg 中未出现在 sessions.txt 中的名字 → 报警
        unknown_val = val_set - all_set
        unknown_neg = neg_set - all_set
        if unknown_val:
            print(f"  [WARN] --val-sessions 中这些 session 不在 {data_dir}/ 中: {sorted(unknown_val)}")
        if unknown_neg:
            print(f"  [WARN] --neg-sessions 中这些 session 不在 {data_dir}/ 中: {sorted(unknown_neg)}")

        train_X, train_y = [], []
        train_names = []
        val_X, val_y = [], []
        train_hit, train_bg = 0, 0
        val_hit, val_bg = 0, 0

        print(f"\n  Session 分割:")
        if val_sessions:
            print(f"  验证集: {val_sessions}")
            print(f"  训练集: 其余 {len(all_sessions) - len(val_set)} 个 session")
        if neg_sessions:
            print(f"  负样本优先保留: {neg_sessions}")
        print(f"  {'─' * 60}")

        for name in all_sessions:
            x_path = os.path.join(data_dir, f"{name}_X.npy")
            y_path = os.path.join(data_dir, f"{name}_y.npy")
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                print(f"  [!] {name}: 缺少切片文件, 跳过")
                continue

            X = np.load(x_path)
            y = np.load(y_path).astype(np.int64)
            nhit = int(y.sum())
            nbg = X.shape[0] - nhit

            if val_sessions and name in val_set:
                val_X.append(X); val_y.append(y)
                val_hit += nhit; val_bg += nbg
                tag = "[VAL]"
            else:
                train_X.append(X); train_y.append(y)
                train_names.append(name)
                train_hit += nhit; train_bg += nbg
                is_neg = neg_sessions and name in neg_set
                tag = "[TRAIN-NEG]" if is_neg else "[TRAIN]"

            print(f"  {tag} {name:25s}: {X.shape[0]:5d} 窗口, hit={nhit:4d}, bg={nbg:4d}")

        if not train_X:
            print("  [X] 未找到任何训练数据!")
            return {"X_train": np.empty((0, 6, 15)), "y_train": np.empty((0,)),
                    "X_val": None, "y_val": None,
                    "mean": None, "std": None, "preproc": None, "format": "sessions"}

        # ── 平衡训练集 ──
        X_train, y_train = _balance_dataset(train_X, train_y, train_names, neg_set, balance_mode)

        # ── 汇总验证集 ──
        X_val_out, y_val_out = None, None
        if val_X:
            X_val_out = np.concatenate(val_X)
            y_val_out = np.concatenate(val_y)
            print(f"  X_val:   {X_val_out.shape}  y_val:   {y_val_out.shape}  "
                  f"(hit={val_hit}, bg={val_bg})")
        else:
            print(f"  (无独立验证集, 将随机分割)")

        # ── 加载 preproc ──
        pp_path = os.path.join(data_dir, "preproc.npz")
        mean, std, preproc = None, None, None
        if os.path.exists(pp_path):
            pp = np.load(pp_path)
            mean = pp["mean"]; std = pp["std"]
            preproc = {k: pp[k].item() if pp[k].ndim == 0 else pp[k]
                       for k in pp.files if k not in ("mean", "std")}

        return {"X_train": X_train, "y_train": y_train,
                "X_val": X_val_out, "y_val": y_val_out,
                "mean": mean, "std": std, "preproc": preproc, "format": "sessions"}

    # ── 格式2: 旧新格式 (X_train.npy + y_train.npy) ──
    xt_path = os.path.join(data_dir, "X_train.npy")
    yt_path = os.path.join(data_dir, "y_train.npy")
    if os.path.exists(xt_path) and os.path.exists(yt_path):
        print("  [兼容] 加载旧格式 X_train.npy + y_train.npy")
        result = {"format": "merged", "X_train": np.load(xt_path),
                  "y_train": np.load(yt_path).astype(np.int64),
                  "X_val": None, "y_val": None,
                  "mean": None, "std": None, "preproc": None}
        xv_path = os.path.join(data_dir, "X_val.npy")
        yv_path = os.path.join(data_dir, "y_val.npy")
        if os.path.exists(xv_path) and os.path.exists(yv_path):
            result["X_val"] = np.load(xv_path)
            result["y_val"] = np.load(yv_path).astype(np.int64)
        pp_path = os.path.join(data_dir, "preproc.npz")
        if os.path.exists(pp_path):
            pp = np.load(pp_path)
            result["mean"] = pp["mean"]; result["std"] = pp["std"]
            result["preproc"] = {k: pp[k].item() if pp[k].ndim == 0 else pp[k]
                                for k in pp.files if k not in ("mean", "std")}
        if val_sessions:
            print("  [!] --val-sessions 对合并格式无效, 请用 sessions.txt 格式")
        return result

    # ── 格式3: 旧旧格式 (X.npy + y.npy) ──
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("  [兼容] 加载旧格式 X.npy + y.npy (无 preproc)")
        return {"format": "legacy", "X_train": np.load(x_path),
                "y_train": np.load(y_path).astype(np.int64),
                "X_val": None, "y_val": None,
                "mean": None, "std": None, "preproc": None}

    # ── 格式4: 原始 session 目录 → 报错引导 ──
    try:
        raw = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))
               and any("_imu_" in f for f in os.listdir(os.path.join(data_dir, d)))]
    except Exception:
        raw = []
    if raw:
        print(f"\n  [X] 检测到 {len(raw)} 个原始 session 目录, 不能直接训练!")
        print(f"      请先运行 slice_dataset.py 进行预处理:")
        print(f"      $ python slice_dataset.py --data_dir {data_dir} --save sliced_data")
        print(f"      然后训练: python train_model.py --data_dir sliced_data")
        raise SystemExit(1)

    print(f"\n  [X] {data_dir} 下未找到预处理数据")
    raise SystemExit(1)


def _balance_random(train_X, train_y, train_names, neg_set):
    """legacy模式: bg≤hit×2, 均匀随机抽取, neg×3."""
    total_hit = sum(int((y == 1).sum()) for y in train_y)
    total_bg = sum(int((y == 0).sum()) for y in train_y)
    total_bg_keep = min(total_bg, total_hit * 2)
    n_sessions = len(train_y)
    rng = np.random.RandomState(42)
    session_bg_counts = np.array([int((y == 0).sum()) for y in train_y], dtype=int)
    session_hit_counts = np.array([int((y == 1).sum()) for y in train_y], dtype=int)
    base_quota = total_bg_keep // n_sessions
    quota = np.full(n_sessions, base_quota, dtype=int)
    quota[:total_bg_keep % n_sessions] += 1
    if neg_set:
        for i, name in enumerate(train_names):
            if name in neg_set:
                quota[i] = min(session_bg_counts[i], quota[i] * 3)
    balanced_X, balanced_y, kept = [], [], []
    for i in range(n_sessions):
        X_sess, y_sess = train_X[i], train_y[i]
        hit_idx = np.where(y_sess == 1)[0]
        bg_idx = np.where(y_sess == 0)[0]
        n_bg_keep = min(quota[i], len(bg_idx))
        keep_bg = rng.choice(bg_idx, n_bg_keep, replace=False) if n_bg_keep > 0 and len(bg_idx) > n_bg_keep else bg_idx
        keep = np.concatenate([hit_idx, keep_bg]); keep.sort()
        balanced_X.append(X_sess[keep]); balanced_y.append(y_sess[keep])
        kept.append(len(keep_bg))
    deficit = total_bg_keep - sum(kept)
    if deficit > 0:
        surplus = [i for i in range(n_sessions) if kept[i] < session_bg_counts[i]]
        if surplus:
            total_surplus = sum(session_bg_counts[i] - kept[i] for i in surplus)
            for i in surplus:
                avail = session_bg_counts[i] - kept[i]
                n_extra = min(avail, int(deficit * avail / total_surplus) + 1)
                n_extra = min(n_extra, deficit)
                if n_extra > 0:
                    bg_idx = np.where(train_y[i] == 0)[0]
                    extra_bg = rng.choice(bg_idx, min(n_extra, len(bg_idx)), replace=False)
                    balanced_X[i] = np.concatenate([balanced_X[i], train_X[i][extra_bg]])
                    balanced_y[i] = np.concatenate([balanced_y[i], np.zeros(len(extra_bg), dtype=train_y[i].dtype)])
                    kept[i] += len(extra_bg); deficit -= len(extra_bg)
    X_train = np.concatenate(balanced_X); y_train = np.concatenate(balanced_y)
    print(f"\n  训练集 (平衡-random): hit={total_hit}, bg 目标={total_bg_keep}, 实际={int((y_train==0).sum())}")
    for i in range(n_sessions):
        is_neg = neg_set and train_names[i] in neg_set
        print(f"    [{i}] {train_names[i]:12s} {session_bg_counts[i]:5d} bg -> 抽 {kept[i]:5d}"
              f"  (hit={session_hit_counts[i]:4d}){' [NEG x3]' if is_neg else ''}")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    return X_train, y_train


def _balance_dataset(train_X: list, train_y: list, train_names: list,
                     neg_set: set, balance_mode: str = "posthit") -> tuple[np.ndarray, np.ndarray]:
    """
    平衡训练集:
      balance_mode="posthit": 优先保留击打后120ms负例, bg≤hit×2.5
      balance_mode="random":  均匀随机抽取, bg≤hit×2.0
      neg_set 中的 session 获得 3x 配额.
    """
    if balance_mode == "random":
        return _balance_random(train_X, train_y, train_names, neg_set)

    POST_HIT_FRAMES = 12  # 200ms @ 60Hz / 1帧stride ≈ 12帧

    total_hit = sum(int((y == 1).sum()) for y in train_y)
    total_bg = sum(int((y == 0).sum()) for y in train_y)
    total_bg_keep = min(total_bg, int(total_hit * 2.5))

    n_sessions = len(train_y)
    rng = np.random.RandomState(42)

    session_bg_counts = np.array([int((y == 0).sum()) for y in train_y], dtype=int)
    session_hit_counts = np.array([int((y == 1).sum()) for y in train_y], dtype=int)

    # Phase 1: 标记每个 session 中"击打后120ms"的负例
    post_hit_masks = []  # 每个session的bool数组
    post_hit_counts = []
    for i in range(n_sessions):
        y_sess = train_y[i]
        mask = np.zeros(len(y_sess), dtype=bool)
        # 找到每个 hit run 的结束位置
        changes = np.diff(np.concatenate([[0], y_sess, [0]]))
        ends = np.where(changes == -1)[0]  # 1→0 的位置, 即 hit run 结束后的第一帧
        for end in ends:
            if end < len(y_sess):
                # 标记之后 POST_HIT_FRAMES 帧 (跳过高位=已经是正例)
                for j in range(end, min(end + POST_HIT_FRAMES, len(y_sess))):
                    if y_sess[j] == 0:
                        mask[j] = True
        post_hit_masks.append(mask)
        post_hit_counts.append(int(mask.sum()))

    total_post_hit = sum(post_hit_counts)
    print(f"\n  [平衡] 击打后200ms负例: {total_post_hit} (占全部bg的 {total_post_hit/total_bg*100:.1f}%)")

    # Phase 2: 分配配额 — 先保留所有post-hit负例, 剩余按session均匀分
    # 如果post-hit已经超过总配额, 按比例下采样post-hit
    if total_post_hit > total_bg_keep:
        post_hit_keep_ratio = total_bg_keep / total_post_hit
        print(f"  [平衡] post-hit({total_post_hit}) > 配额({total_bg_keep}), "
              f"按{post_hit_keep_ratio:.1%}下采样")
    else:
        post_hit_keep_ratio = 1.0

    remaining_quota = max(0, total_bg_keep - int(total_post_hit * post_hit_keep_ratio))
    base_quota = remaining_quota // n_sessions
    quota = np.full(n_sessions, base_quota, dtype=int)
    quota[:remaining_quota % n_sessions] += 1

    # neg_set 中的 session: 3x
    if neg_set:
        for i, name in enumerate(train_names):
            if name in neg_set:
                quota[i] = min(session_bg_counts[i], quota[i] * 3)

    balanced_X, balanced_y = [], []
    kept_per_session = []
    kept_post_hit = []

    for i in range(n_sessions):
        X_sess, y_sess = train_X[i], train_y[i]
        hit_idx = np.where(y_sess == 1)[0]
        bg_idx = np.where(y_sess == 0)[0]

        # 先取所有 post-hit 负例 (可能按下采样比例)
        post_hit_idx = np.where(post_hit_masks[i])[0]
        n_post_keep = max(1, int(len(post_hit_idx) * post_hit_keep_ratio)) if len(post_hit_idx) > 0 else 0
        if n_post_keep > 0 and len(post_hit_idx) > n_post_keep:
            keep_post = rng.choice(post_hit_idx, n_post_keep, replace=False)
        else:
            keep_post = post_hit_idx

        # 再从剩余负例中抽取 quota
        remaining_bg = np.setdiff1d(bg_idx, keep_post)
        n_quota_keep = min(quota[i], len(remaining_bg))
        if n_quota_keep > 0 and len(remaining_bg) > n_quota_keep:
            keep_quota = rng.choice(remaining_bg, n_quota_keep, replace=False)
        else:
            keep_quota = remaining_bg

        keep_bg = np.concatenate([keep_post, keep_quota])
        keep = np.concatenate([hit_idx, keep_bg]); keep.sort()
        balanced_X.append(X_sess[keep])
        balanced_y.append(y_sess[keep])
        kept_per_session.append(len(keep_bg))
        kept_post_hit.append(len(keep_post))

    # deficit redistribution
    actual_kept = sum(kept_per_session)
    deficit = total_bg_keep - actual_kept
    if deficit > 0:
        surplus = [i for i in range(n_sessions) if kept_per_session[i] < session_bg_counts[i]]
        if surplus:
            total_surplus = sum(session_bg_counts[i] - kept_per_session[i] for i in surplus)
            remaining = deficit
            for i in surplus:
                avail = session_bg_counts[i] - kept_per_session[i]
                n_extra = min(avail, int(deficit * avail / total_surplus) + 1)
                n_extra = min(n_extra, remaining)
                if n_extra > 0:
                    bg_idx = np.where(train_y[i] == 0)[0]
                    extra_bg = rng.choice(bg_idx, min(n_extra, len(bg_idx)), replace=False)
                    balanced_X[i] = np.concatenate([balanced_X[i], train_X[i][extra_bg]])
                    balanced_y[i] = np.concatenate([balanced_y[i], np.zeros(len(extra_bg), dtype=train_y[i].dtype)])
                    kept_per_session[i] += len(extra_bg)
                    remaining -= len(extra_bg)

    X_train = np.concatenate(balanced_X)
    y_train = np.concatenate(balanced_y)

    print(f"\n  训练集 (平衡): hit={total_hit}, bg 目标={total_bg_keep}, "
          f"实际={int((y_train==0).sum())}, 其中post-hit={sum(kept_post_hit)}")
    for i in range(n_sessions):
        is_neg = neg_set and train_names[i] in neg_set
        tag = " [NEG x3]" if is_neg else ""
        ph = kept_post_hit[i]
        print(f"    [{i}] {train_names[i]:12s} {session_bg_counts[i]:5d} bg -> 抽 {kept_per_session[i]:5d}"
              f"  (hit={session_hit_counts[i]:4d}, post-hit={ph}){tag}")

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    return X_train, y_train


# ============================================================================
# 二、模型定义
# ============================================================================

def build_inception_time(dls, nf: int = 32, depth: int = 4,
                         seq_len: int | None = None,
                         dropout: float = 0.4):
    """
    InceptionTime 定制版 — 针对短窗口优化
    =========================================================================
    相比 tsai 默认版的两大改动:

    1. 卷积核 [3,5,7] 替代 [39,19,9]
       - k=3 →  ~50ms 局部模式 (踩踏上升沿, ~3帧)
       - k=5 →  ~83ms 中等模式
       - k=7 → ~117ms 较长模式 (衰减过程)

    2. 指数加权池化 替代 GAP (全局平均池化)
       - 靠近窗口末尾的帧权重更大
       - 模型学会"窗口末帧 = 此刻 = 预测目标"

    参数:
      nf=32:   每支路滤波器数 (4路 = 128输出通道)
      depth=4: Inception 模块堆叠层数
      seq_len: 窗口帧数 (None=自动从dls推断)
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
            n_t = seq_len if seq_len else 15  # 动态适配窗口帧数
            init_w = torch.exp(torch.linspace(0, 2.0, n_t))
            self.time_weights = nn.Parameter(init_w)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
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
    print(f"  [模型] 池化: 指数加权 (末尾帧 ~7.4x)")
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
# 三、训练流程
# ============================================================================

# ============================================================================
# 三、训练流程
# ============================================================================

def _imu_rotate_augment(X: np.ndarray, max_angle_deg: float = 15) -> np.ndarray:
    """
    对每个窗口施加随机小角度 3D 旋转, 模拟不同手机朝向.
    旋转 gyro(0-2), accel(3-5), gravity(10-12), 然后重新计算衍生特征(13-17).

    参考: Um et al. (2017) "Data augmentation of wearable sensor data"
          Müller et al. (2024) IMU-based fitness activity recognition
    """
    rng = np.random.RandomState()
    N, C, T = X.shape
    X_aug = X.copy()

    for i in range(N):
        angle = np.deg2rad(rng.uniform(0, max_angle_deg))
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis) + 1e-8

        # Rodrigues 旋转公式
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # 逐帧旋转 gyro, accel, gravity
        for t in range(T):
            X_aug[i, :3, t] = R @ X[i, :3, t]
            X_aug[i, 3:6, t] = R @ X[i, 3:6, t]
            X_aug[i, 10:13, t] = R @ X[i, 10:13, t]

        # 重新计算衍生特征
        ax, ay, az = X_aug[i, 3], X_aug[i, 4], X_aug[i, 5]
        gx, gy, gz = X_aug[i, 0], X_aug[i, 1], X_aug[i, 2]
        grx, gry, grz = X_aug[i, 10], X_aug[i, 11], X_aug[i, 12]

        X_aug[i, 13] = np.sqrt(ax**2 + ay**2 + az**2)
        X_aug[i, 14] = np.sqrt(gx**2 + gy**2 + gz**2)
        X_aug[i, 15] = np.diff(X_aug[i, 13], prepend=X_aug[i, 13, 0])
        grav_norm = np.sqrt(grx**2 + gry**2 + grz**2) + 1e-8
        X_aug[i, 16] = (ax * grx + ay * gry + az * grz) / grav_norm
        X_aug[i, 17] = np.diff(X_aug[i, 14], prepend=X_aug[i, 14, 0])

    return X_aug


def train(X: np.ndarray, y: np.ndarray, model_name: str = "inception",
          epochs: int =50, batch_size: int = 64, lr: float = 5e-4,
          save_dir: str = "models", focal_gamma: float = 0,
          X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
          preproc_mean: np.ndarray | None = None,
          preproc_std: np.ndarray | None = None,
          aug_rotate_deg: float = 0,
          dropout: float = 0.4,
          weight_decay: float = 1e-3):
    """
    完整训练流程:
      1. 数据增强 — IMU 旋转 (模拟不同手机朝向)
      2. 标准化 (每通道 Z-score)
      3. 训练/验证分割:
         - 提供了 X_val/y_val → 直接使用 (session 级独立验证集)
         - 未提供 → 窗口级随机分割 (旧行为, 可能泄露)
      4. InceptionTime / FCN / LiteCNN / TST
      5. CrossEntropyLoss / AdditiveMarginLoss + AdamW (One-Cycle 学习率)
      6. EarlyStopping: valid_loss 连续 10 轮不降 → 停止
      7. 保存验证集 loss 最低的模型 + 预处理参数
      8. 评估: 分类报告 + 混淆矩阵 + 训练曲线

    aug_rotate_deg: IMU 旋转增强最大角度 (度). 0=不增强. 建议 10-15.
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

    # ── 3.1 数据增强: IMU 旋转 (3个随机副本) ──
    if aug_rotate_deg > 0:
        N_COPIES = 3
        print(f"\n  [增强] IMU 旋转: max={aug_rotate_deg}°, {N_COPIES}个随机副本")
        X_parts, y_parts = [X], [y]
        for copy_i in range(N_COPIES):
            X_aug = _imu_rotate_augment(X, aug_rotate_deg)
            X_parts.append(X_aug)
            y_parts.append(y)
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        print(f"  [增强] 训练集: {len(X)} 样本 (原始 + {N_COPIES}x增强 = {N_COPIES+1}x)")
        # 注意: 验证集不做增强

    # ── 3.2 标准化 ──
    # 优先使用 slice_dataset.py 预计算的统计量, 否则从训练集计算
    if preproc_mean is not None and preproc_std is not None:
        mean = preproc_mean.reshape(1, -1, 1)
        std = preproc_std.reshape(1, -1, 1)
        print(f"  [标准化] 使用预计算 mean/std (来自 preproc.npz)")
    else:
        mean = X.mean(axis=(0, 2), keepdims=True)   # (1, 6, 1)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        print(f"  [标准化] 从训练集计算 mean/std")
    X = (X - mean) / std

    if X_val is not None and y_val is not None:
        y_val = y_val.astype(np.int64)
        X_val = (X_val - mean) / std

    # ── 3.2 训练/验证分割 ──
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
        print(f"  提示: 建议用 slice_dataset.py --val-sessions 指定独立验证 session")

    # ── 3.3 构建 tsai DataLoaders ──
    dls = get_ts_dls(X_all, y_all, splits=splits, bs=batch_size)
    dls.c = 2  # 强制二分类 (tsai 1.0.1 默认将二值标签视为 1 类输出, 需显式设 2)

    # ── 3.4 选择模型 ──
    models = {
        "inception": build_inception_time,
        "fcn": build_fcn,
        "lite": build_lite_cnn,
        "tst": build_tst,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(models.keys())}")
    seq_len = X.shape[2] if len(X.shape) == 3 else None
    model = (models[model_name](dls, seq_len=seq_len, dropout=dropout)
             if model_name == "inception"
             else models[model_name](dls))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  模型: {model_name}, 参数量: {param_count:,}")
    win_frames = X.shape[2]
    print(f"  窗口: {win_frames / 60 * 1000:.0f}ms ({win_frames}帧 @60Hz)")
    print(f"  训练集: {len(splits[0])}, 验证集: {len(splits[1])}")

    # ── 3.5 Learner (训练器) ──
    # 构建 loss 函数
    if focal_gamma > 0:
        from tsai.all import FocalLoss
        loss_func = FocalLoss(gamma=focal_gamma)
        print(f"\n  [Loss] tsai FocalLoss  (gamma={focal_gamma})")
    else:
        loss_func = nn.CrossEntropyLoss()
        print(f"\n  [Loss] CrossEntropyLoss")

    print(f"\n  [正则化] dropout={dropout}, weight_decay={weight_decay}")
    learn = Learner(
        dls, model,
        loss_func=loss_func,
        metrics=[accuracy],
        wd=weight_decay,
        cbs=[
            SaveModelCallback(monitor="valid_loss"),
            EarlyStoppingCallback(monitor="valid_loss", patience=10),
        ],
    )
    learn.model.to(device)

    # ── 3.6 One-Cycle 学习率策略 ──
    # 从 lr/10 升至 lr, 再降至 lr/1000, 帮助模型快速收敛并逃离局部最优
    learn.fit_one_cycle(epochs, lr_max=lr)

    # ── 3.7 评估 ──
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

    # ── 3.8 保存模型 ──
    # 同时保存预处理参数 (mean, std), 推理时需要用相同方式标准化
    # 窗口/采样率参数是模型架构常量，与 slice_dataset.py 保持一致
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"hit_detector_{model_name}.pth")
    torch.save({
        "model_state": model.state_dict(),    # 模型权重
        "model_name": model_name,             # 架构名 (推理时重建用)
        "mean": mean,                          # 标准化均值 (1,6,1)
        "std": std,                            # 标准化标准差 (1,6,1)
        "window_sec": X.shape[2] / 60.0,        # 窗口长度 (从帧数推导, N/60Hz)
        "window_frames": X.shape[2],            # 窗口帧数 (从数据推导)
        "target_fs": 60,                       # 采样率
        "channels": X.shape[1],                  # 输入通道数 (从数据推导)
        "classes": 2,                          # 输出类别数
    }, model_path)
    print(f"\n  模型已保存: {model_path}")

    # ── 3.9 绘制训练曲线 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rec = learn.recorder
        train_loss = [v[0] for v in rec.values]
        valid_loss_vals = [v[1] for v in rec.values]
        acc_vals = [v[2] for v in rec.values] if len(rec.values[0]) >= 3 else None

        print(f"  [曲线] epochs={len(train_loss)}, has_accuracy={acc_vals is not None}")
        print(f"  [曲线] train_loss=[{train_loss[0]:.4f}..{train_loss[-1]:.4f}]")
        print(f"  [曲线] valid_loss=[{valid_loss_vals[0]:.4f}..{valid_loss_vals[-1]:.4f}]")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs_arr = range(1, len(train_loss) + 1)

        ax1.plot(epochs_arr, train_loss, "b-", lw=1.5, label="Train Loss")
        ax1.plot(epochs_arr, valid_loss_vals, "r-", lw=1.5, label="Valid Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)
        # 固定 y 轴起点为0, 防止空白
        all_loss = train_loss + valid_loss_vals
        ax1.set_ylim(0, max(all_loss) * 1.1)

        if acc_vals is not None:
            ax2.plot(epochs_arr, acc_vals, "g-", lw=1.5, label="Accuracy")
            ax2.set_ylim(0, 1.05)
        else:
            ax2.text(0.5, 0.5, "accuracy unavailable",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Training Curve - {model_name}", fontsize=14, weight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, f"curve_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close("all")
        print(f"  训练曲线: {path}")
    except Exception as e:
        import traceback
        print(f"  [WARN] 绘制曲线失败: {e}")
        traceback.print_exc()

    return learn, model


# ============================================================================
# 四、命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IMU 击打检测模型训练 (InceptionTime/FCN/LiteCNN/TST)"
    )
    parser.add_argument("--data_dir", default="sliced_data",
                        help="预处理后的数据目录 (slice_dataset.py 的输出, 含 X_train.npy 等)")
    parser.add_argument("--model", default="inception",
                        choices=["inception", "fcn", "lite", "tst"],
                        help="模型架构: inception (推荐), fcn (快速baseline), "
                             "lite (极轻量), tst (Transformer对比)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="最大训练轮数 (早停机制会在 loss 不降时提前停止)")
    parser.add_argument("--bs", type=int, default=64,
                        help="批次大小 (显存不够可减小)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="初始学习率 (One-Cycle 策略会自动调整, 默认 5e-4)")
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
                        help="逗号分隔的负样本 session 名, 例如 'shaking_0,shaking_1'. "
                             "背景样本在下采样时优先保留, 确保纯负样本不被随机丢弃.")
    parser.add_argument("--aug-rotate", type=float, default=0,
                        help="IMU 旋转增强最大角度 (度). 0=不增强. "
                             "建议 10-15. 模拟小幅手机朝向变化, 提升泛化能力.")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout 概率 (默认 0.4, 防过拟合)")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="L2 权重衰减 (默认 1e-3, 防过拟合)")
    parser.add_argument("--no-augment", action="store_true",
                        help="禁用数据增强 (快速训练)")
    parser.add_argument("--balance-mode", type=str, default="posthit",
                        choices=["posthit", "random"],
                        help="负例平衡策略: posthit=优先保留击打后120ms负例(默认), "
                             "random=均匀随机抽取")

    args = parser.parse_args()

    aug_rotate = 0 if args.no_augment else args.aug_rotate

    val_sessions = [s.strip() for s in args.val_sessions.split(",") if s.strip()] \
                   if args.val_sessions else None
    neg_sessions = [s.strip() for s in args.neg_sessions.split(",") if s.strip()] \
                   if args.neg_sessions else None

    print("=" * 55)
    print("  IMU 击打检测 — 训练")
    print(f"  模型: {args.model}")
    if args.focal_loss > 0:
        print(f"  Loss:  FocalLoss (gamma={args.focal_loss})")
    if val_sessions:
        print(f"  验证:  session 级独立 ({', '.join(val_sessions)})")
    if neg_sessions:
        print(f"  负样本优先: ({', '.join(neg_sessions)})")
    print("=" * 55)

    # 加载预处理数据 (按 session 独立切片)
    data = load_session_slices(args.data_dir, val_sessions, neg_sessions,
                               balance_mode=args.balance_mode)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    preproc_mean, preproc_std = data["mean"], data["std"]

    # 打印 preproc 元数据
    if data["preproc"]:
        pp = data["preproc"]
        print(f"  数据格式: {data['format']}")
        print(f"  窗口: {pp.get('window_frames', '?')} 帧 @{pp.get('target_fs', '?')}Hz "
              f"({pp.get('window_sec', 0)*1000:.0f}ms)")
        print(f"  击打区: [-{pp.get('pre_ms', '?')}ms, +{pp.get('post_ms', '?')}ms]")
        print(f"  X_train: {X_train.shape}  y_train: {X_train.shape}")
        if X_val is not None:
            print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    else:
        print(f"  数据格式: {data['format']} (无 preproc.npz, 标准化参数将即时计算)")
        print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

    if len(X_train) == 0:
        print("  [X] 无数据")
        return

    # 训练
    train(X_train, y_train, args.model, args.epochs, args.bs, args.lr, args.save_dir,
          focal_gamma=args.focal_loss, X_val=X_val, y_val=y_val,
          preproc_mean=preproc_mean, preproc_std=preproc_std,
          aug_rotate_deg=aug_rotate,
          dropout=args.dropout, weight_decay=args.weight_decay)


if __name__ == "__main__":
    main()
