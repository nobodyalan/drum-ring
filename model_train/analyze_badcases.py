"""
================================================================================
 模型 Badcase 分析 — 按 session 定位误判并可视化
================================================================================

 对每个 session 的切片数据跑模型推理，找出:
   - FP (False Positive): 真实=背景, 预测=击打  → 误触发
   - FN (False Negative): 真实=击打, 预测=背景  → 漏检

 输出:
   1. 控制台: 按 session 汇总的混淆矩阵 + FP/FN 统计
   2. summary.png: 汇总图 (4合1: 分布/错误率/概率/混淆矩阵)
   3. false_positives/: FP 单窗口波形图
   4. false_negatives/: FN 单窗口波形图

 ────────────────────────────────────────────────────────────────────────────
 参数:
   --model PATH           模型 .pth 路径 (必须)
   --data_dir DIR         切片数据目录 (必须)
   --save DIR             输出目录 (默认: badcase_analysis)
   --session SESSION      只分析指定 session (可多次指定)
   --max-cases N          每个 session 最多画多少个图 (默认: 20)

 用法:
   python analyze_badcases.py --model models/hit_detector_inception.pth \
       --data_dir sliced_datas/test_sliced_data

   python analyze_badcases.py --model models/hit_detector_inception.pth \
       --data_dir sliced_datas/test_sliced_data --session slow_wxt_0

   python analyze_badcases.py --model models/hit_detector_inception.pth \
       --data_dir sliced_datas/test_sliced_data --max-cases 30

   python analyze_badcases.py --model models/hit_detector_inception.pth \
       --data_dir sliced_datas/test_sliced_data --session slow_0 --session double_b_0
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试设置中文字体
_CJK_FONT = None
for _fname in fm.findSystemFonts():
    try:
        _fp = fm.FontProperties(fname=_fname)
        _fn = _fp.get_name().lower()
        if any(k in _fn for k in ["microsoft yahei", "simhei", "simsun",
                                   "noto sans cjk", "wenquanyi", "source han"]):
            _CJK_FONT = _fp.get_name()
            break
    except Exception:
        continue

if _CJK_FONT:
    plt.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
else:
    _CJK_FONT = None

# 根据是否有 CJK 字体选择标签语言
def _L(cn: str, en: str) -> str:
    """根据字体可用性返回中文或英文标签."""
    return cn if _CJK_FONT else en
from matplotlib.gridspec import GridSpec
from collections import defaultdict

import torch

# ============================================================================
# 一、模型加载 (与 realtime_detect.py 一致的架构)
# ============================================================================

def build_inception_model(c_in: int, c_out: int, window_frames: int, dropout: float = 0.2):
    """重建 InceptionTime 模型架构."""
    import torch.nn as nn

    ks = [3, 5, 7]
    nf = 32
    depth = 4
    out_ch = nf * (len(ks) + 1)

    class InceptionModule(nn.Module):
        def __init__(self, ni, nf, ks):
            super().__init__()
            self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False)
            self.convs = nn.ModuleList([
                nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in ks
            ])
            self.maxpool = nn.Sequential(
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(ni, nf, 1, bias=False),
            )
            self.bn = nn.BatchNorm1d(out_ch)
            self.act = nn.ReLU()

        def forward(self, x):
            z = self.bottleneck(x)
            branches = [conv(z) for conv in self.convs]
            branches.append(self.maxpool(x))
            return self.act(self.bn(torch.cat(branches, dim=1)))

    class InceptionBlock(nn.Module):
        def __init__(self, ni, nf, depth, ks):
            super().__init__()
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

    class WeightedInceptionTime(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = InceptionBlock(c_in, nf, depth, ks)
            init_w = torch.exp(torch.linspace(0, 2.0, window_frames))
            self.time_weights = nn.Parameter(init_w)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(out_ch, c_out),
            )

        def forward(self, x):
            feat = self.backbone(x)
            T = feat.shape[-1]
            w = self.time_weights[:T]
            w = torch.softmax(w, dim=0)
            pooled = (feat * w.view(1, 1, -1)).sum(dim=-1)
            return self.head(pooled)

    return WeightedInceptionTime()


def load_model(model_path: str):
    """加载模型及预处理参数."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model_name = ckpt.get("model_name", "inception")
    c_in = ckpt["channels"]
    c_out = ckpt["classes"]
    window_frames = ckpt["window_frames"]

    # 重建模型: 从 state_dict key 判断架构
    all_keys = list(ckpt["model_state"].keys())
    sample_key = all_keys[0]
    # 定制 InceptionTime: 有 backbone.xxx 或 time_weights / head.xxx
    is_custom = any(k.startswith("backbone.") or k in ("time_weights",)
                    or k.startswith("head.") for k in all_keys)
    if is_custom:
        model = build_inception_model(c_in, c_out, window_frames)
    elif "inceptionblock" in sample_key:
        # 旧版 tsai InceptionTime
        from tsai.all import InceptionTime
        model = InceptionTime(c_in=c_in, c_out=c_out, nf=32, depth=6)
    else:
        raise ValueError(f"未知模型架构, state_dict key: {sample_key}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def _to_np(arr):
        if isinstance(arr, np.ndarray):
            return arr.squeeze().astype(np.float32)
        return arr.squeeze().cpu().numpy().astype(np.float32)

    mean = _to_np(ckpt["mean"])
    std = _to_np(ckpt["std"])

    return model, mean, std, c_in, window_frames


# ============================================================================
# 二、数据加载
# ============================================================================

def load_sliced_sessions(data_dir: str, session_filter: list[str] | None = None):
    """加载 sessions.txt 中所有 session 的切片数据."""
    sessions_path = os.path.join(data_dir, "sessions.txt")
    if not os.path.exists(sessions_path):
        raise FileNotFoundError(f"未找到 {sessions_path}")

    with open(sessions_path, "r", encoding="utf-8") as f:
        all_sessions = [line.strip() for line in f if line.strip()]

    if session_filter:
        all_sessions = [s for s in all_sessions if s in session_filter]

    data = {}
    for name in all_sessions:
        x_path = os.path.join(data_dir, f"{name}_X.npy")
        y_path = os.path.join(data_dir, f"{name}_y.npy")
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"  [!] {name}: 缺少切片文件, 跳过")
            continue
        X = np.load(x_path)
        y = np.load(y_path).astype(np.int64)
        data[name] = {"X": X, "y": y}

    return data


# ============================================================================
# 三、推理 & Badcase 检测
# ============================================================================

def run_inference(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  batch_size: int = 512) -> np.ndarray:
    """
    对全部窗口做推理, 返回 softmax 概率 (N, 2).
    """
    # 标准化
    mean_t = mean.reshape(1, -1, 1)
    std_t = std.reshape(1, -1, 1)
    X_norm = (X - mean_t) / (std_t + 1e-8)

    probs_all = []
    for i in range(0, len(X_norm), batch_size):
        batch = torch.from_numpy(X_norm[i:i + batch_size])
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)

    return np.concatenate(probs_all, axis=0)


def find_badcases(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    找出所有 FP 和 FN, 返回结构化的 badcase 列表.

    返回:
      { "fp": [(idx, p_hit), ...], "fn": [(idx, p_hit), ...] }
    """
    pred = probs.argmax(axis=1)
    p_hit = probs[:, 1]

    fp_mask = (y_true == 0) & (pred == 1)
    fn_mask = (y_true == 1) & (pred == 0)

    fp_cases = [(int(i), float(p_hit[i]))
                for i in np.where(fp_mask)[0]]
    fn_cases = [(int(i), float(p_hit[i]))
                for i in np.where(fn_mask)[0]]

    # 按 p_hit 降序排列 (置信度最高的误判排前面)
    fp_cases.sort(key=lambda x: -x[1])
    fn_cases.sort(key=lambda x: x[1])  # 置信度最低 (最不像hit) 的排前面

    return {"fp": fp_cases, "fn": fn_cases}


# ============================================================================
# 四、可视化
# ============================================================================

CHANNEL_LABELS_18 = [
    "gyro_x", "gyro_y", "gyro_z",      # 0-2
    "accel_x", "accel_y", "accel_z",   # 3-5
    "quat_x", "quat_y", "quat_z", "quat_w",  # 6-9
    "grav_x", "grav_y", "grav_z",      # 10-12
    "accel_mag", "gyro_mag",           # 13-14
    "accel_jerk", "vertical_accel",    # 15-16
    "gyro_jerk",                       # 17
]

CHANNEL_GROUPS = [
    ("Gyroscope", [0, 1, 2], ["#e74c3c", "#2ecc71", "#3498db"]),
    ("Accel", [3, 4, 5], ["#e67e22", "#9b59b6", "#1abc9c"]),
    ("Gravity", [10, 11, 12], ["#f39c12", "#e74c3c", "#2980b9"]),
    ("Derived", [13, 14, 15, 16, 17],
     ["#2c3e50", "#8e44ad", "#c0392b", "#16a085", "#d35400"]),
]


def plot_single_badcase(X_window: np.ndarray, y_true: int, prob_hit: float,
                        session_name: str, case_idx: int, case_type: str,
                        save_path: str):
    """
    绘制单个 badcase 的 18 通道波形图.

    参数:
        X_window: (18, T) 单窗口数据
        y_true: 真实标签 (0 或 1)
        prob_hit: 模型预测的 P(hit)
        case_type: "FP" 或 "FN"
    """
    T = X_window.shape[1]
    t_ms = np.arange(T) / 60.0 * 1000  # frames → ms

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 1.2, 0.3], hspace=0.35)

    # 标签含义
    true_label = "HIT" if y_true == 1 else "BG"
    pred_label = "HIT" if prob_hit > 0.5 else "BG"
    verdict_color = "#e74c3c" if y_true != (1 if prob_hit > 0.5 else 0) else "#27ae60"

    for group_idx, (group_name, channels, colors) in enumerate(CHANNEL_GROUPS):
        ax = fig.add_subplot(gs[group_idx])
        for ch, color in zip(channels, colors):
            label = CHANNEL_LABELS_18[ch]
            ax.plot(t_ms, X_window[ch], color=color, lw=1.0, alpha=0.8, label=label)
        ax.set_ylabel(group_name, fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", fontsize=6, ncol=len(channels))
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.axhline(y=0, color="gray", lw=0.5, alpha=0.3)
        ax.tick_params(labelsize=7)

    # 概率条
    ax_prob = fig.add_subplot(gs[4])
    ax_prob.barh([0], [prob_hit], color="#e74c3c" if prob_hit > 0.5 else "#3498db",
                 height=0.6)
    ax_prob.barh([0], [1.0], color="lightgray", height=0.6, alpha=0.3, zorder=0)
    ax_prob.axvline(x=0.5, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(-0.5, 0.5)
    ax_prob.set_xlabel("P(hit)", fontsize=10, fontweight="bold")
    ax_prob.set_yticks([])
    ax_prob.text(prob_hit + 0.02, 0, f"{prob_hit:.3f}", va="center", fontsize=11,
                 fontweight="bold", color=verdict_color)

    # 标题
    fig.suptitle(
        f"[{case_type}] {session_name}  # {case_idx}  |  "
        f"True: {true_label}  Pred: {pred_label}  |  P(hit)={prob_hit:.3f}",
        fontsize=11, fontweight="bold", color=verdict_color, y=0.995,
    )

    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary(session_stats: list[dict], save_dir: str):
    """绘制汇总图: 各 session 错误分布 + 概率分布对比."""
    names = [s["name"] for s in session_stats]
    n_total = np.array([s["total"] for s in session_stats])
    n_fp = np.array([s["fp_count"] for s in session_stats])
    n_fn = np.array([s["fn_count"] for s in session_stats])
    fp_rate = np.array([s["fp_rate"] for s in session_stats])
    fn_rate = np.array([s["fn_rate"] for s in session_stats])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ---- 左上: 各 session 样本分布 (堆叠柱状图) ----
    ax = axes[0, 0]
    x = np.arange(len(names))
    tp = np.array([s["tp"] for s in session_stats])
    tn = np.array([s["tn"] for s in session_stats])
    ax.bar(x, tn + tp, color="#2ecc71", alpha=0.7, label="TN+TP (正确)")
    ax.bar(x, n_fn, bottom=tn, color="#e74c3c", alpha=0.8, label="FN (漏检)")
    ax.bar(x, n_fp, bottom=tn + n_fn, color="#e67e22", alpha=0.8, label="FP (误报)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Prediction Results by Session", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # ---- 右上: FP/FN 率 (%) ----
    ax = axes[0, 1]
    width = 0.35
    ax.bar(x - width / 2, fp_rate * 100, width, color="#e67e22", alpha=0.8, label="FP Rate")
    ax.bar(x + width / 2, fn_rate * 100, width, color="#e74c3c", alpha=0.8, label="FN Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Error Rate (%)", fontsize=10)
    ax.set_title("FP / FN Rate by Session", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    # 标注具体的百分数
    for i in range(len(names)):
        if fp_rate[i] > 0.005:
            ax.text(i - width / 2, fp_rate[i] * 100 + 0.3, f"{fp_rate[i]*100:.1f}%",
                    ha="center", fontsize=6, color="#e67e22", fontweight="bold")
        if fn_rate[i] > 0.005:
            ax.text(i + width / 2, fn_rate[i] * 100 + 0.3, f"{fn_rate[i]*100:.1f}%",
                    ha="center", fontsize=6, color="#e74c3c", fontweight="bold")

    # ---- 左下: 全部数据的概率分布 (分正确/错误) ----
    ax = axes[1, 0]
    all_correct_hit = []
    all_correct_bg = []
    all_fp_probs = []
    all_fn_probs = []
    for s in session_stats:
        if "correct_hit_probs" in s:
            all_correct_hit.extend(s["correct_hit_probs"])
        if "correct_bg_probs" in s:
            all_correct_bg.extend(s["correct_bg_probs"])
        if "fp_probs" in s:
            all_fp_probs.extend(s["fp_probs"])
        if "fn_probs" in s:
            all_fn_probs.extend(s["fn_probs"])

    bins = np.linspace(0, 1, 51)
    if all_correct_bg:
        ax.hist(all_correct_bg, bins=bins, color="#2ecc71", alpha=0.5, label=f"TN (n={len(all_correct_bg)})")
    if all_correct_hit:
        ax.hist(all_correct_hit, bins=bins, color="#3498db", alpha=0.5, label=f"TP (n={len(all_correct_hit)})")
    if all_fp_probs:
        ax.hist(all_fp_probs, bins=bins, color="#e67e22", alpha=0.7, label=f"FP (n={len(all_fp_probs)})")
    if all_fn_probs:
        ax.hist(all_fn_probs, bins=bins, color="#e74c3c", alpha=0.7, label=f"FN (n={len(all_fn_probs)})")
    ax.axvline(x=0.5, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("P(hit)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Probability Distribution: Correct vs. Errors", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ---- 右下: 混淆矩阵汇总 ----
    ax = axes[1, 1]
    total_tn = sum(s["tn"] for s in session_stats)
    total_fp = sum(s["fp_count"] for s in session_stats)
    total_fn = sum(s["fn_count"] for s in session_stats)
    total_tp = sum(s["tp"] for s in session_stats)
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    im = ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=max(total_tn, total_tp))
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > max(total_tn, total_tp) / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i,j]/cm.sum()*100:.1f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=text_color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred BG", "Pred HIT"], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True BG", "True HIT"], fontsize=10)
    ax.set_title(f"Confusion Matrix (N={cm.sum():,})", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Badcase Analysis Summary", fontsize=14, fontweight="bold", y=0.998)
    plt.tight_layout()
    summary_path = os.path.join(save_dir, "summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  汇总图已保存: {summary_path}")


# ============================================================================
# 五、主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="模型 Badcase 分析 — 按 session 定位误判并可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析全部 session
  python analyze_badcases.py --model models/test_model/hit_detector_inception.pth \\
      --data_dir sliced_datas/test_sliced_data --save badcase_analysis

  # 只看某个 session
  python analyze_badcases.py --model models/test_model/hit_detector_inception.pth \\
      --data_dir sliced_datas/test_sliced_data --session double_a_0

  # 限制每个 session 最多画 30 个 badcase
  python analyze_badcases.py --model models/test_model/hit_detector_inception.pth \\
      --data_dir sliced_datas/test_sliced_data --max-cases 30
        """,
    )
    parser.add_argument("--model", required=True, help="模型 .pth 路径")
    parser.add_argument("--data_dir", required=True, help="切片数据目录 (含 sessions.txt)")
    parser.add_argument("--save", default="badcase_analysis", help="输出目录 (默认: badcase_analysis/)")
    parser.add_argument("--session", type=str, default=None,
                        help="只分析指定 session (可多次指定)", action="append")
    parser.add_argument("--max-cases", type=int, default=20,
                        help="每个 session 最多画多少个 badcase (默认: 20)")

    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)

    # 1. 加载模型
    print("=" * 60)
    print("  Badcase 分析")
    print(f"  模型: {args.model}")
    print(f"  数据: {args.data_dir}")
    print("=" * 60)

    model, mean, std, n_channels, window_frames = load_model(args.model)
    print(f"\n  模型: InceptionTime, {n_channels}ch, {window_frames}frames")
    print(f"  mean/std: {mean.shape}")

    # 2. 加载数据
    data = load_sliced_sessions(args.data_dir, args.session)
    if not data:
        print("  [X] 未找到任何 session 数据!")
        return
    print(f"\n  加载了 {len(data)} 个 session")

    # 3. 逐 session 分析
    all_session_stats = []
    total_fp_cases = 0
    total_fn_cases = 0

    fp_dir = os.path.join(args.save, "false_positives")
    fn_dir = os.path.join(args.save, "false_negatives")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    print(f"\n  {'─' * 80}")
    print(f"  {'Session':<22s} {'Total':>6s} {'TP':>6s} {'TN':>6s} "
          f"{'FP':>6s} {'FN':>6s} {'FP%':>7s} {'FN%':>7s} {'Acc':>7s}")
    print(f"  {'─' * 80}")

    for name in sorted(data.keys()):
        X = data[name]["X"]       # (N, C, T)
        y = data[name]["y"]       # (N,)

        # 推理
        probs = run_inference(model, X, mean, std)
        pred = probs.argmax(axis=1)
        p_hit = probs[:, 1]

        # 统计
        tp = int(((y == 1) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        total = len(y)
        n_hit = int((y == 1).sum())
        n_bg = int((y == 0).sum())

        fp_rate = fp / max(n_bg, 1)
        fn_rate = fn / max(n_hit, 1)
        acc = (tp + tn) / max(total, 1)

        # 收集概率分布
        correct_hit_probs = p_hit[(y == 1) & (pred == 1)].tolist()
        correct_bg_probs = p_hit[(y == 0) & (pred == 0)].tolist()
        fp_probs = p_hit[(y == 0) & (pred == 1)].tolist()
        fn_probs = p_hit[(y == 1) & (pred == 0)].tolist()

        stats = {
            "name": name, "total": total, "tp": tp, "tn": tn,
            "fp_count": fp, "fn_count": fn,
            "fp_rate": fp_rate, "fn_rate": fn_rate, "acc": acc,
            "correct_hit_probs": correct_hit_probs,
            "correct_bg_probs": correct_bg_probs,
            "fp_probs": fp_probs, "fn_probs": fn_probs,
        }
        all_session_stats.append(stats)

        total_fp_cases += fp
        total_fn_cases += fn

        print(f"  {name:<22s} {total:6d} {tp:6d} {tn:6d} "
              f"{fp:6d} {fn:6d} {fp_rate*100:6.1f}% {fn_rate*100:6.1f}% {acc*100:6.1f}%")

        # 找 badcase 并画图
        badcases = find_badcases(y, probs)

        # FP 图
        fp_to_plot = badcases["fp"][:args.max_cases]
        for rank, (idx, prob) in enumerate(fp_to_plot):
            save_path = os.path.join(fp_dir, f"{name}_fp_{rank+1:03d}.png")
            plot_single_badcase(X[idx], y[idx], prob, name, rank + 1, "FP", save_path)

        # FN 图
        fn_to_plot = badcases["fn"][:args.max_cases]
        for rank, (idx, prob) in enumerate(fn_to_plot):
            save_path = os.path.join(fn_dir, f"{name}_fn_{rank+1:03d}.png")
            plot_single_badcase(X[idx], y[idx], prob, name, rank + 1, "FN", save_path)

        if fp > 0 or fn > 0:
            fp_plotted = min(fp, args.max_cases)
            fn_plotted = min(fn, args.max_cases)
            print(f"         → FP: {fp}个 (画了{fp_plotted}), FN: {fn}个 (画了{fn_plotted})")

    print(f"  {'─' * 80}")

    # 4. 汇总图
    total_cases = sum(s["total"] for s in all_session_stats)
    total_tp = sum(s["tp"] for s in all_session_stats)
    total_tn = sum(s["tn"] for s in all_session_stats)
    total_fp = sum(s["fp_count"] for s in all_session_stats)
    total_fn = sum(s["fn_count"] for s in all_session_stats)

    print(f"\n  {'=' * 50}")
    print(f"  全量汇总 (N={total_cases:,})")
    print(f"  TP={total_tp}, TN={total_tn}, FP={total_fp}, FN={total_fn}")
    print(f"  准确率: {(total_tp+total_tn)/max(total_cases,1)*100:.2f}%")
    print(f"  FP 率:  {total_fp/max(total_fp+total_tn,1)*100:.2f}%")
    print(f"  FN 率:  {total_fn/max(total_fn+total_tp,1)*100:.2f}%")
    print(f"  {'=' * 50}")

    plot_summary(all_session_stats, args.save)

    print(f"\n  FP 单图: {fp_dir}/ ({total_fp_cases} total, "
          f"每个session最多{args.max_cases}张)")
    print(f"  FN 单图: {fn_dir}/ ({total_fn_cases} total, "
          f"每个session最多{args.max_cases}张)")
    print(f"\n  完成!")


if __name__ == "__main__":
    main()
