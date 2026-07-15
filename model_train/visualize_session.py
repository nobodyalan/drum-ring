"""
可视化单个 session 的 IMU 数据 + 踩踏时刻标注
==============================================
读取 train_data/ 下的一个 session，画出 gyro/accel/gravity/derived 通道波形。
红色虚线 = Arduino 触发, 蓝色实线 = 峰值对齐参考点, 绿色区域 = hit zone.

────────────────────────────────────────────────────────────────────────────
参数:
  session                  Session 名称或路径
  --data-dir DIR           数据目录 (默认: train_data)
  --range T_MIN T_MAX      显示的时间范围 (秒)
  --clicks START END       显示第 START 到 END 次踩踏
  --accel-only             只看加速度通道
  --save PATH              保存图片到文件
  --title TITLE            自定义标题
  --list                   列出所有可用 session
  --pre MS                 hit zone 前向 ms (默认: 70)
  --post MS                hit zone 后向 ms (默认: 150)
  --search-pre-ms MS       峰值搜索前向 ms (默认: 0)
  --search-post-ms MS      峰值搜索后向 ms (默认: 100)
  --no-refine              不做峰值对齐，直接显示原始 Arduino 位置

用法:
  python visualize_session.py w_0
  python visualize_session.py w_0 --range 5 20
  python visualize_session.py w_0 --clicks 10 25
  python visualize_session.py w_0 --accel-only
  python visualize_session.py w_0 --pre 150 --post -20         # 零延迟 hit zone
  python visualize_session.py w_0 --search-post-ms 120         # 调整峰值搜索窗口
  python visualize_session.py w_0 --no-refine                  # 看原始 Arduino 位置
  python visualize_session.py w_0 --save plot.png
  python visualize_session.py --list
  python visualize_session.py --list --data-dir ../other_data
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ==================== 数据加载 (与 train_model.py 一致的逻辑) ====================

def load_session(session_dir: str) -> dict | None:
    files = os.listdir(session_dir)
    imu_files = [f for f in files if "_imu_" in f and f.endswith(".npy")]
    click_files = [f for f in files if "click_timestamps" in f and f.endswith(".npy")]
    if not imu_files or not click_files:
        return None

    imu = np.load(os.path.join(session_dir, imu_files[0]))
    clicks = np.load(os.path.join(session_dir, click_files[0]))

    imu_data = imu[:, :6].astype(np.float32)
    # 时间戳永远是最后一列 (旧格式 col 6, 新格式 col 13)
    imu_time = imu[:, -1] / 1_000_000

    # 时间对齐: 优先使用录制时保存的真实 t0
    t0_files = [f for f in files if "_t0_" in f and f.endswith(".npy")]
    if t0_files:
        t0 = np.load(os.path.join(session_dir, t0_files[0])).item()
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
    click_rel = click_rel[valid]

    n_total = len(clicks)
    is_positive = n_total > 0

    # 从目录名推断样本类型
    d_lower = os.path.basename(session_dir).lower()
    if any(k in d_lower for k in ["shaking", "no_move", "noise", "idle", "neg"]):
        sample_type = "negative"
    elif any(k in d_lower for k in ["weak", "strong", "w_", "s_", "f_"]):
        sample_type = "positive"
    else:
        sample_type = "positive" if is_positive else "unknown"

    return {
        "imu": imu_data,
        "t_rel": imu_time,
        "click_rel": click_rel,
        "name": os.path.basename(session_dir),
        "n_clicks": n_total,
        "is_positive": is_positive,
        "sample_type": sample_type,
    }


# ==================== 绘图 ====================

CHANNEL_NAMES = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
CHANNEL_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c"]
CHANNEL_UNITS = ["rad/s", "rad/s", "rad/s", "g", "g", "g"]


def refine_clicks_to_peaks(t_rel: np.ndarray, click_rel: np.ndarray,
                            imu_data: np.ndarray,
                            search_pre_ms: int = 0,
                            search_post_ms: int = 100) -> np.ndarray:
    """
    Align Arduino click to accel_mag global max in [click, click+120ms].

    True impact always within 0-120ms post-trigger. Bounce at 150ms+
    is naturally excluded. Global argmax is safe in this narrow window.
    """
    if len(click_rel) == 0:
        return click_rel.copy()

    ax = imu_data[:, 3].astype(np.float64)
    ay = imu_data[:, 4].astype(np.float64)
    az = imu_data[:, 5].astype(np.float64)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    refined = []
    for c in click_rel:
        # Time-mask for exact [click+pre, click+post] window
        mask = (t_rel >= c - search_pre_ms / 1000.0) &                (t_rel <= c + search_post_ms / 1000.0)
        if mask.any():
            peak_offset = np.argmax(accel_mag[mask])
            refined.append(t_rel[mask][peak_offset])
        else:
            refined.append(c)

    return np.array(refined, dtype=click_rel.dtype)



def plot_session(session: dict, t_range: tuple | None = None,
                 click_range: tuple | None = None,
                 accel_only: bool = False, title: str | None = None,
                 save_path: str | None = None,
                 pre_ms: int = 70, post_ms: int = 150,
                 search_pre_ms: int = 0, search_post_ms: int = 100,
                 no_refine: bool = False):
    """核心绘图函数.
       pre_ms/post_ms: 击打区扩展 ms, 控制绿色 hit zone 显示范围 (默认 70/150)."""

    imu = session["imu"]
    t_rel = session["t_rel"]
    clicks = session["click_rel"]
    name = session["name"]
    n_total = session["n_clicks"]
    is_positive = session["is_positive"]
    sample_type = session["sample_type"]

    # 时间范围
    if click_range is not None:
        c_start, c_end = click_range
        c_start = max(0, c_start)
        c_end = min(len(clicks), c_end)
        if c_start < len(clicks):
            t_min = clicks[c_start] - 0.3
            t_max = clicks[c_end - 1] + 0.5 if c_end > 0 else t_rel[-1]
        else:
            t_min, t_max = t_rel[0], t_rel[-1]
    elif t_range is not None:
        t_min, t_max = t_range
    else:
        t_min, t_max = t_rel[0], t_rel[-1]

    # 裁剪数据
    mask = (t_rel >= t_min) & (t_rel <= t_max)
    t_plot = t_rel[mask]
    imu_plot = imu[mask]
    clicks_in = clicks[(clicks >= t_min) & (clicks <= t_max)]

    # ── 峰值对齐: 将 Arduino 点击对齐到最近的 accel_mag 峰值 ──
    if not no_refine and len(clicks_in) > 0:
        refined_clicks = refine_clicks_to_peaks(
            t_rel, clicks_in, imu, search_pre_ms, search_post_ms)
        # 打印每次点击的偏移量
        shifts_ms = (refined_clicks - clicks_in) * 1000
        print(f"\n  峰值对齐偏移 (peak - Arduino):")
        for ci, s in enumerate(shifts_ms):
            print(f"    click #{ci+1}: {s:+.0f}ms")
        print(f"    均值: {shifts_ms.mean():+.0f}ms, 标准差: {shifts_ms.std():.0f}ms")
    else:
        refined_clicks = clicks_in.copy() if len(clicks_in) > 0 else clicks_in

    # 计算 accel_mag (合加速度幅值)
    accel_mag = np.sqrt(imu_plot[:, 3]**2 + imu_plot[:, 4]**2 + imu_plot[:, 5]**2)

    # ---- 决定子图布局 ----
    if accel_only:
        row_labels = [("accel", [3, 4, 5])]
    else:
        row_labels = [
            ("gyro",  [0, 1, 2]),
            ("accel", [3, 4, 5]),
        ]

    n_rows = len(row_labels) + 1  # +1 for accel_mag
    fig = plt.figure(figsize=(14, 3.2 * n_rows))

    # 用 GridSpec 让最后一排 (magnitude) 稍高
    from matplotlib.gridspec import GridSpec
    height_ratios = [1.0] * len(row_labels) + [1.3]
    gs = GridSpec(n_rows, 1, figure=fig, height_ratios=height_ratios, hspace=0.12)

    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]

    GYRO_COLORS = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db"}
    ACCEL_COLORS = {"x": "#e67e22", "y": "#9b59b6", "z": "#1abc9c"}

    # ---- 绘制每排子图 ----
    PRE_S = pre_ms / 1000   # 击打前 s
    POST_S = post_ms / 1000  # 击打后 s

    for row_idx, (label, chs) in enumerate(row_labels):
        ax = axes[row_idx]
        color_map = GYRO_COLORS if label == "gyro" else ACCEL_COLORS
        unit = "rad/s" if label == "gyro" else "g"

        for ch in chs:
            axis_letter = "xyz"[ch % 3]
            color = color_map[axis_letter]
            ax.plot(t_plot, imu_plot[:, ch], color=color, lw=0.7, alpha=0.85,
                    label=f"{axis_letter}")

        ax.set_ylabel(f"{label}\n({unit})", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, ncol=3,
                  framealpha=0.5, edgecolor="none")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.set_xlim(t_min, t_max)

        # ---- 标注正例 (Hit Zone: [peak-PRE_MS, peak+POST_MS]) ----
        for ci, c_orig in enumerate(clicks_in):
            c_peak = refined_clicks[ci]
            # 原始 Arduino 触发线 (红色虚线)
            ax.axvline(x=c_orig, color="#e74c3c", lw=1.2, alpha=0.55, ls="--", zorder=2)
            # 峰值对齐后的参考线 (蓝色实线)
            ax.axvline(x=c_peak, color="#2980b9", lw=1.8, alpha=0.75, ls="-", zorder=3)
            # Hit Zone 半透明绿色区域 = 正例标签 (基于峰值)
            ax.axvspan(c_peak - PRE_S, c_peak + POST_S, color="#27ae60", alpha=0.10, zorder=0)

    # ---- 第三排: 合加速度幅值 (accel magnitude) ----
    ax = axes[-1]
    ax.plot(t_plot, accel_mag, color="#2c3e50", lw=1.0, alpha=0.9)
    ax.fill_between(t_plot, 0, accel_mag, color="#2c3e50", alpha=0.08)
    ax.set_ylabel("accel mag\n(g)", fontsize=10, fontweight="bold")
    ax.set_xlabel("time (s)", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.set_xlim(t_min, t_max)

    # Hit Zone 标注 + 编号 (基于峰值对齐)
    for ci, c_orig in enumerate(clicks_in):
        c_peak = refined_clicks[ci]
        # 原始 Arduino 触发线 (红色虚线)
        ax.axvline(x=c_orig, color="#e74c3c", lw=1.2, alpha=0.55, ls="--", zorder=2)
        # 峰值对齐后的参考线 (蓝色实线)
        ax.axvline(x=c_peak, color="#2980b9", lw=1.5, alpha=0.7, ls="-", zorder=3)
        # Hit Zone 基于峰值
        ax.axvspan(c_peak - PRE_S, c_peak + POST_S, color="#27ae60", alpha=0.10, zorder=0)

    # 在 magnitude 轴上标注每次踩踏的编号和时间偏移 (基于峰值)
    T0 = refined_clicks[0] if len(refined_clicks) > 0 else t_min
    for ci, c_peak in enumerate(refined_clicks):
        dt = c_peak - T0
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else accel_mag.max() * 1.1
        ax.annotate(
            f"#{ci + 1}\n+{dt:.2f}s",
            xy=(c_peak, y_top * 0.88),
            fontsize=5.5, color="#c0392b", fontweight="bold",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#e74c3c", alpha=0.7, lw=0.6),
        )

    # ---- 精细化时间轴刻度 ----
    import matplotlib.ticker as ticker
    duration = t_max - t_min
    if duration <= 3:
        major_interval = 0.2
        minor_interval = 0.05
    elif duration <= 10:
        major_interval = 0.5
        minor_interval = 0.1
    elif duration <= 30:
        major_interval = 1.0
        minor_interval = 0.2
    elif duration <= 60:
        major_interval = 2.0
        minor_interval = 0.5
    elif duration <= 120:
        major_interval = 5.0
        minor_interval = 1.0
    else:
        major_interval = 10.0
        minor_interval = 2.0

    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_interval))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_interval))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        # 主刻度网格 (稍粗)
        ax.grid(True, which="major", alpha=0.25, lw=0.6)
        # 次刻度网格 (更细更淡)
        ax.grid(True, which="minor", alpha=0.12, lw=0.3)
        ax.tick_params(which="major", labelsize=8, length=5)
        ax.tick_params(which="minor", length=3)

    # ---- 标注正/负标签区域 ----
    # 正例 (Hit Zone): [click-10ms, click+190ms] = 绿色区域
    # 负例 (Background):  其余所有区域 = 无标记
    if is_positive:
        badge_text = f"POS={n_total} hits"
        badge_color = "#27ae60"   # green
    else:
        badge_text = "NEG sample"
        badge_color = "#e74c3c"   # red
        for ax in axes:
            ax.text(0.5, 0.5, "NO STOMP\n(All Background)",
                    transform=ax.transAxes, fontsize=28, color="#e74c3c",
                    ha="center", va="center", alpha=0.15, fontweight="bold",
                    rotation=15)

    # ---- 标题 ----
    n_shown = len(clicks_in)
    info_parts = [f"session: {name}", f"[{badge_text}]"]
    if is_positive:
        info_parts.append(f"{n_shown}/{n_total} clicks shown")
    if click_range is not None:
        info_parts.append(f"click #{click_range[0]}-#{click_range[1] - 1}")
    elif t_range is not None:
        info_parts.append(f"t = [{t_min:.1f}s, {t_max:.1f}s]")

    suptitle = title or f"IMU Recording  —  {' | '.join(info_parts)}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995, color=badge_color)

    # 图例说明
    if is_positive:
        footer_text = (f"[GREEN zone] = POSITIVE label (hit: peak-{pre_ms}ms ~ peak+{post_ms}ms)  |  "
                       "other = NEGATIVE label (background)  |  "
                       "red dashed = Arduino trigger  |  blue solid = accel_mag peak  |  "
                       "accel_mag = sqrt(ax^2+ay^2+az^2)")
    else:
        footer_text = ("ALL frames = NEGATIVE (background)  |  no stomp detected  |  "
                       "accel_mag = sqrt(ax^2+ay^2+az^2)")
    fig.text(0.99, 0.005, footer_text,
             ha="right", va="bottom", fontsize=7, color="gray",
             transform=fig.transFigure)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"saved: {save_path}")

    plt.show()
    return fig


# ==================== 命令行 ====================

def main():
    parser = argparse.ArgumentParser(
        description="可视化 IMU 录制 session + 踩踏标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_session.py w_0                    # 画全部 (默认 hit-zone: -70/+150ms)
  python visualize_session.py w_0 --range 5 20       # 5-20秒
  python visualize_session.py w_0 --clicks 10 25     # 第10-25次踩踏
  python visualize_session.py w_0 --accel-only       # 只看加速度
  python visualize_session.py w_0 --pre 50 --post 200  # 自定义 hit-zone 范围
  python visualize_session.py w_0 --save plot.png    # 保存到文件
  python visualize_session.py --list                 # 列出 train_data/ 下可用session
  python visualize_session.py --list --data-dir ../other_data  # 列出其他目录
        """,
    )
    parser.add_argument("session", nargs="?", default=None,
                        help="Session 名称或路径. "
                             "可以是名字 (如 w_0, 自动在 --data-dir 下查找), "
                             "也可以是相对/绝对路径 (如 ../other_data/w_0)")
    parser.add_argument("--data-dir", default="train_data",
                        help="数据目录, session 为名字时在此目录下查找 (默认: train_data)")
    parser.add_argument("--range", nargs=2, type=float, default=None,
                        metavar=("T_MIN", "T_MAX"),
                        help="显示的时间范围 (秒)")
    parser.add_argument("--clicks", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="显示第 START 到 END 次踩踏")
    parser.add_argument("--accel-only", action="store_true",
                        help="只看加速度 3 通道 (不画陀螺仪)")
    parser.add_argument("--save", type=str, default=None,
                        metavar="PATH",
                        help="保存图片到文件而不是显示")
    parser.add_argument("--title", type=str, default=None,
                        help="自定义标题")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用 session 并退出")
    parser.add_argument("--pre", type=int, default=70,
                        help="击打前扩展 ms, 控制绿色 hit-zone 起点 (默认: 70)")
    parser.add_argument("--post", type=int, default=150,
                        help="击打后扩展 ms, 控制绿色 hit-zone 终点 (默认: 150)")
    parser.add_argument("--search-pre-ms", type=int, default=0,
                        help="峰值搜索窗口前向 ms (默认: 50)")
    parser.add_argument("--search-post-ms", type=int, default=100,
                        help="峰值搜索窗口后向 ms (默认: 200)")
    parser.add_argument("--no-refine", action="store_true",
                        help="不做峰值对齐，直接用原始 Arduino 时间戳")

    args = parser.parse_args()

    data_dir = args.data_dir

    # --list 模式: 列出可用 session
    if args.list:
        # 支持列出任意目录 (包括 --data-dir 和直接路径)
        list_dir = args.data_dir
        if args.session and (os.sep in args.session or "/" in args.session):
            list_dir = args.session  # 直接把 session 当路径用

        print(f"\n  {list_dir}/ 下的可用 session:\n")
        if not os.path.isdir(list_dir):
            print(f"  目录不存在: {list_dir}")
            return
        sessions = sorted(
            d for d in os.listdir(list_dir)
            if os.path.isdir(os.path.join(list_dir, d))
        )
        if not sessions:
            print("  (无子目录)")
            return
        pos_count = 0
        neg_count = 0
        for s in sessions:
            path = os.path.join(list_dir, s)
            imu_files = [f for f in os.listdir(path) if "_imu_" in f and f.endswith(".npy")]
            click_files = [f for f in os.listdir(path) if "click_timestamps" in f and f.endswith(".npy")]
            has_data = "OK" if (imu_files and click_files) else "NO"
            n_clicks = 0
            if imu_files and click_files:
                clicks_arr = np.load(os.path.join(path, click_files[0]))
                n_clicks = len(clicks_arr)
                if n_clicks > 0:
                    tag = "+ POS"
                    pos_count += 1
                else:
                    tag = "- NEG"
                    neg_count += 1
            else:
                tag = "?"
            click_str = f"{n_clicks} clicks" if n_clicks > 0 else "no clicks"
            print(f"    [{has_data}] [{tag}] {s:25s} ({click_str})")
        print(f"\n  Positive: {pos_count}  |  Negative: {neg_count}")
        print()
        return

    if args.session is None:
        parser.print_help()
        print("\n  请指定 session 名称, 或 --list 查看可用列表")
        return

    # ── 解析 session 路径: 支持名字 (在 --data-dir 下查找) 和直接路径 ──
    if os.sep in args.session or "/" in args.session or os.path.isabs(args.session):
        # 直接路径 (含目录分隔符 or 绝对路径)
        session_path = args.session
    else:
        # 名字: 拼接 --data-dir
        session_path = os.path.join(data_dir, args.session)

    if not os.path.isdir(session_path):
        print(f"[X] Session 不存在: {session_path}")
        print(f"   试试 --list 查看 {data_dir}/ 下可用 session")
        print(f"   或直接传路径: python visualize_session.py ../other_data/w_0")
        return

    session = load_session(session_path)
    if session is None:
        print(f"[X] {args.session} 缺少 IMU 或 click_timestamps 文件")
        return

    print(f"\n  Session: {session['name']}")
    print(f"  IMU 帧数: {len(session['t_rel']):,}")
    print(f"  时长: {session['t_rel'][-1] - session['t_rel'][0]:.1f}s")
    print(f"  踩踏次数: {len(session['click_rel'])}")
    print()

    plot_session(
        session,
        t_range=tuple(args.range) if args.range else None,
        click_range=tuple(args.clicks) if args.clicks else None,
        accel_only=args.accel_only,
        title=args.title,
        save_path=args.save,
        pre_ms=args.pre,
        post_ms=args.post,
        search_pre_ms=args.search_pre_ms,
        search_post_ms=args.search_post_ms,
        no_refine=args.no_refine,
    )


if __name__ == "__main__":
    main()
