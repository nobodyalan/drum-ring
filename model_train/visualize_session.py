"""
可视化单个 session 的 IMU 数据 + 踩踏时刻标注
==============================================
读取 train_data/ 下的一个 session，画出 6 通道 IMU 波形，
并在 Arduino 检测到的踩踏瞬间画竖线标注。

用法:
  python visualize_session.py w_0                          # 画 w_0 的全部数据
  python visualize_session.py w_0 --range 10 30            # 只画 10-30 秒
  python visualize_session.py w_0 --clicks 5 15            # 只看第 5-15 次踩踏
  python visualize_session.py w_0 --accel-only             # 只看加速度 3 通道
  python visualize_session.py --list                       # 列出所有可用 session
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
    imu_time = imu[:, 6] / 1_000_000

    # 时间对齐
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

    return {
        "imu": imu_data,
        "t_rel": imu_time,
        "click_rel": click_rel,
        "name": os.path.basename(session_dir),
    }


# ==================== 绘图 ====================

CHANNEL_NAMES = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
CHANNEL_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c"]
CHANNEL_UNITS = ["rad/s", "rad/s", "rad/s", "g", "g", "g"]


def plot_session(session: dict, t_range: tuple | None = None,
                 click_range: tuple | None = None,
                 accel_only: bool = False, title: str | None = None,
                 save_path: str | None = None):
    """核心绘图函数"""

    imu = session["imu"]
    t_rel = session["t_rel"]
    clicks = session["click_rel"]
    name = session["name"]

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

        # 踩踏竖线 + 半透明区域
        for c in clicks_in:
            ax.axvline(x=c, color="#e74c3c", lw=1.5, alpha=0.6, ls="-", zorder=3)
            ax.axvspan(c, c + 0.19, color="#e74c3c", alpha=0.06, zorder=0)

    # ---- 第三排: 合加速度幅值 (accel magnitude) ----
    ax = axes[-1]
    ax.plot(t_plot, accel_mag, color="#2c3e50", lw=1.0, alpha=0.9)
    ax.fill_between(t_plot, 0, accel_mag, color="#2c3e50", alpha=0.08)
    ax.set_ylabel("accel mag\n(g)", fontsize=10, fontweight="bold")
    ax.set_xlabel("time (s)", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.set_xlim(t_min, t_max)

    # 踩踏竖线 + 编号标注
    for ci, c in enumerate(clicks_in):
        ax.axvline(x=c, color="#e74c3c", lw=1.5, alpha=0.6, ls="-", zorder=3)
        ax.axvspan(c, c + 0.19, color="#e74c3c", alpha=0.06, zorder=0)

    # 在 magnitude 轴上标注每次踩踏的编号和时间偏移
    T0 = clicks_in[0] if len(clicks_in) > 0 else t_min
    for ci, c in enumerate(clicks_in):
        dt = c - T0
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else accel_mag.max() * 1.1
        ax.annotate(
            f"#{ci + 1}\n+{dt:.2f}s",
            xy=(c, y_top * 0.88),
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

    # ---- 标题 ----
    n_total = len(clicks)
    n_shown = len(clicks_in)
    info_parts = [f"session: {name}", f"{n_shown}/{n_total} clicks"]
    if click_range is not None:
        info_parts.append(f"click #{click_range[0]}-#{click_range[1] - 1}")
    elif t_range is not None:
        info_parts.append(f"t = [{t_min:.1f}s, {t_max:.1f}s]")

    suptitle = title or f"IMU Recording  —  {' | '.join(info_parts)}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)

    # 图例说明
    fig.text(0.99, 0.005,
             "red lines = Arduino click  |  pink shading = +190ms hit zone  |  "
             "accel mag = sqrt(ax^2 + ay^2 + az^2)",
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
  python visualize_session.py w_0                    # 画全部
  python visualize_session.py w_0 --range 5 20       # 5-20秒
  python visualize_session.py w_0 --clicks 10 25     # 第10-25次踩踏
  python visualize_session.py w_0 --accel-only       # 只看加速度
  python visualize_session.py w_0 --save plot.png    # 保存到文件
  python visualize_session.py --list                 # 列出可用session
        """,
    )
    parser.add_argument("session", nargs="?", default=None,
                        help="Session 名称 (例如 w_0, f_2, shaking_0)")
    parser.add_argument("--data-dir", default="train_data",
                        help="数据目录 (默认: train_data)")
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

    args = parser.parse_args()

    data_dir = args.data_dir

    # --list 模式
    if args.list:
        print(f"\n  {data_dir}/ 下的可用 session:\n")
        if not os.path.isdir(data_dir):
            print("  目录不存在!")
            return
        sessions = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        for s in sessions:
            path = os.path.join(data_dir, s)
            imu_files = [f for f in os.listdir(path) if "_imu_" in f and f.endswith(".npy")]
            click_files = [f for f in os.listdir(path) if "click_timestamps" in f and f.endswith(".npy")]
            has_data = "[OK]" if (imu_files and click_files) else "[NO]"
            print(f"    [{has_data}] {s}")
        print()
        return

    if args.session is None:
        parser.print_help()
        print("\n  请指定 session 名称, 或 --list 查看可用列表")
        return

    # 加载
    session_path = os.path.join(data_dir, args.session)
    if not os.path.isdir(session_path):
        print(f"❌ Session 不存在: {session_path}")
        print(f"   试试 --list 查看可用 session")
        return

    session = load_session(session_path)
    if session is None:
        print(f"❌ {args.session} 缺少 IMU 或 click_timestamps 文件")
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
    )


if __name__ == "__main__":
    main()
