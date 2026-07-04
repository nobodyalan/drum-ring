"""
双手机物理延迟测量
==================
同时监听两台手机 (不同端口) 的 IMU 数据, 通过比对踩踏冲击
的时间差, 精确测量脚底→裤兜的物理延迟。

原理:
  手机A (绑脚底) ──WiFi──┐
                          ├── PC 同时收包, 同台时钟打时间戳
  手机B (在裤兜) ──WiFi──┘

  同一 WiFi 下两个包的传输延迟几乎相等, 相减后抵消:
    时间差 = (冲击到达B − 冲击到达A) + (WiFi_B − WiFi_A)
           ≈ 物理延迟 ± 1.2ms

设置:
  手机A: ZigSim → PC_IP:8888
  手机B: ZigSim → PC_IP:8889

用法:
  python measure_latency.py                 # 终端显示
  python measure_latency.py --plot          # 实时可视化 + 终端
  python measure_latency.py --record        # 同时保存原始数据到 latency_data/
"""

import sys
import os
import time
import struct
import socket
import threading
import argparse
from collections import deque

import numpy as np

# 必须先设置 matplotlib 后端
import matplotlib
matplotlib.use('TkAgg')

# ==================== 默认参数 (与 realtime_simple.py 完全一致) ====================
PORT_FOOT = 8888
PORT_POCKET = 8889
RATIO_THRESH = 2.5    # STA/LTA 触发比值
MAG_HIGH = 0.8         # mag 高阈值
MAG_LOW = 0.3          # mag 释放低阈值
JERK_THRESH = 0.35     # jerk 快速通道阈值
FAST_RATIO = 1.2       # 快速通道最低 ratio
RESET_THRESH = 1.5     # ratio 低于此值释放滞回
QUIET_FRAMES = 5       # ratio < reset_thresh 持续帧数
COOLDOWN_MS = 150
PAIR_WINDOW_S = 0.5
PLOT_HISTORY_S = 5


# ==================== OSC 解析 ====================

def parse_osc_bundle(raw: bytes) -> tuple:
    """解析 OSC Bundle (ZigSim 格式). 返回 (gx,gy,gz, ax,ay,az)"""
    data = memoryview(raw)
    acc = gyr = None
    try:
        offset = 16 if bytes(data[:8]) == b"#bundle\0" else 0
        is_b = (offset == 16)
        while True:
            if is_b:
                if offset + 4 > len(data):
                    break
                el = struct.unpack(">I", data[offset:offset + 4])[0]
                offset += 4
                chunk = data[offset:offset + el]
                offset += el
            else:
                chunk = data
            ae = chunk.tobytes().find(b"\0")
            if ae <= 0:
                break
            astr = bytes(chunk[:ae]).decode("ascii", errors="replace")
            ao = (ae + 4) & ~3
            te = chunk[ao:].tobytes().find(b"\0")
            if te <= 0:
                break
            do = (ao + te + 4) & ~3
            if do + 12 <= len(chunk):
                vals = struct.unpack(">fff", chunk[do:do + 12])
                if "/accel" in astr:
                    acc = (float(vals[0]), float(vals[1]), float(vals[2]))
                elif "/gyro" in astr:
                    gyr = (float(vals[0]), float(vals[1]), float(vals[2]))
            if not is_b:
                break
    except Exception:
        pass
    gx, gy, gz = gyr or (0.0, 0.0, 0.0)
    ax, ay, az = acc or (0.0, 0.0, 0.0)
    return gx, gy, gz, ax, ay, az


# ==================== 单通道检测器 ====================

class ChannelDetector:
    """
    与 realtime_simple.SimpleDetector 检测逻辑完全一致,
    仅移除音频输出和独立线程, 改为每帧 process_frame() 调用的接口。
    """

    def __init__(self, label: str,
                 ratio_thresh: float = RATIO_THRESH,
                 mag_high: float = MAG_HIGH,
                 mag_low: float = MAG_LOW,
                 jerk_thresh: float = JERK_THRESH,
                 fast_ratio: float = FAST_RATIO,
                 reset_thresh: float = RESET_THRESH,
                 quiet_frames: int = QUIET_FRAMES,
                 cooldown_ms: int = COOLDOWN_MS,
                 record_data: bool = False,
                 plot_history: bool = False):
        self.label = label
        self.ratio_thresh = ratio_thresh
        self.mag_high = mag_high
        self.mag_low = mag_low
        self.jerk_thresh = jerk_thresh
        self.fast_ratio = fast_ratio
        self.reset_thresh = reset_thresh
        self.quiet_frames = quiet_frames
        self.cooldown_s = cooldown_ms / 1000.0
        self.record_data = record_data
        self.plot_history_enabled = plot_history

        # --- 状态变量 (与 SimpleDetector 完全一致) ---
        self.mag_history: deque = deque(maxlen=30)
        self.armed = False
        self.quiet_count = 0
        self.mag_ready = True
        self.last_trigger_time = 0.0
        self.prev_mag = 0.0

        self.hit_times: deque = deque(maxlen=100)
        self.raw_data: list = [] if record_data else None
        self.plot_data: deque = deque(maxlen=PLOT_HISTORY_S * 60)

        self.lock = threading.Lock()

    def process_frame(self, ax: float, ay: float, az: float) -> float | None:
        """
        输入一帧加速度, 执行与 SimpleDetector.run() 完全相同的检测逻辑。
        返回: 触发时返回时间戳, 否则返回 None
        """
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        now = time.time()

        with self.lock:
            self.mag_history.append(mag)

            if self.record_data:
                self.raw_data.append((now, mag))
            if len(self.mag_history) < 2:
                return None

            # STA/LTA ratio (与 realtime_simple 完全一致)
            short = np.mean(list(self.mag_history)[-2:])
            long = np.mean(self.mag_history)
            ratio = short / (long + 1e-8)

            # jerk
            jerk = mag - self.prev_mag
            self.prev_mag = mag

            if self.plot_history_enabled:
                self.plot_data.append((now, mag, ratio, jerk, self.armed, False))

            in_cooldown = (now - self.last_trigger_time) < self.cooldown_s

            # 双通道检测
            trig_main = ratio > self.ratio_thresh and mag > self.mag_high
            trig_fast = jerk > self.jerk_thresh and ratio > self.fast_ratio

            # 加速度双阈值: mag < mag_low 时重新就绪
            if not self.mag_ready and mag < self.mag_low:
                self.mag_ready = True

            is_hit = False
            if not self.armed:
                if (trig_main or trig_fast) and not in_cooldown and self.mag_ready:
                    self.armed = True
                    self.mag_ready = False
                    self.quiet_count = 0
                    self.last_trigger_time = now
                    self.hit_times.append(now)
                    is_hit = True
            else:
                if ratio < self.reset_thresh:
                    self.quiet_count += 1
                    if self.quiet_count >= self.quiet_frames:
                        self.armed = False
                        self.quiet_count = 0
                else:
                    self.quiet_count = 0

            # 标记绘图数据中的触发帧
            if is_hit and self.plot_history_enabled and self.plot_data:
                t, m, r, j, a, _ = self.plot_data[-1]
                self.plot_data[-1] = (t, m, r, j, a, True)

            return now if is_hit else None

    def get_recent_hit(self, window_s: float) -> float | None:
        with self.lock:
            if not self.hit_times:
                return None
            latest = self.hit_times[-1]
            if time.time() - latest < window_s:
                return latest
            return None

    def get_plot_data(self):
        with self.lock:
            return list(self.plot_data)


# ==================== 配对与统计 ====================

class LatencyMeasurer:
    """管理两个通道, 配对事件, 统计延迟"""

    def __init__(self, plot_enabled: bool = False):
        self.lock = threading.Lock()
        self.delays: list = []        # 所有延迟样本 (ms)
        self.pair_count = 0
        self.last_report_time = time.time()
        self.plot_enabled = plot_enabled
        # 绘图用: (t_foot, t_pocket, delay_ms)
        self.paired_events: list = []
        # 配对状态机: 记录待配对 foot 触发时间
        self._pending_foot: float | None = None

    def try_pair(self, ch_foot: ChannelDetector, ch_pocket: ChannelDetector):
        """
        有序配对: foot 先触发 → pocket 后触发 → 绑定一对。
        防止 foot 二次触发 (回弹) 被错误匹配到 pocket 的第一次触发。
        """
        t_foot = ch_foot.get_recent_hit(PAIR_WINDOW_S)
        t_pocket = ch_pocket.get_recent_hit(PAIR_WINDOW_S)

        # foot 触发: 记录为待配对
        if t_foot is not None:
            with self.lock:
                self._pending_foot = t_foot

        # pocket 触发: 与待配对的 foot 匹配
        if t_pocket is not None:
            with self.lock:
                pending = self._pending_foot
            if pending is not None:
                delta_s = t_pocket - pending
                # foot 必须在 pocket 之前, 且间隔在合理范围内
                if 0 < delta_s < PAIR_WINDOW_S:
                    delay_ms = delta_s * 1000.0
                    with self.lock:
                        self.delays.append(delay_ms)
                        self.pair_count += 1
                        if self.plot_enabled:
                            self.paired_events.append((pending, t_pocket, delay_ms))
                        self._pending_foot = None  # 已配对, 清空

                    print(f"\n  PAIR #{self.pair_count}: "
                          f"foot={pending:.3f}  pocket={t_pocket:.3f}  "
                          f"delay={delay_ms:+.0f}ms", flush=True)

    def get_stats_snapshot(self):
        """线程安全获取当前统计快照"""
        with self.lock:
            return (list(self.delays), list(self.paired_events))


# ==================== OSC 监听 ====================

def osc_listener(port: int, label: str, detector: ChannelDetector):
    """后台线程: 监听指定端口的 OSC 数据"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        sock.settimeout(0.1)
        print(f"[{label}] 监听 0.0.0.0:{port}")
    except Exception as e:
        print(f"[{label}] 端口 {port} 绑定失败: {e}")
        return

    pkt_count = 0
    while True:
        try:
            raw, addr = sock.recvfrom(65536)
        except socket.timeout:
            continue
        except Exception:
            break

        _, _, _, ax, ay, az = parse_osc_bundle(raw)
        detector.process_frame(ax, ay, az)

        pkt_count += 1
        if pkt_count == 1:
            print(f"[{label}] 第1帧! from {addr} "
                  f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f})")

    sock.close()
    print(f"[{label}] 已关闭 ({pkt_count} 包)")


# ==================== 实时绘图 ====================

def run_plot(ch_foot: ChannelDetector, ch_pocket: ChannelDetector,
             measurer: LatencyMeasurer, mag_high: float, running_flag: list):
    """后台线程: 实时显示 foot + pocket 加速度波形, 标注触发位置"""
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore", message=".*outside of the main thread.*")

    # 尝试加载中文字体, 回退到英文
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    plt.ion()
    fig, (ax_mag, ax_delay) = plt.subplots(2, 1, figsize=(14, 7),
                                           height_ratios=[3, 1],
                                           sharex=False)

    line_foot, = ax_mag.plot([], [], "#3498db", lw=1.0, label="Foot (ankle)", alpha=0.85)
    line_pocket, = ax_mag.plot([], [], "#e74c3c", lw=1.0, label="Pocket (hip)", alpha=0.85)
    ax_mag.axhline(y=mag_high, color="gray", ls=":", lw=0.8, alpha=0.5,
                   label=f"threshold={mag_high}g")
    ax_mag.set_ylabel("accel mag (g)")
    ax_mag.set_ylim(-0.5, 5)
    ax_mag.legend(loc="upper left", fontsize=8)
    ax_mag.grid(True, alpha=0.15)
    ax_mag.set_title("Foot vs Pocket Acceleration  (", fontsize=11)

    # 触发散点
    foot_scatter = ax_mag.scatter([], [], c="#2980b9", s=50, zorder=6, marker='v',
                                  edgecolors='white', linewidth=0.5)
    pocket_scatter = ax_mag.scatter([], [], c="#c0392b", s=50, zorder=6, marker='^',
                                    edgecolors='white', linewidth=0.5)

    # 动态元素: 触发竖线 + 配对连线 + 标注
    dyn_elems = []

    ax_delay.set_ylabel("delay (ms)")
    ax_delay.set_xlabel("pair #")
    ax_delay.set_ylim(-50, 300)
    ax_delay.axhline(y=0, color="gray", ls=":", lw=0.5)
    ax_delay.grid(True, alpha=0.15)
    ax_delay.set_title("Paired Delay per Stomp", fontsize=10)
    delay_scatter, = ax_delay.plot([], [], "o", color="#8e44ad", ms=5,
                                   label="per-stomp")
    delay_mean_line = None

    plt.tight_layout()

    while running_flag[0]:
        foot_data = ch_foot.get_plot_data()
        pocket_data = ch_pocket.get_plot_data()
        delays, paired = measurer.get_stats_snapshot()

        now = time.time()
        window_start = now - PLOT_HISTORY_S

        # --- 清除上一帧的动态元素 ---
        for e in dyn_elems:
            e.remove()
        dyn_elems.clear()

        # --- 加速度波形 ---
        # Foot (蓝色)
        if foot_data:
            ft = np.array([d[0] for d in foot_data])
            fm = np.array([d[1] for d in foot_data])
            fh = np.array([d[5] for d in foot_data])   # is_hit at index 5
            mask = ft >= window_start
            line_foot.set_data(ft[mask] - window_start, fm[mask])
            # 触发散点
            hit_mask = fh[mask]
            if hit_mask.any():
                foot_scatter.set_offsets(np.column_stack([
                    ft[mask][hit_mask] - window_start,
                    fm[mask][hit_mask]
                ]))
            else:
                foot_scatter.set_offsets(np.empty((0, 2)))
            # 触发竖线 (蓝色虚线)
            hit_times = ft[mask][hit_mask]
            hit_mags = fm[mask][hit_mask]
            for ht, hm in zip(hit_times, hit_mags):
                vl = ax_mag.axvline(x=ht - window_start, ymin=0,
                                    ymax=hm / ax_mag.get_ylim()[1],
                                    color="#3498db", ls="--", lw=1.2, alpha=0.5,
                                    zorder=3)
                dyn_elems.append(vl)

        # Pocket (红色)
        if pocket_data:
            pt = np.array([d[0] for d in pocket_data])
            pm = np.array([d[1] for d in pocket_data])
            ph = np.array([d[5] for d in pocket_data])   # is_hit at index 5
            mask = pt >= window_start
            line_pocket.set_data(pt[mask] - window_start, pm[mask])
            hit_mask = ph[mask]
            if hit_mask.any():
                pocket_scatter.set_offsets(np.column_stack([
                    pt[mask][hit_mask] - window_start,
                    pm[mask][hit_mask]
                ]))
            else:
                pocket_scatter.set_offsets(np.empty((0, 2)))
            # 触发竖线 (红色虚线)
            hit_times = pt[mask][hit_mask]
            hit_mags = pm[mask][hit_mask]
            for ht, hm in zip(hit_times, hit_mags):
                vl = ax_mag.axvline(x=ht - window_start, ymin=0,
                                    ymax=hm / ax_mag.get_ylim()[1],
                                    color="#e74c3c", ls="--", lw=1.2, alpha=0.5,
                                    zorder=3)
                dyn_elems.append(vl)

        # 配对紫色虚线
        for t_f, t_p, d_ms in paired:
            if t_f > window_start:
                f_mag, p_mag = None, None
                if foot_data:
                    for d in foot_data:
                        if abs(d[0] - t_f) < 0.02:
                            f_mag = d[1]; break
                if pocket_data:
                    for d in pocket_data:
                        if abs(d[0] - t_p) < 0.02:
                            p_mag = d[1]; break
                if f_mag is None:
                    f_mag = mag_high + 0.5
                if p_mag is None:
                    p_mag = mag_high + 0.5

                y_top = max(f_mag, p_mag) + 0.8
                ln = ax_mag.plot(
                    [t_f - window_start, t_p - window_start],
                    [y_top, y_top],
                    color="#8e44ad", lw=1.5, alpha=0.7, ls="-",
                    zorder=4
                )
                dyn_elems.extend(ln)
                # 两端的短竖线
                for tx, my, c in [(t_f, f_mag, "#3498db"), (t_p, p_mag, "#e74c3c")]:
                    vl = ax_mag.plot(
                        [tx - window_start, tx - window_start],
                        [my, y_top],
                        color=c, lw=0.8, alpha=0.5, ls=":",
                        zorder=4
                    )
                    dyn_elems.extend(vl)
                # 标注
                mid = (t_f + t_p) / 2 - window_start
                ann = ax_mag.annotate(
                    f"{d_ms:+.0f}ms",
                    xy=(mid, y_top + 0.15),
                    fontsize=8, color="#8e44ad", ha="center",
                    fontweight="bold", alpha=0.9
                )
                dyn_elems.append(ann)

        ax_mag.set_xlim(0, PLOT_HISTORY_S)

        # --- 延迟散点图 ---
        if delays:
            delay_scatter.set_data(range(1, len(delays) + 1), delays)
            ax_delay.set_xlim(0, max(len(delays) + 1, 10))
            if delay_mean_line:
                delay_mean_line.remove()
            mean_val = np.mean(delays)
            delay_mean_line = ax_delay.axhline(
                y=mean_val, color="#e74c3c", ls="--", lw=1, alpha=0.6,
                label=f"mean={mean_val:.0f}ms"
            )
            ax_delay.legend(loc="upper right", fontsize=7)

        # 更新标题显示最新延迟
        latest = f"{paired[-1][2]:+.0f}ms" if paired else "--"
        ax_mag.set_title(
            f"Foot vs Pocket Acceleration  (latest delay: {latest})", fontsize=11)

        try:
            plt.pause(0.1)
        except Exception:
            break

    plt.close()


# ==================== 主循环 ====================

def main():
    parser = argparse.ArgumentParser(
        description="双手机物理延迟测量 — 同时监听两台 ZigSim 手机",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
设置:
  手机A (绑脚底/脚踝): ZigSim → PC_IP:8888
  手机B (在裤兜):      ZigSim → PC_IP:8889

示例:
  python measure_latency.py                     # 终端模式
  python measure_latency.py --plot              # 实时可视化 + 终端
  python measure_latency.py --record --plot     # 可视化 + 保存原始数据
        """,
    )
    parser.add_argument("--port-foot", type=int, default=PORT_FOOT,
                        help=f"手机A (脚底) 端口 (默认: {PORT_FOOT})")
    parser.add_argument("--port-pocket", type=int, default=PORT_POCKET,
                        help=f"手机B (裤兜) 端口 (默认: {PORT_POCKET})")
    parser.add_argument("--ratio-thresh", type=float, default=RATIO_THRESH,
                        help=f"STA/LTA 触发比值 (默认: {RATIO_THRESH})")
    parser.add_argument("--mag-high", type=float, default=MAG_HIGH,
                        help=f"mag 触发高阈值 g (默认: {MAG_HIGH})")
    parser.add_argument("--mag-low", type=float, default=MAG_LOW,
                        help=f"mag 释放低阈值 g (默认: {MAG_LOW})")
    parser.add_argument("--jerk-thresh", type=float, default=JERK_THRESH,
                        help=f"jerk 快速通道阈值 g/帧 (默认: {JERK_THRESH})")
    parser.add_argument("--fast-ratio", type=float, default=FAST_RATIO,
                        help=f"快速通道最低 ratio (默认: {FAST_RATIO})")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_MS,
                        help=f"触发后冷却 ms (默认: {COOLDOWN_MS})")
    parser.add_argument("--record", action="store_true",
                        help="保存原始数据到 latency_data/ 目录")
    parser.add_argument("--plot", action="store_true",
                        help="实时可视化: foot+pocket 加速度波形 + 延迟统计")
    args = parser.parse_args()

    print("=" * 55)
    print("  双手机物理延迟测量")
    print("=" * 55)
    print(f"  检测参数 (与 realtime_simple.py 一致):")
    print(f"    主通道: ratio > {args.ratio_thresh} AND mag > {args.mag_high}g")
    print(f"    快速通道: jerk > {args.jerk_thresh}g/帧 AND ratio > {args.fast_ratio}")
    print(f"    加速度滞回: mag < {args.mag_low}g 释放")
    print(f"    ratio 释放: < {RESET_THRESH} 持续 {QUIET_FRAMES} 帧")
    print(f"    冷却: {args.cooldown}ms")
    print(f"  配对窗口: {PAIR_WINDOW_S*1000:.0f}ms")
    if args.plot:
        print(f"  可视化: 最近 {PLOT_HISTORY_S}s 波形")
    print()

    # 数据记录目录
    record_dir = None
    if args.record:
        record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "latency_data")
        os.makedirs(record_dir, exist_ok=True)
        print(f"  原始数据将保存到: {record_dir}/\n")

    # 两个通道使用完全相同的检测参数
    det_kwargs = dict(
        ratio_thresh=args.ratio_thresh,
        mag_high=args.mag_high,
        mag_low=args.mag_low,
        jerk_thresh=args.jerk_thresh,
        fast_ratio=args.fast_ratio,
        cooldown_ms=args.cooldown,
        record_data=args.record,
        plot_history=args.plot,
    )
    ch_foot = ChannelDetector("foot", **det_kwargs)
    ch_pocket = ChannelDetector("pocket", **det_kwargs)

    measurer = LatencyMeasurer(plot_enabled=args.plot)

    # 启动 OSC 监听线程
    t1 = threading.Thread(target=osc_listener,
                          args=(args.port_foot, "FOOT", ch_foot),
                          daemon=True)
    t2 = threading.Thread(target=osc_listener,
                          args=(args.port_pocket, "POCKET", ch_pocket),
                          daemon=True)
    t1.start()
    t2.start()
    time.sleep(0.5)

    # 启动绘图线程
    plot_running = [True]
    if args.plot:
        plot_thread = threading.Thread(
            target=run_plot,
            args=(ch_foot, ch_pocket, measurer, args.mag_high, plot_running),
            daemon=True)
        plot_thread.start()

    print("  开始踩踏! Ctrl+C 停止并输出统计\n")

    # 主循环: 持续尝试配对
    try:
        while True:
            time.sleep(0.01)
            measurer.try_pair(ch_foot, ch_pocket)

            # 每 5 秒更新统计
            now = time.time()
            if now - measurer.last_report_time > 5.0 and measurer.delays:
                with measurer.lock:
                    delays = np.array(measurer.delays)
                if len(delays) > 0:
                    print(f"\n  --- 5s 统计 ({len(delays)} 样本) ---")
                    print(f"  mean={np.mean(delays):.0f}ms  "
                          f"median={np.median(delays):.0f}ms  "
                          f"std={np.std(delays):.0f}ms  "
                          f"min={np.min(delays):.0f}ms  max={np.max(delays):.0f}ms")
                measurer.last_report_time = now

    except KeyboardInterrupt:
        pass

    plot_running[0] = False

    # 输出最终统计
    print("\n\n" + "=" * 55)
    print("  最终统计")
    print("=" * 55)

    with measurer.lock:
        delays = np.array(measurer.delays)

    if len(delays) == 0:
        print("\n  没有配对到踩踏事件。")
        print("  检查:")
        print("    1. 两台手机是否都在发送 ZigSim?")
        print("    2. 端口是否正确? (脚底=8888, 裤兜=8889)")
        print("    3. 阈值是否太高? 试试 --mag-high 0.4 --jerk-thresh 0.2")
    else:
        print(f"\n  配对次数: {len(delays)}")
        print(f"  延迟 (脚底→裤兜):")
        print(f"    均值:    {np.mean(delays):.0f} ms")
        print(f"    中位数:  {np.median(delays):.0f} ms")
        print(f"    标准差:  {np.std(delays):.1f} ms")
        print(f"    最小值:  {np.min(delays):.0f} ms")
        print(f"    最大值:  {np.max(delays):.0f} ms")
        print(f"    范围:    [{np.min(delays):.0f}, {np.max(delays):.0f}] ms")

    # 保存原始数据
    if args.record and ch_foot.raw_data and ch_pocket.raw_data:
        foot_path = os.path.join(record_dir, "foot_raw.csv")
        pocket_path = os.path.join(record_dir, "pocket_raw.csv")
        delays_path = os.path.join(record_dir, "delays.csv")

        with open(foot_path, "w") as f:
            f.write("timestamp,mag\n")
            for t, mag in ch_foot.raw_data:
                f.write(f"{t:.6f},{mag:.4f}\n")

        with open(pocket_path, "w") as f:
            f.write("timestamp,mag\n")
            for t, mag in ch_pocket.raw_data:
                f.write(f"{t:.6f},{mag:.4f}\n")

        with open(delays_path, "w") as f:
            f.write("delay_ms\n")
            for d in delays:
                f.write(f"{d:.1f}\n")

        print(f"\n  原始数据已保存:")
        print(f"    {foot_path}")
        print(f"    {pocket_path}")
        print(f"    {delays_path}")


if __name__ == "__main__":
    main()
