"""
实时踩踏检测 — X轴加速度 (STA/LTA + jerk)
==========================================
接收 iPhone ZigSim 发送的 IMU 数据 (UDP :8888),
仅使用 X 轴加速度 (ax) 替代合加速度, 做实时踩踏检测。

检测策略:
  - 仅用 ax (单轴绝对值) 计算 STA/LTA ratio + jerk
  - 主通道: ratio > ratio_thresh AND abs(ax) > mag_high
  - 快速通道: jerk > jerk_thresh AND ratio > fast_ratio
  - 加速度双阈值滞回: abs(ax) > mag_high 触发, abs(ax) < mag_low 释放
  - 任一满足 → 触发 WAV 音效 + 冷却
  - ratio < reset_thresh 持续 N 帧 → 滞回复位

用法:
  python realtime_mag_jerk.py -v
  python realtime_mag_jerk.py --plot
  python realtime_mag_jerk.py --mag-high 0.4 --mag-low 0.15 --jerk-thresh 0.2 --plot
  │      参数      │ 默认值 │
  ├────────────────┼────────┤
  │ --ratio-thresh │ 2.5    │
  ├────────────────┼────────┤
  │ --mag-high     │ 0.4    │
  ├────────────────┼────────┤
  │ --mag-low      │ 0.15   │
  ├────────────────┼────────┤
  │ --jerk-thresh  │ 0.2    │
  ├────────────────┼────────┤
  │ --cooldown     │ 150    │
  ├────────────────┼────────┤
  │ --sound-wav    │ 1.wav  │
  ├────────────────┼────────┤
  │ --fast-ratio   │ 1.2    │
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

# 必须先设置 matplotlib 后端, 避免后台线程 GUI 警告
import matplotlib
matplotlib.use('TkAgg')

# ==================== 默认参数 (X轴单轴, 阈值低于合加速度) ====================
OSC_PORT = 8888
RATIO_THRESH = 2

MAG_HIGH = 0.6

    # 向上提腿正向加速度 (预判, 比踩踏更早)
MAG_LOW = 0.15    # 释放低阈值 (回落到此值以下重新就绪)
JERK_THRESH = 2       # 正向 jerk (向上提腿)
JERK_DOWN_THRESH = 2.0  # 负向 jerk (向下踩踏, 绝对值)
COOLDOWN_MS = 120
FAST_RATIO = 1.2
SOUND_WAV = "1.wav"


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


# ==================== 音效 ====================

def play_sound(wav_path: str = "1.wav"):
    """播放触发音效。WAV → 系统音效 → 终端 BEL。"""
    try:
        import winsound
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, wav_path)
        if os.path.exists(full_path):
            winsound.PlaySound(full_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
    except Exception:
        pass
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass
    print("\a", end="", flush=True)


# ==================== 检测器 ====================

class SimpleDetector:
    """
    STA/LTA 能量比 + jerk 快速通道检测, 无需模型。

    主通道:   ratio > ratio_thresh AND mag > mag_high  (STA/LTA, 2~3帧确认)
    快速通道: jerk > jerk_thresh AND ratio > self.fast_ratio           (冲击第1帧生效)
    """

    def __init__(self,
                 ratio_thresh: float = RATIO_THRESH,
                 mag_high: float = MAG_HIGH,
                 mag_low: float = MAG_LOW,
                 jerk_thresh: float = JERK_THRESH,
                 jerk_down_thresh: float = JERK_DOWN_THRESH,
                 fast_ratio: float = FAST_RATIO,
                 reset_thresh: float = 1.5,
                 quiet_frames: int = 5,
                 cooldown_ms: int = COOLDOWN_MS,
                 sound_wav: str = SOUND_WAV):
        self.ratio_thresh = ratio_thresh
        self.mag_high = mag_high
        self.mag_low = mag_low
        self.jerk_thresh = jerk_thresh
        self.jerk_down_thresh = jerk_down_thresh
        self.fast_ratio = fast_ratio
        self.reset_thresh = reset_thresh
        self.quiet_frames = quiet_frames
        self.cooldown_s = cooldown_ms / 1000.0
        self.sound_wav = sound_wav

        self.buffer: deque = deque(maxlen=3)
        self.lock = threading.Lock()
        self.frame_count = 0
        self.running = True

        self.mag_history: deque = deque(maxlen=30)
        self.armed = False
        self.quiet_count = 0
        self.mag_ready = True  # 加速度双阈值: mag < mag_low 后重新就绪
        self.last_trigger_time = 0.0

        self.total_frames = 0
        self.total_triggers = 0
        self.prev_mag = 0.0
        self.max_mag = 0.0

        self.plot_history: deque = deque(maxlen=300)
        self.trigger_frames: deque = deque(maxlen=50)

    def add_imu_frame(self, gx, gy, gz, ax, ay, az):
        with self.lock:
            self.buffer.append([gx, gy, gz, ax, ay, az])
            self.frame_count += 1

    def run(self, verbose: bool = False):
        print(f"\n  实时踩踏检测 (X轴加速度, STA/LTA + jerk)")
        print(f"  主通道: ratio > {self.ratio_thresh:.1f} AND ax > {self.mag_high:.2f}g")
        print(f"  快速通道: jerk > {self.jerk_thresh:.2f} (向上) OR jerk < -{self.jerk_down_thresh:.1f} (向下) AND ratio > {self.fast_ratio}")
        print(f"  加速度滞回: ax < {self.mag_low:.2f}g 释放后重新就绪")
        print(f"  释放: ratio < {self.reset_thresh:.1f} 持续 {self.quiet_frames} 帧")
        print(f"  冷却: {self.cooldown_s * 1000:.0f}ms | 音效: {self.sound_wav}")
        print(f"  等待 IMU 数据...\n")

        last_frame_count = 0
        last_status = time.time()

        while self.running:
            time.sleep(0.001)

            with self.lock:
                cur_count = self.frame_count

            if cur_count == 0:
                if time.time() - last_status > 3.0:
                    print("  等待 IMU 数据... (未收到)")
                    last_status = time.time()
                continue

            if cur_count == last_frame_count:
                continue
            last_frame_count = cur_count

            # 取最新帧
            with self.lock:
                if len(self.buffer) == 0:
                    continue
                frame = self.buffer[-1]
                ax, ay, az = frame[3], frame[4], frame[5]

            mag = ax                  # 原始值: 阈值判断 + jerk
            mag_abs = abs(ax)          # 绝对值: STA/LTA ratio 计算
            self.total_frames += 1
            self.mag_history.append(mag_abs)
            if abs(mag) > self.max_mag:
                self.max_mag = abs(mag)

            if len(self.mag_history) < 2:
                continue

            # STA/LTA
            short = np.mean(list(self.mag_history)[-2:])
            long  = np.mean(self.mag_history)
            ratio = short / (long + 1e-8)

            # jerk
            jerk = mag - self.prev_mag
            self.prev_mag = mag

            self.plot_history.append((mag, ratio, jerk, self.armed))

            now = time.time()
            in_cooldown = (now - self.last_trigger_time) < self.cooldown_s

            trg_main = ratio > self.ratio_thresh and mag > self.mag_high
            trg_fast = (jerk > self.jerk_thresh or jerk < -self.jerk_down_thresh) and ratio > self.fast_ratio

            # 加速度双阈值: ax 回落到 mag_low 以下重新就绪
            if not self.mag_ready and mag < self.mag_low:
                self.mag_ready = True

            if not self.armed:
                if (trg_main or trg_fast) and not in_cooldown and self.mag_ready:
                    self.armed = True
                    self.mag_ready = False
                    self.quiet_count = 0
                    self.total_triggers += 1
                    channel = ("FAST-DOWN" if jerk < -self.jerk_down_thresh else "FAST-UP") if (trg_fast and not trg_main) else "STA/LTA"
                    print(f"\n  DETECT! (#{self.total_triggers}) [{channel}] | "
                          f"ratio={ratio:.1f} ax={mag:+.2f}g jerk={jerk:+.2f}")

                    threading.Thread(target=play_sound, args=(self.sound_wav,),
                                     daemon=True).start()

                    self.last_trigger_time = now
                    self.trigger_frames.append(self.total_frames)
            else:
                if ratio < self.reset_thresh:
                    self.quiet_count += 1
                    if self.quiet_count >= self.quiet_frames:
                        self.armed = False
                        self.quiet_count = 0
                else:
                    self.quiet_count = 0

            if verbose:
                self._print_verbose(ratio, mag, jerk)
            elif self.total_frames % 60 == 0:
                cdl = max(0.0, self.cooldown_s - (now - self.last_trigger_time))
                cd_str = f"cool {cdl:.1f}s" if cdl > 0.01 else "ready"
                state = "ARMED" if self.armed else "idle"
                print(f"  frame={self.total_frames:6d} | "
                      f"ratio={ratio:4.1f} ax={mag:+5.2f}g jerk={jerk:+5.2f} | "
                      f"{state:5s} | {cd_str} | trig={self.total_triggers}")

    def _print_verbose(self, ratio: float, mag: float, jerk: float):
        bar_len = int(min(ratio / self.ratio_thresh, 1.0) * 30)
        bar = ("=" * max(0, bar_len - 1) + ">") if bar_len > 0 else "|"
        bar = bar.ljust(30)

        trg_main = ratio > self.ratio_thresh and mag > self.mag_high
        trg_fast = (jerk > self.jerk_thresh or jerk < -self.jerk_down_thresh) and ratio > self.fast_ratio

        if self.armed:
            marker = "** HIT **"
        elif trg_main or trg_fast:
            marker = ">>TRIG<<"
        else:
            marker = "        "

        cdl = max(0.0, self.cooldown_s - (time.time() - self.last_trigger_time))
        cd = f"CD{cdl:.1f}" if cdl > 0.01 else "   "
        state = "A" if self.armed else ("." if self.mag_ready else "m")

        print(f"\r  [{bar}] r={ratio:3.1f} ax={mag:+.2f} j={jerk:+.2f} | "
              f"{state} {cd} | trig={self.total_triggers} | {marker}",
              end="", flush=True)

    def stop(self):
        self.running = False


# ==================== OSC 监听 ====================

def osc_listener(detector: SimpleDetector, port: int, debug: bool = False):
    """后台线程: 监听 UDP OSC 端口, 解析 IMU 并送入检测器"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        sock.settimeout(0.1)
        print(f"[OSC] 监听 UDP 0.0.0.0:{port}")
    except Exception as e:
        print(f"[OSC] 端口 {port} 绑定失败: {e}")
        detector.stop()
        return

    pkt_count = 0
    last_debug = time.time()

    while detector.running:
        try:
            raw, addr = sock.recvfrom(65536)
        except socket.timeout:
            continue
        except Exception:
            break

        pkt_count += 1
        gx, gy, gz, ax, ay, az = parse_osc_bundle(raw)
        detector.add_imu_frame(gx, gy, gz, ax, ay, az)

        if debug:
            now = time.time()
            if now - last_debug > 1.0:
                print(f"  [DEBUG] OSC #{pkt_count} from {addr}, "
                      f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f})")
                last_debug = now
        elif pkt_count == 1:
            print(f"[OSC] 第1帧! from {addr}, "
                  f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f})")

    sock.close()
    if pkt_count > 0:
        print(f"[OSC] 已关闭 ({pkt_count} 包)")
    else:
        print("[OSC] 已关闭 (无数据)")


# ==================== 实时绘图 ====================

def run_plot(detector: SimpleDetector, ratio_thresh: float,
             mag_high: float, mag_low: float, jerk_thresh: float):
    """后台线程: 实时显示 accel_mag + ratio + jerk."""
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax_mag, ax_ratio, ax_jerk) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Real-time X-Axis STA/LTA + Jerk Monitor", fontsize=13)

    # X轴加速度 (正=上, 负=下)
    line_mag, = ax_mag.plot([], [], "k-", lw=1.2, label="ax")
    ax_mag.axhline(y=mag_high, color="green", ls="--", lw=1, alpha=0.7,
                   label=f"mag_high={mag_high}")
    ax_mag.set_ylabel("ax (g)")
    ax_mag.set_ylim(-3, 3)
    ax_mag.axhline(y=mag_low, color="orange", ls="--", lw=1, alpha=0.7,
                   label=f"mag_low={mag_low}")
    ax_mag.legend(loc="upper right", fontsize=7)
    ax_mag.grid(True, alpha=0.2)

    # ratio
    line_ratio, = ax_ratio.plot([], [], "b-", lw=1.2, label="ratio")
    ax_ratio.axhline(y=ratio_thresh, color="red", ls="--", lw=1, alpha=0.7,
                     label=f"ratio={ratio_thresh}")
    ax_ratio.axhline(y=1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylim(-0.5, 10)
    ax_ratio.legend(loc="upper right", fontsize=7)
    ax_ratio.grid(True, alpha=0.2)

    # jerk
    line_jerk, = ax_jerk.plot([], [], "#e67e22", lw=1.0, label="jerk")
    ax_jerk.axhline(y=jerk_thresh, color="red", ls="--", lw=1, alpha=0.7,
                    label=f"jerk={jerk_thresh}")
    ax_jerk.axhline(y=0, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax_jerk.set_ylabel("jerk (g/frame)")
    ax_jerk.set_xlabel("Time (s)")
    ax_jerk.set_ylim(-2, 5)
    ax_jerk.legend(loc="upper right", fontsize=7)
    ax_jerk.grid(True, alpha=0.2)

    trigger_lines = []
    plt.tight_layout()

    while detector.running:
        with detector.lock:
            history = list(detector.plot_history)
            trig_frames = list(detector.trigger_frames)

        if len(history) > 2:
            t = np.arange(len(history)) / 60.0

            mag_vals   = np.array([h[0] for h in history])
            ratio_vals = np.array([h[1] for h in history])
            jerk_vals  = np.array([h[2] for h in history])

            line_mag.set_data(t, mag_vals)
            lo = min(-3, mag_vals.min() * 1.3)
            hi = max(3, mag_vals.max() * 1.3)
            ax_mag.set_ylim(lo, hi)

            line_ratio.set_data(t, ratio_vals)
            ax_ratio.set_ylim(-0.5, max(10, ratio_vals.max() * 1.3))

            line_jerk.set_data(t, jerk_vals)
            ax_jerk.set_ylim(min(-2, jerk_vals.min() * 1.3),
                             max(5, jerk_vals.max() * 1.3))

            ax_mag.set_xlim(t[0], t[-1] + 0.1)

            # 触发红竖线
            for vl in trigger_lines:
                vl.remove()
            trigger_lines.clear()
            frame_offset = detector.total_frames - len(history) + 1
            for tf in trig_frames:
                rel_frame = tf - frame_offset
                if 0 <= rel_frame < len(t):
                    tt = t[rel_frame]
                    for ax in (ax_mag, ax_ratio, ax_jerk):
                        vl = ax.axvline(x=tt, color="red", lw=1.2, alpha=0.5,
                                        ls="-", zorder=0)
                        trigger_lines.append(vl)

        try:
            plt.pause(0.1)
        except Exception:
            break

    plt.close()


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="实时踩踏检测 — X轴加速度 + STA/LTA + jerk (无需模型)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python realtime_mag_jerk.py -v
  python realtime_mag_jerk.py --mag-high 0.4 --mag-low 0.15 --jerk-thresh 0.2 -v
  python realtime_mag_jerk.py --plot
        """,
    )
    parser.add_argument("--port", type=int, default=OSC_PORT,
                        help=f"OSC UDP 端口 (默认: {OSC_PORT})")
    parser.add_argument("--ratio-thresh", type=float, default=RATIO_THRESH,
                        help=f"STA/LTA 触发比值 (默认: {RATIO_THRESH})")
    parser.add_argument("--mag-high", type=float, default=MAG_HIGH,
                        help=f"mag 触发高阈值 g (默认: {MAG_HIGH})")
    parser.add_argument("--mag-low", type=float, default=MAG_LOW,
                        help=f"mag 释放低阈值 g, 低于此值后重新就绪 (默认: {MAG_LOW})")
    parser.add_argument("--jerk-thresh", type=float, default=JERK_THRESH,
                        help=f"jerk 正向阈值 g/帧, 向上提腿 (默认: {JERK_THRESH})")
    parser.add_argument("--jerk-down-thresh", type=float, default=JERK_DOWN_THRESH,
                        help=f"jerk 负向阈值 g/帧, 向下踩踏 绝对值 (默认: {JERK_DOWN_THRESH})")
    parser.add_argument("--fast-ratio", type=float, default=FAST_RATIO,
                        help=f"快速通道最低 ratio (默认: {FAST_RATIO})")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_MS,
                        help=f"触发后冷却 ms (默认: {COOLDOWN_MS})")
    parser.add_argument("--quiet-frames", type=int, default=5,
                        help=f"ratio<reset_thresh 持续帧数才释放滞回 (默认: 5)")
    parser.add_argument("--sound-wav", type=str, default=SOUND_WAV,
                        help=f"触发音效 WAV 文件 (默认: {SOUND_WAV})")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式: 每秒打印 OSC 原始数据")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="实时模式: 单行动态刷新")
    parser.add_argument("--plot", action="store_true",
                        help="实时绘图: mag + ratio + jerk 波形")

    args = parser.parse_args()

    print("=" * 55)
    print("  实时踩踏检测 — X轴加速度 (STA/LTA + jerk)")
    print("=" * 55)

    detector = SimpleDetector(
        ratio_thresh=args.ratio_thresh,
        mag_high=args.mag_high,
        mag_low=args.mag_low,
        jerk_thresh=args.jerk_thresh,
        jerk_down_thresh=args.jerk_down_thresh,
        fast_ratio=args.fast_ratio,
        quiet_frames=args.quiet_frames,
        cooldown_ms=args.cooldown,
        sound_wav=args.sound_wav,
    )

    osc_thread = threading.Thread(
        target=osc_listener, args=(detector, args.port, args.debug), daemon=True)
    osc_thread.start()
    time.sleep(0.5)

    if not detector.running:
        print("\nOSC failed, check ZigSim is sending and port is free.")
        return

    if args.plot:
        plot_thread = threading.Thread(
            target=run_plot,
            args=(detector, args.ratio_thresh, args.mag_high, args.mag_low, args.jerk_thresh),
            daemon=True)
        plot_thread.start()

    try:
        detector.run(verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        detector.stop()
        print(f"\nStats:")
        print(f"  Frames: {detector.total_frames}")
        print(f"  Triggers: {detector.total_triggers}")
        print(f"  Max mag: {detector.max_mag:.2f}g")
if __name__ == "__main__":
    main()
