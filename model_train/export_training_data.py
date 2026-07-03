"""
训练数据采集 — 全量录制
=======================
实时采集 Arduino 敲击 + iPhone IMU，保存完整 IMU 录制和敲击时间戳。
切片由 slice_dataset.py 负责。

输出格式:
  {prefix}_imu_{freq}.npy              — (T, 7) IMU 数组
        列: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, timestamp_us
  {prefix}_click_timestamps_{freq}.npy — (N,) 敲击时间戳 (Unix 秒)

用法:
  # 采集 weak 手势
  python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/weak_0

  # 采集 strong 手势
  python export_training_data.py --port COM6 --wifi --strength strong --save_dir train_data/strong_0

  # 然后切片:
  python slice_dataset.py --data_dir train_data --save sliced_data

依赖: pip install pyserial numpy
"""

import sys
import time
import struct
import socket
import threading
import json
import os
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ===================== 配置 =====================
TARGET_FS = 60            # IMU 采样率 (Hz), 用于旧格式命名
DEFAULT_BAUD = 115200

# ===================== OSC 解析 =====================

def parse_osc_bundle(raw: bytes) -> tuple:
    """解析 OSC Bundle, 返回 (gyro_x,gyro_y,gyro_z, accel_x,accel_y,accel_z)"""
    data = memoryview(raw)
    acc = gyr = None
    try:
        offset = 16 if bytes(data[:8]) == b"#bundle\0" else 0
        is_b = (offset == 16)
        while True:
            if is_b:
                if offset + 4 > len(data): break
                el = struct.unpack(">I", data[offset:offset+4])[0]
                offset += 4
                chunk = data[offset:offset+el]
                offset += el
            else:
                chunk = data
            ae = chunk.tobytes().find(b"\0")
            if ae <= 0: break
            astr = bytes(chunk[:ae]).decode("ascii", errors="replace")
            ao = (ae + 4) & ~3
            te = chunk[ao:].tobytes().find(b"\0")
            if te <= 0: break
            do = (ao + te + 4) & ~3
            if do + 12 <= len(chunk):
                vals = struct.unpack(">fff", chunk[do:do+12])
                if "/accel" in astr:
                    acc = (float(vals[0]), float(vals[1]), float(vals[2]))
                elif "/gyro" in astr:
                    gyr = (float(vals[0]), float(vals[1]), float(vals[2]))
            if not is_b: break
    except Exception:
        pass
    gx, gy, gz = gyr or (0, 0, 0)
    ax, ay, az = acc or (0, 0, 0)
    return gx, gy, gz, ax, ay, az


# ===================== 数据采集器 =====================

@dataclass
class HitEvent:
    pc_time: float
    sensor_idx: int
    count: int
    delta: int

class TrainingCollector:
    """实时录制 IMU 流 + Arduino 事件，保存完整记录供切片"""

    def __init__(self, gesture_id: int = 0,
                 strength: str = "strong", session: int = 0, sub: int = 0):
        self.gesture_id = gesture_id
        self.strength = strength
        self.session = session
        self.sub = sub
        # IMU 环形缓冲 (用于实时显示/调试)
        buffer_secs = 2.0
        self.imu_buffer: deque[tuple] = deque(maxlen=int(buffer_secs * TARGET_FS))
        # 全量 IMU 记录
        self.imu_continuous: list[tuple] = []  # (pc_time, gx, gy, gz, ax, ay, az)
        self.all_hits: list[HitEvent] = []      # 所有敲击事件
        self.lock = threading.Lock()
        self.start_time: Optional[float] = None  # 录制起始 PC 时间

    def add_imu(self, t: float, gx: float, gy: float, gz: float,
                ax: float, ay: float, az: float):
        if self.start_time is None:
            self.start_time = t
        with self.lock:
            self.imu_buffer.append((t, gx, gy, gz, ax, ay, az))
            self.imu_continuous.append((t, gx, gy, gz, ax, ay, az))

    def add_hit(self, event: HitEvent):
        with self.lock:
            self.all_hits.append(event)

    def save(self, save_dir: str, freq: int = TARGET_FS):
        """保存完整 IMU 录制 + 敲击时间戳 (供 slice_dataset.py 切片)"""
        os.makedirs(save_dir, exist_ok=True)

        # 加锁快照，防止 OSC 线程并发追加
        with self.lock:
            imu_snapshot = list(self.imu_continuous)
            hits_snapshot = list(self.all_hits)

        if not imu_snapshot:
            print("⚠ 没有 IMU 数据!")
            return

        # ── 构建 (T, 7) IMU 数组 ──
        T = len(imu_snapshot)
        imu_arr = np.zeros((T, 7), dtype=np.float64)
        t0 = imu_snapshot[0][0]  # 第一帧的 PC 时间
        for i, (t, gx, gy, gz, ax, ay, az) in enumerate(imu_snapshot):
            imu_arr[i, 0] = gx
            imu_arr[i, 1] = gy
            imu_arr[i, 2] = gz
            imu_arr[i, 3] = ax
            imu_arr[i, 4] = ay
            imu_arr[i, 5] = az
            imu_arr[i, 6] = (t - t0) * 1_000_000  # 微秒

        # ── 构建 Click Timestamps ──
        click_ts = np.array([h.pc_time for h in hits_snapshot], dtype=np.float64)

        # ── 命名: whr_gesture_{gid}_{strength}_{session}_{sub}_{type}_{freq}.npy ──
        prefix = f"whr_gesture_{self.gesture_id}_{self.strength}_{self.session}_{self.sub}"
        imu_path = os.path.join(save_dir, f"{prefix}_imu_{freq}.npy")
        click_path = os.path.join(save_dir, f"{prefix}_click_timestamps_{freq}.npy")

        np.save(imu_path, imu_arr)
        np.save(click_path, click_ts)
        # 保存 t0 (第一帧 IMU 的 PC 绝对时间), 供下游精确时间对齐
        t0_path = os.path.join(save_dir, f"{prefix}_t0_{freq}.npy")
        np.save(t0_path, np.array([t0]))

        duration_s = (imu_arr[-1, 6] - imu_arr[0, 6]) / 1_000_000
        print(f"\n{'='*60}")
        print(f"  ✅ 录制数据已保存: {save_dir}/")
        print(f"     {prefix}_imu_{freq}.npy            — {imu_arr.shape} float64")
        print(f"       列: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, timestamp_us")
        print(f"       时长: {duration_s:.1f}s, 帧数: {T}")
        print(f"     {prefix}_click_timestamps_{freq}.npy — {click_ts.shape} float64")
        print(f"       敲击数: {len(click_ts)} (Unix 秒)")
        print(f"{'='*60}")
        print(f"\n  下一步: python slice_dataset.py --data_dir {save_dir}/.. --save sliced_data")


# ===================== 实时采集主循环 =====================

def collect(port: str, save_dir: str, use_wifi: bool = True,
            duration: float = 0, baud: int = DEFAULT_BAUD,
            gesture_id: int = 0, strength: str = "strong",
            session: int = 0, sub: int = 0):
    """实时采集模式 — 仅录制全量数据，不切片"""

    collector = TrainingCollector(
        gesture_id=gesture_id, strength=strength,
        session=session, sub=sub,
    )
    running = [True]

    # ── Arduino ──
    ser = None
    if port:
        try:
            import serial
            ser = serial.Serial(port, baud, timeout=0.001)
            ser.set_buffer_size(rx_size=65536)
            time.sleep(1.5)
            ser.reset_input_buffer()
            ser.readline()
            print(f"[Arduino] ✅ {port} @ {baud}")
        except Exception as e:
            print(f"[Arduino] ⚠ {e}")

    # ── OSC ──
    osc_error = [None]
    osc_started = [False]
    def osc_loop():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", 8888))
            sock.settimeout(0.1)
            osc_started[0] = True
            print("[OSC] Listening UDP :8888")
        except Exception as e:
            osc_error[0] = str(e)
            print(f"[OSC] ❌ 绑定失败: {e}")
            return
        while running[0]:
            try:
                raw, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except Exception:
                break
            t = time.time()
            gx, gy, gz, ax, ay, az = parse_osc_bundle(raw)
            collector.add_imu(t, gx, gy, gz, ax, ay, az)
        sock.close()

    osc_thread = threading.Thread(target=osc_loop, daemon=True)
    osc_thread.start()
    time.sleep(0.5)
    if osc_error[0]:
        print(f"\n❌ OSC 启动失败: {osc_error[0]}")
        print("   请检查是否有其他程序占用 8888 端口")
    if not osc_started[0]:
        print(f"\n❌ OSC 线程未能启动\n")

    # ── 主循环: 读 Arduino 事件 ──
    print(f"\n{'='*60}")
    print(f"  全量数据录制 — Strength: {strength}")
    print(f"  敲击传感器开始录制, 按 Enter 停止...")
    print(f"  切片请用: python slice_dataset.py --data_dir ...")
    print(f"{'='*60}\n")

    hit_count = [0]
    buf = bytearray()

    try:
        t0 = time.time()
        last_status = time.time()
        while running[0]:
            if ser:
                try:
                    raw = ser.read(ser.in_waiting or 1)
                    if raw:
                        buf.extend(raw)
                        while b"\n" in buf:
                            line_bytes, buf = buf.split(b"\n", 1)
                            line = line_bytes.decode("utf-8", errors="replace").strip()
                            try:
                                parts = line.split()
                                if len(parts) >= 7 and parts[0] == "Sensor" and parts[2] == "FALLING":
                                    count = int(parts[3].split("=")[1])
                                    delta = int(parts[5].split("=")[1])
                                    event = HitEvent(
                                        pc_time=time.time(),
                                        sensor_idx=int(parts[1]),
                                        count=count,
                                        delta=delta,
                                    )
                                    collector.add_hit(event)
                                    hit_count[0] += 1
                                    print(f"  🥁 Hit #{hit_count[0]} | delta={delta:4d}")
                            except Exception:
                                pass
                except Exception:
                    pass

            # 定期状态提示
            if time.time() - last_status > 5.0:
                imu_count = len(collector.imu_continuous)
                if imu_count == 0 and time.time() - t0 > 5:
                    print(f"  ⚠ 5秒内未收到 IMU 数据! 检查 ZigSim 是否在发送")
                else:
                    print(f"  [状态] IMU帧={imu_count} | 敲击={hit_count[0]}")
                last_status = time.time()

            if duration > 0 and time.time() - t0 > duration:
                break

            if duration <= 0:
                import msvcrt
                if msvcrt.kbhit():
                    msvcrt.getch()
                    break
    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        if ser:
            ser.close()

    print(f"\n  录制结束 | 敲击={hit_count[0]} | IMU帧={len(collector.imu_continuous)}\n")
    collector.save(save_dir)


# ===================== 查看数据 =====================

def inspect(data_dir: str):
    """查看已录制的训练数据 (legacy 格式)"""
    import matplotlib.pyplot as plt

    files = os.listdir(data_dir)
    imu_file = [f for f in files if "_imu_" in f and f.endswith(".npy")]
    click_file = [f for f in files if "click_timestamps" in f and f.endswith(".npy")]

    if not imu_file or not click_file:
        print(f"⚠ {data_dir} 下未找到录制文件")
        return

    imu = np.load(os.path.join(data_dir, imu_file[0]))
    clicks = np.load(os.path.join(data_dir, click_file[0]))

    print(f"IMU shape: {imu.shape}")
    print(f"Clicks: {len(clicks)}")
    print(f"时长: {(imu[-1, 6] - imu[0, 6]) / 1_000_000:.1f}s")

    # 画图: 6 通道 + 敲击标记
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    titles = ["Gyro X", "Gyro Y", "Gyro Z", "Accel X", "Accel Y", "Accel Z"]
    t_axis = imu[:, 6] / 1_000_000  # 微秒 → 秒

    for ch in range(6):
        ax = axes[ch // 2, ch % 2]
        ax.plot(t_axis, imu[:, ch], "b-", lw=0.5)
        # 标记敲击
        t0 = clicks[0] if len(clicks) > 0 else 0
        for c in clicks[:min(len(clicks), 50)]:
            # 转为相对秒
            rel_c = c - t0 + t_axis[0]
            if t_axis[0] <= rel_c <= t_axis[-1]:
                ax.axvline(x=rel_c, color="red", ls="--", lw=1, alpha=0.4)
        ax.set_title(titles[ch])
        ax.set_xlabel("秒")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training Data — {os.path.basename(data_dir)} ({len(clicks)} hits)", fontsize=14)
    plt.tight_layout()
    plt.show()


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(
        description="训练数据采集 — 全量录制 (切片交由 slice_dataset.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 采集 weak 手势
  python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/weak_0

  # 采集 strong 手势
  python export_training_data.py --port COM6 --wifi --strength strong --save_dir train_data/strong_0

  # 采集指定时长 (30秒自动停止)
  python export_training_data.py --port COM6 --wifi --strength weak --save_dir train_data/weak_1 --duration 30

  # 查看录制数据
  python export_training_data.py --inspect train_data/weak_0
        """
    )
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="Arduino 串口号")
    parser.add_argument("--baud", "-b", type=int, default=DEFAULT_BAUD,
                        help=f"波特率 (默认 {DEFAULT_BAUD})")
    parser.add_argument("--wifi", action="store_true",
                        help="WiFi 热点 OSC 模式")
    parser.add_argument("--strength", type=str, default="strong",
                        help="力度标签 weak/strong, 用于目录命名 (默认 strong)")
    parser.add_argument("--save_dir", "-s", type=str, default="train_data",
                        help="输出目录")
    parser.add_argument("--duration", "-d", type=float, default=0,
                        help="采集时长 (秒, 0=手动 Enter 停止)")
    parser.add_argument("--gesture-id", type=int, default=0,
                        help="手势编号 (用于文件名)")
    parser.add_argument("--session", type=int, default=0,
                        help="采集会话编号")
    parser.add_argument("--sub", type=int, default=0,
                        help="子会话编号")
    parser.add_argument("--inspect", type=str, default=None,
                        help="查看已录制数据目录")
    parser.add_argument("--list-ports", action="store_true")

    args = parser.parse_args()

    if args.list_ports:
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                print(f"  {p.device} — {p.description}")
        except Exception:
            print("pyserial 未安装")
        return

    if args.inspect:
        inspect(args.inspect)
        return

    # 自动检测串口
    if not args.port:
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            if ports:
                args.port = ports[0].device
                print(f"🔍 自动检测 Arduino: {args.port}")
        except Exception:
            pass

    if not args.port:
        print("⚠ 未检测到 Arduino，请手动指定 --port")
        return

    collect(
        port=args.port,
        save_dir=args.save_dir,
        use_wifi=args.wifi,
        duration=args.duration,
        baud=args.baud,
        gesture_id=args.gesture_id,
        strength=args.strength,
        session=args.session,
        sub=args.sub,
    )


if __name__ == "__main__":
    main()
