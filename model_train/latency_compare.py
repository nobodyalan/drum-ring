"""
延迟对比工具 — Arduino 传感器 vs 手机陀螺仪
=============================================
同时读取两个数据源，以 PC 本地时钟为公共参考，
在同一张图上实时叠加显示，从而直观比较延迟。

数据源:
  1. Arduino Serial — 压电传感器敲击事件 (200Hz)
  2. ZigSim OSC UDP — iPhone 陀螺仪/加速度计 (~100Hz)

延迟分析方法:
  - 方法A (实时): Arduino 检测到敲击 → 标记时间点 →
                 在手机 IMU 波形图上画出垂直线，肉眼观察 IMU 响应滞后
  - 方法B (事件): 检测 IMU 信号的突变起始点 (onset detection) →
                 自动计算 Arduino 事件与 IMU onset 的时间差
  - 方法C (互相关): 离线分析 — 用滑动窗口互相关估计延迟

用法:
  python latency_compare.py --port COM6 --osc-port 8888 [--baud 115200]

依赖:
  pip install pyserial matplotlib numpy
"""

import sys
import time
import struct
import socket
import argparse
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ===================== pyserial 可选导入 =====================
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[WARN] pyserial 未安装，无法读取 Arduino。请运行: pip install pyserial")

import matplotlib
import matplotlib.pyplot as plt

# ===================== 中文字体 =====================
_CN_FONTS = ["Microsoft YaHei", "SimHei", "KaiTi", "SimSun"]
for _f in _CN_FONTS:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _f
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

# ===================== 常量 =====================
HISTORY_SECONDS = 8.0        # 显示最近 N 秒
OSC_BUFFER = 65536
SERIAL_TIMEOUT = 0.01        # Serial 读取超时 (秒)

# ===================== 数据结构 =====================

@dataclass
class ArduinoEvent:
    """Arduino 检测到的敲击/释放事件"""
    pc_time: float          # PC 接收到的时间 (time.time())
    sensor_index: int
    edge: str               # "FALLING" = 敲击, "RISING" = 释放
    count: int
    raw: int
    delta: int
    min_raw: int

@dataclass
class IMUFrame:
    """一帧 IMU 数据"""
    pc_time: float
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

@dataclass
class DelaySample:
    """一次延迟测量结果"""
    arduino_time: float
    imu_onset_time: float
    delay_ms: float

# ===================== 环形缓冲区 =====================

class RingBuffer:
    """带时间戳的环形缓冲区，按 HISTORY_SECONDS 自动清理"""
    def __init__(self):
        self.times: list[float] = []
        self.values: list[dict] = []

    def append(self, t: float, data: dict):
        self.times.append(t)
        self.values.append(data)
        self._trim(t)

    def _trim(self, now: float):
        cutoff = now - HISTORY_SECONDS - 1.0
        while self.times and self.times[0] < cutoff:
            self.times.pop(0)
            self.values.pop(0)

    def get_window(self, t_start: float, t_end: float):
        """获取 [t_start, t_end] 区间内的数据"""
        result = []
        for t, v in zip(self.times, self.values):
            if t_start <= t <= t_end:
                result.append((t, v))
        return result

    def __len__(self):
        return len(self.times)


# ===================== OSC 解析器 (与 zigsim_imu_osc.py 相同) =====================

class OSCParser:
    @staticmethod
    def parse(raw: bytes) -> list[dict]:
        results = []
        data = memoryview(raw)
        if len(data) < 4:
            return results
        if bytes(data[:8]) == b"#bundle\0":
            offset = 16
            while offset < len(data) - 4:
                elem_len = struct.unpack(">I", data[offset:offset+4])[0]
                offset += 4
                if elem_len > 0 and offset + elem_len <= len(data):
                    msg = OSCParser._parse_message(data[offset:offset+elem_len])
                    if msg:
                        results.append(msg)
                    offset += elem_len
        else:
            msg = OSCParser._parse_message(data)
            if msg:
                results.append(msg)
        return results

    @staticmethod
    def _parse_message(data: memoryview) -> Optional[dict]:
        try:
            addr_end = 0
            for i in range(len(data)):
                if data[i] == 0:
                    addr_end = i
                    break
            if addr_end == 0:
                return None
            address = bytes(data[:addr_end]).decode("ascii")
            addr_offset = (addr_end + 4) & ~3
            if addr_offset >= len(data) or data[addr_offset] != ord(b","):
                return None
            tag_start = addr_offset
            tag_end = 0
            for i in range(tag_start, len(data)):
                if data[i] == 0:
                    tag_end = i
                    break
            if tag_end == 0:
                return None
            types = bytes(data[tag_start+1:tag_end]).decode("ascii")
            data_offset = (tag_end + 4) & ~3
            values = []
            for t in types:
                if t == "f" and data_offset + 4 <= len(data):
                    v = struct.unpack(">f", data[data_offset:data_offset+4])[0]
                    values.append(float(v))
                    data_offset += 4
                elif t == "i" and data_offset + 4 <= len(data):
                    v = struct.unpack(">i", data[data_offset:data_offset+4])[0]
                    values.append(v)
                    data_offset += 4
                elif t == "d" and data_offset + 8 <= len(data):
                    v = struct.unpack(">d", data[data_offset:data_offset+8])[0]
                    values.append(float(v))
                    data_offset += 8
                elif t == "s":
                    s_end = 0
                    for i in range(data_offset, len(data)):
                        if data[i] == 0:
                            s_end = i
                            break
                    data_offset = (s_end + 4) & ~3
                else:
                    break
            return {"address": address, "value": values[0] if len(values) == 1 else values}
        except Exception:
            return None


def extract_imu_osc(messages: list[dict]) -> dict:
    result = {
        "accel": {"x": 0.0, "y": 0.0, "z": 0.0},
        "gyro":  {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    for msg in messages:
        addr = msg.get("address", "")
        val = msg.get("value", 0.0)
        if "/accel" in addr.lower():
            if isinstance(val, list) and len(val) >= 3:
                result["accel"]["x"] = float(val[0])
                result["accel"]["y"] = float(val[1])
                result["accel"]["z"] = float(val[2])
        elif "/gyro" in addr.lower():
            if isinstance(val, list) and len(val) >= 3:
                result["gyro"]["x"] = float(val[0])
                result["gyro"]["y"] = float(val[1])
                result["gyro"]["z"] = float(val[2])
    return result


# ===================== IMU Onset 检测 =====================

class IMUOnsetDetector:
    """
    检测 IMU 信号的突变起始点 (onset)。
    用短时能量/斜率阈值判断是否是动作开始。

    原理: 陀螺仪在静止时接近 0, 敲击瞬间会产生尖锐的角速度脉冲。
    """
    def __init__(self, window_size: int = 5, gyro_threshold: float = 2.0,
                 accel_threshold: float = 0.5):
        self.window_size = window_size
        self.gyro_threshold = gyro_threshold   # rad/s
        self.accel_threshold = accel_threshold   # g
        self.gyro_mag_history = deque(maxlen=window_size)
        self.accel_mag_history = deque(maxlen=window_size)
        self.last_onset_time = 0.0
        self.cooldown = 0.15  # 两次 onset 检测的最小间隔 (秒)

    def feed(self, imu: IMUFrame) -> Optional[float]:
        """
        输入一帧 IMU 数据，如果检测到运动 onset，返回 onset 时间。
        """
        gyro_mag = np.sqrt(imu.gyro_x**2 + imu.gyro_y**2 + imu.gyro_z**2)
        accel_mag = np.sqrt(imu.accel_x**2 + imu.accel_y**2 + imu.accel_z**2)

        self.gyro_mag_history.append(gyro_mag)
        self.accel_mag_history.append(accel_mag)

        if len(self.gyro_mag_history) < self.window_size:
            return None

        # 用滑动窗口的前半段 vs 后半段对比
        half = self.window_size // 2
        baseline_gyro = np.mean(list(self.gyro_mag_history)[:half])
        current_gyro = np.mean(list(self.gyro_mag_history)[half:])
        gyro_jump = current_gyro - baseline_gyro

        baseline_accel = np.mean(list(self.accel_mag_history)[:half])
        current_accel = np.mean(list(self.accel_mag_history)[half:])
        accel_jump = current_accel - baseline_accel

        # 间距 > cooldown 且 陀螺仪跳变 > 阈值 或 加速度跳变 > 阈值
        if (imu.pc_time - self.last_onset_time > self.cooldown and
            (gyro_jump > self.gyro_threshold or accel_jump > self.accel_threshold)):
            self.last_onset_time = imu.pc_time
            return imu.pc_time

        return None


# ===================== 全局状态 =====================
arduino_events: list[ArduinoEvent] = []     # Arduino 敲击事件
imu_buffer = RingBuffer()                    # IMU 时间序列
onset_times: list[float] = []                # IMU 检测到的 onset 时间
delay_samples: list[DelaySample] = []        # 延迟测量结果

# 线程安全锁
data_lock = threading.Lock()

# 统计
arduino_event_count = 0
imu_packet_count = 0
osc_start_time = time.time()

# 控制
stop_flag = threading.Event()


# ===================== Arduino Serial 读取 =====================

def parse_arduino_line(line: str) -> Optional[ArduinoEvent]:
    """
    解析 Arduino 输出:
      "Sensor 0 FALLING count=1 raw=45 delta=120 min=42"
      "Sensor 0 RISING count=1 raw=210 delta=55 min=42"
    """
    line = line.strip()
    if not line or line.startswith("Starting"):
        return None

    try:
        parts = line.split()
        if len(parts) < 6:
            return None

        # Sensor <idx> <EDGE> count=<n> raw=<r> delta=<d> min=<m>
        sensor_idx = int(parts[1])
        edge = parts[2]  # FALLING or RISING
        count = int(parts[3].split("=")[1])
        raw_val = int(parts[4].split("=")[1])
        delta = int(parts[5].split("=")[1])
        min_raw = int(parts[6].split("=")[1])

        return ArduinoEvent(
            pc_time=time.time(),
            sensor_index=sensor_idx,
            edge=edge,
            count=count,
            raw=raw_val,
            delta=delta,
            min_raw=min_raw,
        )
    except (IndexError, ValueError):
        return None


def parse_demo01_line(line: str) -> Optional[ArduinoEvent]:
    """
    解析 demo01.ino 的简单输出:
      " Counter: 1"
    """
    line = line.strip()
    if "Counter:" in line:
        try:
            count = int(line.split(":")[1].strip())
            return ArduinoEvent(
                pc_time=time.time(),
                sensor_index=0,
                edge="FALLING",
                count=count,
                raw=0,
                delta=0,
                min_raw=0,
            )
        except (IndexError, ValueError):
            pass
    return None


def arduino_reader(port: str, baud: int):
    """后台线程：持续读取 Arduino Serial"""
    global arduino_event_count

    if not HAS_SERIAL:
        print("[Arduino] pyserial 未安装，跳过 Arduino 读取")
        return

    try:
        ser = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
        print(f"[Arduino] 已连接 {port} @ {baud} baud")
    except serial.SerialException as e:
        print(f"[Arduino] ⚠ 无法打开 {port}: {e}")
        return

    time.sleep(1.5)  # 等待 Arduino reset 完成
    ser.reset_input_buffer()

    while not stop_flag.is_set():
        try:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            try:
                line = line_bytes.decode("utf-8", errors="replace")
            except Exception:
                continue

            # 尝试两种解析格式
            event = parse_arduino_line(line)
            if event is None:
                event = parse_demo01_line(line)
            if event is None:
                continue

            arduino_event_count += 1
            with data_lock:
                arduino_events.append(event)
                # 限制事件数量
                cutoff = time.time() - HISTORY_SECONDS - 1.0
                while arduino_events and arduino_events[0].pc_time < cutoff:
                    arduino_events.pop(0)

            # 控制台输出
            if event.edge == "FALLING":
                print(f"  🥁 Arduino 敲击 #{event.count} @ t={event.pc_time:.3f}")

        except Exception as e:
            if not stop_flag.is_set():
                print(f"[Arduino] 读取错误: {e}")
            break

    ser.close()
    print("[Arduino] 已断开")


# ===================== OSC UDP 读取 =====================

def osc_listener(port: int, gyro_threshold: float = 1.5,
                 accel_threshold: float = 0.3, onset_cooldown: float = 0.15,
                 debug: bool = False):
    """后台线程：持续接收 ZigSim OSC 数据"""
    global imu_packet_count

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 设置更大的接收缓冲
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    except Exception:
        pass
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.3)

    print(f"[OSC] 监听 UDP 0.0.0.0:{port} (gyro阈值={gyro_threshold}, accel阈值={accel_threshold})")

    onset_detector = IMUOnsetDetector(
        window_size=6,
        gyro_threshold=gyro_threshold,
        accel_threshold=accel_threshold,
    )
    onset_detector.cooldown = onset_cooldown

    first_pkt = True

    while not stop_flag.is_set():
        try:
            raw, addr = sock.recvfrom(OSC_BUFFER)
        except socket.timeout:
            continue
        except OSError:
            break
        except Exception as e:
            if not stop_flag.is_set():
                print(f"[OSC] 接收异常: {e}")
            continue

        recv_time = time.time()
        imu_packet_count += 1

        # 第一个包到达时打印诊断信息
        if first_pkt:
            first_pkt = False
            print(f"[OSC] ✅ 收到第一个数据包! 来自 {addr[0]}:{addr[1]}, 大小={len(raw)}字节")

        msgs = OSCParser.parse(raw)

        # Debug 模式：打印 OSC 地址
        if debug and imu_packet_count <= 10:
            for m in msgs:
                print(f"  [OSC DEBUG] {m.get('address', '?')} = {m.get('value', '?')}")
            if imu_packet_count == 10:
                print(f"  [OSC DEBUG] (之后不再打印)")

        imu_dict = extract_imu_osc(msgs)

        frame = IMUFrame(
            pc_time=recv_time,
            accel_x=imu_dict["accel"]["x"],
            accel_y=imu_dict["accel"]["y"],
            accel_z=imu_dict["accel"]["z"],
            gyro_x=imu_dict["gyro"]["x"],
            gyro_y=imu_dict["gyro"]["y"],
            gyro_z=imu_dict["gyro"]["z"],
        )

        with data_lock:
            imu_buffer.append(recv_time, {
                "accel_x": frame.accel_x,
                "accel_y": frame.accel_y,
                "accel_z": frame.accel_z,
                "gyro_x": frame.gyro_x,
                "gyro_y": frame.gyro_y,
                "gyro_z": frame.gyro_z,
            })

        # IMU onset 检测 + 延迟匹配
        onset_t = onset_detector.feed(frame)
        if onset_t is not None:
            with data_lock:
                onset_times.append(onset_t)
                # 找到此 onset 之前最近的 Arduino 事件
                if arduino_events:
                    recent_hits = [e for e in arduino_events
                                   if e.edge == "FALLING" and e.pc_time < onset_t]
                    if recent_hits:
                        closest_hit = recent_hits[-1]
                        delay_ms = (onset_t - closest_hit.pc_time) * 1000
                        # 合理的延迟范围: 0 ~ 500ms
                        if 0 < delay_ms < 500:
                            delay_samples.append(DelaySample(
                                arduino_time=closest_hit.pc_time,
                                imu_onset_time=onset_t,
                                delay_ms=delay_ms,
                            ))
                            print(f"  ⏱ 延迟测量: {delay_ms:.1f}ms "
                                  f"(Arduino→IMU onset, 敲击#{closest_hit.count})")
                            # 保留最近 50 个
                            if len(delay_samples) > 50:
                                delay_samples.pop(0)

    sock.close()
    print("[OSC] 已停止")


# ===================== 实时绘图 =====================

# 保存 Animation 引用，防止被 GC 回收
_active_anim = None

def run_combined_plot():
    """三行子图: 加速度计 | 陀螺仪 | 延迟统计"""
    global _active_anim
    import matplotlib.animation as animation

    # 不使用 plt.ion() — FuncAnimation 自带事件循环
    fig, (ax_acc, ax_gyro, ax_delay) = plt.subplots(3, 1, figsize=(14, 11))

    # 初始标题 (等待数据)
    fig.suptitle("延迟对比: Arduino 传感器 vs iPhone 陀螺仪\n"
                 "等待数据中... 请确保 ZigSim 正在发送 OSC 数据",
                 fontsize=12, fontweight="bold")

    # ---- 加速度计子图 ----
    ax_acc.set_title("加速度计 [g] (叠加 Arduino 敲击标记)")
    ax_acc.set_ylabel("g")
    ax_acc.set_ylim(-4, 4)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.axhline(y=0, color="gray", ls="--", alpha=0.5)
    (l_ax,) = ax_acc.plot([], [], "r-", label="Accel X", lw=0.8)
    (l_ay,) = ax_acc.plot([], [], "g-", label="Accel Y", lw=0.8)
    (l_az,) = ax_acc.plot([], [], "b-", label="Accel Z", lw=0.8)
    ax_acc.legend(loc="upper right", fontsize=7)

    # ---- 陀螺仪子图 ----
    ax_gyro.set_title("陀螺仪 [rad/s] (红色竖线 = Arduino 敲击, 蓝色虚线 = IMU onset)")
    ax_gyro.set_ylabel("rad/s")
    ax_gyro.set_ylim(-10, 10)
    ax_gyro.set_xlabel("时间 [s] (相对当前时刻)")
    ax_gyro.grid(True, alpha=0.3)
    ax_gyro.axhline(y=0, color="gray", ls="--", alpha=0.5)
    (l_gx,) = ax_gyro.plot([], [], "r-", label="Gyro X", lw=0.8)
    (l_gy,) = ax_gyro.plot([], [], "g-", label="Gyro Y", lw=0.8)
    (l_gz,) = ax_gyro.plot([], [], "b-", label="Gyro Z", lw=0.8)
    ax_gyro.legend(loc="upper right", fontsize=7)

    # ---- 延迟统计子图 ----
    ax_delay.set_title("延迟测量 [ms] (Arduino→IMU Onset)")
    ax_delay.set_ylabel("延迟 (ms)")
    ax_delay.set_ylim(0, 300)
    ax_delay.set_xlabel("测量序号")
    ax_delay.grid(True, alpha=0.3)
    ax_delay.axhline(y=50, color="green", ls="--", alpha=0.5, label="50ms (优秀)")
    ax_delay.axhline(y=100, color="orange", ls="--", alpha=0.5, label="100ms")
    ax_delay.axhline(y=200, color="red", ls="--", alpha=0.3, label="200ms")
    (l_delay_scatter,) = ax_delay.plot([], [], "ko", ms=4, alpha=0.7)
    (l_delay_avg,) = ax_delay.plot([], [], "r-", lw=2, label="移动平均")
    ax_delay.legend(loc="upper right", fontsize=7)

    # 存储动态创建的标记对象（统一清理避免内存泄漏）
    dyn_objects: list = []

    lines = [l_ax, l_ay, l_az, l_gx, l_gy, l_gz, l_delay_scatter, l_delay_avg]

    fig.tight_layout(pad=3.0)

    last_data_count = [0]  # mutable counter for tracking data arrival

    def animate(frame):
        nonlocal dyn_objects

        with data_lock:
            now = time.time()

            # ---- 清理上次的标记 (包括竖线和注释文本) ----
            for obj in dyn_objects:
                try:
                    obj.remove()
                except Exception:
                    pass
            dyn_objects.clear()

            # ---- 准备 IMU 数据 (转换为相对时间) ----
            imu_times_raw = imu_buffer.times.copy()
            imu_vals_raw = imu_buffer.values.copy()

            # 过滤掉太旧的数据
            cutoff = now - HISTORY_SECONDS
            filtered_times = []
            filtered_vals = []
            for t, v in zip(imu_times_raw, imu_vals_raw):
                if t >= cutoff:
                    filtered_times.append(t)
                    filtered_vals.append(v)

            # 转为相对当前时间的秒数
            if filtered_times:
                rel_times = [t - now for t in filtered_times]
            else:
                rel_times = []

            ax_data = [v.get("accel_x", 0) for v in filtered_vals]
            ay_data = [v.get("accel_y", 0) for v in filtered_vals]
            az_data = [v.get("accel_z", 0) for v in filtered_vals]
            gx_data = [v.get("gyro_x", 0) for v in filtered_vals]
            gy_data = [v.get("gyro_y", 0) for v in filtered_vals]
            gz_data = [v.get("gyro_z", 0) for v in filtered_vals]

            # ---- 更新 IMU 曲线 ----
            if rel_times:
                l_ax.set_data(rel_times, ax_data)
                l_ay.set_data(rel_times, ay_data)
                l_az.set_data(rel_times, az_data)
                l_gx.set_data(rel_times, gx_data)
                l_gy.set_data(rel_times, gy_data)
                l_gz.set_data(rel_times, gz_data)

                # 数据首次到达时打印提示
                if last_data_count[0] == 0:
                    last_data_count[0] = len(rel_times)
                    print(f"[图表] ✅ 开始渲染 IMU 数据 ({last_data_count[0]} 点)")
            else:
                for l in [l_ax, l_ay, l_az, l_gx, l_gy, l_gz]:
                    l.set_data([], [])

            # ---- 画 Arduino 敲击标记 (红色竖线) ----
            for evt in arduino_events:
                if evt.edge == "FALLING":
                    rel_t = evt.pc_time - now
                    if -HISTORY_SECONDS <= rel_t <= 0:
                        hl = ax_acc.axvline(x=rel_t, color="red", alpha=0.6,
                                            lw=1.5, ls="-")
                        dyn_objects.append(hl)
                        ann = ax_acc.annotate(f"HIT#{evt.count}",
                                              (rel_t, ax_acc.get_ylim()[1] * 0.9),
                                              fontsize=6, color="red", rotation=90,
                                              ha="right", va="top")
                        dyn_objects.append(ann)
                        hl2 = ax_gyro.axvline(x=rel_t, color="red", alpha=0.6,
                                              lw=1.5, ls="-")
                        dyn_objects.append(hl2)

            # ---- 画 IMU onset 标记 (蓝色虚线) ----
            for ot in onset_times:
                if ot > 0:
                    rel_t = ot - now
                    if -HISTORY_SECONDS <= rel_t <= 0:
                        hl = ax_gyro.axvline(x=rel_t, color="blue", alpha=0.5,
                                             lw=1.0, ls="--")
                        dyn_objects.append(hl)

            # ---- 更新 X 轴范围 ----
            for ax_obj in (ax_acc, ax_gyro):
                ax_obj.set_xlim(-HISTORY_SECONDS, 0.5)

            # ---- 动态调整 Y 轴 ----
            if ax_data:
                all_acc = ax_data + ay_data + az_data
                acc_abs_max = max(abs(v) for v in all_acc) * 1.3
                ax_acc.set_ylim(-max(acc_abs_max, 2), max(acc_abs_max, 2))
            if gx_data:
                all_gyro = gx_data + gy_data + gz_data
                gyro_abs_max = max(abs(v) for v in all_gyro) * 1.3
                ax_gyro.set_ylim(-max(gyro_abs_max, 5), max(gyro_abs_max, 5))

            # ---- 轴标签 ----
            ax_acc.set_xlabel("时间偏移 [s] | 红色竖线=Arduino敲击")
            ax_gyro.set_xlabel("时间偏移 [s] | 红色=Arduino敲击, 蓝色虚线=IMU onset")

            # ---- 延迟散点图 ----
            if delay_samples:
                indices = list(range(len(delay_samples)))
                delay_vals = [d.delay_ms for d in delay_samples]
                l_delay_scatter.set_data(indices, delay_vals)

                if len(delay_vals) >= 3:
                    window = min(5, len(delay_vals))
                    avg_vals = np.convolve(delay_vals, np.ones(window)/window, mode="valid")
                    avg_indices = list(range(window - 1, len(delay_vals)))
                    l_delay_avg.set_data(avg_indices, avg_vals)
                else:
                    l_delay_avg.set_data([], [])

                ax_delay.set_xlim(0, max(len(delay_samples), 20))
                if delay_vals:
                    dmax = max(delay_vals) * 1.2
                    ax_delay.set_ylim(0, max(dmax, 100))
            else:
                l_delay_scatter.set_data([], [])
                l_delay_avg.set_data([], [])

            # ---- 标题更新 ----
            avg_delay = (np.mean([d.delay_ms for d in delay_samples])
                         if delay_samples else 0)
            status = "运行中" if imu_packet_count > 0 else "等待 ZigSim OSC 数据..."
            fig.suptitle(
                f"延迟对比: Arduino 传感器 vs iPhone 陀螺仪 | {status}\n"
                f"IMU帧数: {imu_packet_count} | Arduino事件: {arduino_event_count} | "
                f"延迟测量: {len(delay_samples)}次 | 平均延迟: {avg_delay:.1f}ms",
                fontsize=12, fontweight="bold"
            )

        return lines + dyn_objects

    _active_anim = animation.FuncAnimation(
        fig, animate, interval=40,
        blit=False, cache_frame_data=False
    )

    def on_close(event):
        stop_flag.set()
    fig.canvas.mpl_connect("close_event", on_close)

    plt.tight_layout(pad=2.5)
    plt.show()


# ===================== 纯控制台模式 (无 GUI) =====================

def run_console_only():
    """不弹窗，仅在控制台输出延迟数据"""
    print("\n" + "=" * 60)
    print("  延迟对比 — 控制台模式")
    print("  敲击传感器/Arduino，观察延迟数据...")
    print("  按 Ctrl+C 停止")
    print("=" * 60 + "\n")

    try:
        while not stop_flag.is_set():
            time.sleep(0.5)
            with data_lock:
                if delay_samples:
                    recent = delay_samples[-5:]
                    delays = [f"{d.delay_ms:.0f}ms" for d in recent]
                    avg = np.mean([d.delay_ms for d in delay_samples])
                    print(f"\r  最近5次延迟: {delays} | 整体平均: {avg:.1f}ms  "
                          f"| Arduino事件: {arduino_event_count}  | IMU帧: {imu_packet_count}",
                          end="")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n⏹ 用户停止")

    # 最终统计
    with data_lock:
        if delay_samples:
            delays = [d.delay_ms for d in delay_samples]
            print(f"\n{'='*60}")
            print(f"  延迟统计 (共 {len(delays)} 次测量):")
            print(f"    均值: {np.mean(delays):.1f}ms")
            print(f"    中位数: {np.median(delays):.1f}ms")
            print(f"    标准差: {np.std(delays):.1f}ms")
            print(f"    最小: {np.min(delays):.1f}ms")
            print(f"    最大: {np.max(delays):.1f}ms")
            print(f"{'='*60}")
        else:
            print("\n⚠ 没有采集到延迟数据。请检查:")
            print("  1. Arduino 是否连接并输出敲击事件")
            print("  2. ZigSim 是否发送 IMU 数据")
            print("  3. 尝试降低 onset 检测阈值 (--gyro-thresh)")


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(
        description="延迟对比: Arduino 压电传感器 vs 手机陀螺仪",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python latency_compare.py --port COM3 --osc-port 8888
  python latency_compare.py --port COM3 --osc-port 8888 --no-plot
  python latency_compare.py --port COM5 --osc-port 8888 --gyro-thresh 2.5 --accel-thresh 0.5
  python latency_compare.py --osc-port 8888 --no-arduino    # 只测 IMU，不连 Arduino

调整 onset 检测灵敏度:
  --gyro-thresh 值越小越灵敏 (默认 1.5 rad/s)
  --accel-thresh 值越小越灵敏 (默认 0.3 g)
  --onset-cooldown onset 检测的最小间隔 (默认 0.15s)
        """
    )
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="Arduino 串口号 (如 COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", "-b", type=int, default=115200,
                        help="Arduino 波特率 (默认: 115200)")
    parser.add_argument("--osc-port", "-o", type=int, default=8888,
                        help="OSC UDP 监听端口 (默认: 8888)")
    parser.add_argument("--no-plot", action="store_true",
                        help="仅控制台模式，不弹窗")
    parser.add_argument("--no-arduino", action="store_true",
                        help="不连接 Arduino (仅显示 IMU)")
    parser.add_argument("--gyro-thresh", type=float, default=1.5,
                        help="陀螺仪 onset 检测阈值 rad/s (默认: 1.5)")
    parser.add_argument("--accel-thresh", type=float, default=0.3,
                        help="加速度计 onset 检测阈值 g (默认: 0.3)")
    parser.add_argument("--onset-cooldown", type=float, default=0.15,
                        help="onset 检测冷却时间 秒 (默认: 0.15)")
    parser.add_argument("--list-ports", action="store_true",
                        help="列出可用串口后退出")
    parser.add_argument("--save", type=str, default=None,
                        help="保存延迟数据到 CSV 文件")
    parser.add_argument("--debug", action="store_true",
                        help="打印前10个 OSC 包的原始地址和数值")
    args = parser.parse_args()

    # 列串口
    if args.list_ports:
        if HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            print("可用串口:")
            for p in ports:
                print(f"  {p.device} — {p.description}")
        else:
            print("pyserial 未安装，无法列出串口")
        return

    print()
    print("=" * 60)
    print("  延迟对比工具 — Arduino 传感器 vs 手机陀螺仪")
    print("=" * 60)
    print()
    print(f"  OSC UDP: 0.0.0.0:{args.osc_port}")
    print(f"  Onset陀螺仪阈值: {args.gyro_thresh} rad/s")
    print(f"  Onset加速度阈值: {args.accel_thresh} g")
    print(f"  Onset冷却时间: {args.onset_cooldown}s")
    print()
    print("  操作步骤:")
    print("    1. 确保 Arduino 已连接 USB")
    print("    2. iPhone 上 Zig Sim → Protocol: OSC → Port: 8888 → 开始发送")
    print("    3. 敲击传感器，观察图表")
    print("    4. 红色竖线 = Arduino 检测到敲击")
    print("    5. 蓝色虚线 = IMU 检测到运动 onset")
    print("    6. 两者时间差 = 传感器→手机的端到端延迟")
    print()

    # 自动检测 Arduino 串口
    if not args.no_arduino and args.port is None:
        if HAS_SERIAL:
            ports = list(serial.tools.list_ports.comports())
            if ports:
                args.port = ports[0].device
                print(f"  🔍 自动检测到 Arduino: {args.port}")
            else:
                print("  ⚠ 未检测到串口设备，将只接收 IMU 数据")
                args.no_arduino = True
        else:
            args.no_arduino = True

    # 启动 Arduino 读取线程
    if not args.no_arduino and args.port:
        arduino_thread = threading.Thread(
            target=arduino_reader, args=(args.port, args.baud), daemon=True
        )
        arduino_thread.start()
    else:
        print("  [跳过] Arduino 未连接")

    # 启动 OSC 接收线程
    osc_thread = threading.Thread(
        target=osc_listener,
        args=(args.osc_port, args.gyro_thresh, args.accel_thresh,
              args.onset_cooldown, args.debug),
        daemon=True
    )
    osc_thread.start()

    time.sleep(0.5)

    try:
        if args.no_plot:
            run_console_only()
        else:
            run_combined_plot()
    except KeyboardInterrupt:
        print("\n⏹ 用户停止")
    finally:
        stop_flag.set()

        # 保存数据
        if args.save and delay_samples:
            import csv
            with open(args.save, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "arduino_time", "imu_onset_time", "delay_ms"])
                for i, d in enumerate(delay_samples):
                    writer.writerow([i, d.arduino_time, d.imu_onset_time, d.delay_ms])
            print(f"\n📁 延迟数据已保存到: {args.save}")

        # 打印统计
        if delay_samples:
            delays = [d.delay_ms for d in delay_samples]
            print(f"\n{'='*60}")
            print(f"  延迟统计 (共 {len(delays)} 次测量):")
            print(f"    均值:   {np.mean(delays):.1f} ms")
            print(f"    中位数: {np.median(delays):.1f} ms")
            print(f"    标准差: {np.std(delays):.1f} ms")
            print(f"    最小:   {np.min(delays):.1f} ms")
            print(f"    最大:   {np.max(delays):.1f} ms")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
