"""
实时踩踏检测 — 从 ZigSim OSC 流实时预测
========================================
接收 iPhone ZigSim 发送的 IMU 数据 (UDP :8888),
用训练好的 InceptionTime 模型做滑动窗口实时预测。

检测策略: 纯概率双阈值滞回
  - 每帧推理 → P(hit) ∈ [0,1] (CE Loss 校准, 概率语义跨 session 一致)
  - P(hit) > --prob-high → 触发提示音 + 进入冷却
  - P(hit) < --prob-low  → 释放滞回 (允许再次触发)
  - 冷却期内不触发, 防止连击

用法:
  python realtime_detect.py
  python realtime_detect.py --calibrate
  python realtime_detect.py --prob-high 0.85 --prob-low 0.4 -v
  python realtime_detect.py --cooldown 500
"""

import sys
import time
import struct
import socket
import threading
import argparse
from collections import deque

import numpy as np
import torch
from tsai.all import InceptionTime

# ==================== 默认参数 ====================
OSC_PORT = 8888
PROB_HIGH = 0.85              # 概率高阈值 (超过触发)
PROB_LOW = 0.4                # 概率低阈值 (低于释放滞回)
COOLDOWN_MS = 400             # 触发后冷却时间 (ms)
SOUND_FREQ = 800              # 提示音频率 (Hz)
SOUND_DURATION_MS = 100       # 提示音持续 (ms)
MODEL_PATH = "models/hit_detector_inception.pth"
CALIBRATE_SEC = 30            # 校准模式采集秒数


# ==================== OSC 解析 ====================
# 与 export_training_data.py 完全一致的 ZigSim Bundle 解析

def parse_osc_bundle(raw: bytes) -> tuple:
    """
    解析 OSC Bundle (ZigSim 格式).
    返回 (gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z)
    解析失败时对应通道为 0.
    """
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
            # 地址字符串
            ae = chunk.tobytes().find(b"\0")
            if ae <= 0:
                break
            astr = bytes(chunk[:ae]).decode("ascii", errors="replace")
            ao = (ae + 4) & ~3
            # 类型标签
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


# ==================== 模型加载 ====================

def load_model(model_path: str):
    """
    加载训练好的模型及预处理参数。自动适配 InceptionTime / FCN / LiteCNN 架构。
    返回 (model, mean, std, window_frames)
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # ── 校验 checkpoint 格式 ──
    if "model_state" not in ckpt:
        raise ValueError(
            f"模型文件 {model_path} 格式不兼容 (缺少 model_state)。"
            f"请用 train_model.py 重新训练。"
        )
    required = ["channels", "classes", "mean", "std", "window_frames", "target_fs"]
    missing = [k for k in required if k not in ckpt or ckpt[k] is None]
    if missing:
        raise ValueError(
            f"模型文件 {model_path} 缺少字段: {missing}。"
            f"可能是旧版格式, 请用 train_model.py 重新训练。"
        )

    model_name = ckpt.get("model_name", "inception")
    c_in = ckpt["channels"]
    c_out = ckpt["classes"]

    # ── 按架构重建模型 ──
    if model_name == "fcn":
        from tsai.all import FCN
        model = FCN(c_in, c_out)
    elif model_name == "lite":
        import torch.nn as nn

        class LiteCNN(nn.Module):
            def __init__(self, c_in, c_out):
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv1d(c_in, c_in, kernel_size=3, groups=c_in, padding=1),
                    nn.Conv1d(c_in, 16, kernel_size=1),
                    nn.BatchNorm1d(16), nn.ReLU(),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv1d(16, 16, kernel_size=3, groups=16, padding=1),
                    nn.Conv1d(16, 32, kernel_size=1),
                    nn.BatchNorm1d(32), nn.ReLU(),
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, c_out),
                )

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.pool(x)
                return self.head(x)

        model = LiteCNN(c_in, c_out)
    elif model_name == "tst":
        from tsai.all import TST
        model = TST(c_in, c_out, d_model=64, n_heads=4, fc_dropout=0.3)
    elif model_name == "inception" or model_name is None:
        # ── 定制 InceptionTime: ks=[3,5,7], depth=4, 加权池化 ──
        # 与 train_model.py 的 build_inception_time 架构完全一致
        import torch.nn as nn

        ks = [3, 5, 7]
        nf = 32
        depth = 4
        out_ch = nf * (len(ks) + 1)  # 128

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
                # 可学习时序权重 (与 train_model.py 一致)
                max_t = 15  # window_frames
                init_w = torch.exp(torch.linspace(0, 2.0, max_t))  # 1.0 → 7.4
                self.time_weights = nn.Parameter(init_w)
                self.head = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(out_ch, c_out),
                )

            def forward(self, x):
                feat = self.backbone(x)
                T = feat.shape[-1]
                w = self.time_weights[:T]
                w = torch.softmax(w, dim=0)
                pooled = (feat * w.view(1, 1, -1)).sum(dim=-1)
                return self.head(pooled)

        model = WeightedInceptionTime()

        # ── 兼容旧模型 (tsai InceptionTime, ks=[39,19,9], GAP) ──
        # 用 state_dict key 判断: 旧模型有 'inceptionblock.inception.0.bottleneck.weight'
        sample_key = list(ckpt["model_state"].keys())[0]
        if "inceptionblock" in sample_key:
            print(f"  检测到旧版 tsai InceptionTime 模型, 改回对应架构")
            model = InceptionTime(c_in=c_in, c_out=c_out, nf=32, depth=6)
    else:
        raise ValueError(f"未知模型类型: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 预处理参数: 训练时的每通道均值和标准差
    # (checkpoint 中可能是 torch.Tensor 或 np.ndarray, 统一为 float32 numpy)
    def _to_np(arr):
        if isinstance(arr, np.ndarray):
            return arr.squeeze().astype(np.float32)
        return arr.squeeze().cpu().numpy().astype(np.float32)

    mean = _to_np(ckpt["mean"])   # (6,)
    std  = _to_np(ckpt["std"])    # (6,)
    window_frames = ckpt["window_frames"]
    target_fs = ckpt["target_fs"]

    print(f"模型已加载: {model_path}")
    print(f"  架构: {model_name.upper() if model_name != 'inception' else 'InceptionTime'}"
          f"  |  窗口: {window_frames} 帧 "
          f"({window_frames / target_fs * 1000:.0f}ms @{target_fs}Hz)")
    print(f"  输入: {c_in} 通道  |  输出: {c_out} 类")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    return model, mean, std, window_frames


# ==================== 提示音 ====================

def play_beep(freq: int = 800, duration_ms: int = 100):
    """播放短促提示音。Windows 用 winsound, 其他平台 print BEL 字符。"""
    try:
        import winsound
        winsound.Beep(freq, duration_ms)
    except ImportError:
        # 非 Windows: 终端响铃
        print("\a", end="", flush=True)


# ==================== 实时检测器 ====================

class RealtimeDetector:
    """
    维护滑动窗口缓冲，每收到新 IMU 帧就跑一次模型推理。

    检测策略: 纯概率双阈值滞回
      - 每帧推理得到 P(hit) ∈ [0, 1], CE Loss 天生校准概率
      - P(hit) > --prob-high → 触发提示音 + 进入冷却
      - P(hit) < --prob-low  → 释放滞回 (允许再次触发)
      - 概率语义跨 session 一致, 冷却防连击
    """

    def __init__(self, model, mean: np.ndarray, std: np.ndarray,
                 window_frames: int,
                 prob_high: float = 0.85,
                 prob_low: float = 0.4,
                 cooldown_ms: int = 400,
                 sound_freq: int = 800,
                 sound_dur: int = 100):
        self.model = model
        self.mean = mean
        self.std = std
        self.window_frames = window_frames

        self.prob_high = prob_high
        self.prob_low = prob_low
        self.cooldown_s = cooldown_ms / 1000.0
        self.sound_freq = sound_freq
        self.sound_dur = sound_dur

        # 环形缓冲: 最新的 window_frames 帧 IMU
        self.buffer: deque = deque(maxlen=window_frames)
        self.lock = threading.Lock()
        self.frame_count = 0

        # 检测状态: 双阈值滞回状态机
        self.armed = False              # 滞回: True=已触发, 等释放
        self.last_trigger_time = 0.0

        # 实时绘图历史 (最近 300 帧 ≈ 5 秒)
        self.plot_prob: deque = deque(maxlen=300)

        # 统计
        self.total_frames = 0
        self.total_triggers = 0
        self.max_prob = float("-inf")

        self.running = True

    # ---- OSC 线程调用 ----
    def add_imu_frame(self, gx, gy, gz, ax, ay, az):
        """线程安全地追加一帧 IMU 数据"""
        with self.lock:
            self.buffer.append([gx, gy, gz, ax, ay, az])
            self.frame_count += 1

    # ---- 推理 ----
    def predict(self) -> float:
        """
        取最近 window_frames 帧 → 标准化 → 模型推理.
        返回 softmax 概率 P(hit) ∈ [0, 1].
        CE Loss 训练出的概率语义跨 session 一致.
        """
        with self.lock:
            window = np.array(list(self.buffer), dtype=np.float32).T  # (6, T)

        # Z-score 标准化
        mean = self.mean.reshape(6, 1)
        std  = self.std.reshape(6, 1)
        window_norm = (window - mean) / (std + 1e-8)

        x = torch.from_numpy(window_norm).unsqueeze(0)  # (1, 6, T)

        with torch.no_grad():
            logits = self.model(x)                       # (1, 2)
            prob = torch.softmax(logits, dim=1)[0, 1].item()  # P(hit)

        return prob

    # ---- 主循环 ----
    def run(self, verbose: bool = False):
        print(f"\n  实时检测开始")
        print(f"  策略: 纯概率双阈值滞回")
        print(f"  高阈值={self.prob_high:.2f} (触发)  低阈值={self.prob_low:.2f} (释放)")
        print(f"  冷却: {self.cooldown_s * 1000:.0f}ms")
        print(f"  等待 IMU 数据...\n")

        last_frame_count = 0
        last_status = time.time()

        while self.running:
            time.sleep(0.001)

            with self.lock:
                cur_count = self.frame_count
                buf_len = len(self.buffer)

            # 等待窗口填满
            if buf_len < self.window_frames:
                if time.time() - last_status > 3.0:
                    if self.frame_count == 0:
                        print(f"  缓冲中... {buf_len}/{self.window_frames} 帧 "
                              f"(未收到 IMU 数据!)")
                    else:
                        print(f"  缓冲中... {buf_len}/{self.window_frames} 帧")
                    last_status = time.time()
                continue

            # 没有新帧 → 不重复推理
            if cur_count == last_frame_count:
                continue
            last_frame_count = cur_count

            # ---- 推理 ----
            prob = self.predict()
            self.total_frames += 1
            if prob > self.max_prob:
                self.max_prob = prob

            self.plot_prob.append(prob)

            now = time.time()
            in_cooldown = (now - self.last_trigger_time) < self.cooldown_s

            # ---- 双阈值滞回状态机 ----
            if not self.armed:
                # IDLE: P(hit) > prob_high + 不在冷却 → 触发
                if prob > self.prob_high and not in_cooldown:
                    self.armed = True
                    self.total_triggers += 1
                    print(f"\n  DETECT! (#{self.total_triggers}) | "
                          f"P(hit)={prob:.3f} > {self.prob_high}")

                    threading.Thread(
                        target=play_beep,
                        args=(self.sound_freq, self.sound_dur),
                        daemon=True,
                    ).start()

                    self.last_trigger_time = now
            else:
                # ARMED: P(hit) 降到低阈值以下 → 释放, 准备下一次检测
                if prob < self.prob_low:
                    self.armed = False

            # ---- 输出 ----
            if verbose:
                self._print_verbose(prob)
            elif self.total_frames % 120 == 0:
                cooldown_left = max(0.0, self.cooldown_s - (now - self.last_trigger_time))
                cd_str = f"cool {cooldown_left:.1f}s" if cooldown_left > 0.01 else "ready"
                state = "ARMED" if self.armed else "idle"
                print(f"  frame={self.total_frames:6d} | "
                      f"P(hit)={prob:.3f} | "
                      f"{state:5s} | "
                      f"{cd_str} | "
                      f"trig={self.total_triggers}")

    def _print_verbose(self, prob: float):
        """单行动态刷新: 实时展示概率和判断结果, 不滚屏."""
        # 概率 [0, 1] → 条长度 0~30
        bar_len = int(prob * 30)
        bar = ("=" * max(0, bar_len - 1) + ">") if bar_len > 0 else "|"
        bar = bar.ljust(30)

        # 标记
        if self.armed:
            marker = "** HIT **"
        elif prob > self.prob_high:
            marker = ">>TRIG<<"
        else:
            marker = "        "

        cdl = max(0.0, self.cooldown_s - (time.time() - self.last_trigger_time))
        cd = f"CD{cdl:.1f}" if cdl > 0.01 else "   "
        state = "A" if self.armed else "."

        print(f"\r  [{bar}] P={prob:.3f} | {state} {cd} | trig={self.total_triggers} | {marker}",
              end="", flush=True)

    def stop(self):
        self.running = False


# ==================== OSC 监听线程 ====================

def osc_listener(detector: RealtimeDetector, port: int, debug: bool = False):
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
            if now - last_debug > 1.0:  # 每秒打印一次, 避免刷屏
                print(f"  [DEBUG] 收到 OSC 包 #{pkt_count} 来自 {addr}, "
                      f"大小={len(raw)}B, "
                      f"gyro=({gx:+.3f},{gy:+.3f},{gz:+.3f}), "
                      f"accel=({ax:+.3f},{ay:+.3f},{az:+.3f})")
                last_debug = now
        elif pkt_count == 1:
            print(f"[OSC] 收到第 1 帧! 来自 {addr}, "
                  f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f})")

    sock.close()
    if pkt_count > 0:
        print(f"[OSC] 已关闭 (共收到 {pkt_count} 包)")
    else:
        print("[OSC] 已关闭 (未收到任何数据!)")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="实时踩踏检测 — IMU 流 + InceptionTime 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认参数 (纯概率双阈值滞回)
  python realtime_detect.py

  # 校准模式: 采集背景数据, 显示概率分布
  python realtime_detect.py --calibrate

  # 根据校准结果调参
  python realtime_detect.py --prob-high 0.85 --prob-low 0.4 -v

  # 降低冷却
  python realtime_detect.py --cooldown 300
        """,
    )
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"模型 .pth 路径 (默认: {MODEL_PATH})")
    parser.add_argument("--port", type=int, default=OSC_PORT,
                        help=f"OSC UDP 端口 (默认: {OSC_PORT})")
    parser.add_argument("--prob-high", type=float, default=PROB_HIGH,
                        help=f"概率高阈值, 超过触发 (默认: {PROB_HIGH})")
    parser.add_argument("--prob-low", type=float, default=PROB_LOW,
                        help=f"概率低阈值, 低于释放滞回 (默认: {PROB_LOW})")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_MS,
                        help=f"触发后冷却 ms (默认: {COOLDOWN_MS})")
    parser.add_argument("--freq", type=int, default=SOUND_FREQ,
                        help=f"提示音频率 Hz (默认: {SOUND_FREQ})")
    parser.add_argument("--sound-ms", type=int, default=SOUND_DURATION_MS,
                        help=f"提示音持续 ms (默认: {SOUND_DURATION_MS})")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式: 打印收到的 OSC 原始数据")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="实时模式: 每帧动态显示概率和判断结果")
    parser.add_argument("--plot", action="store_true",
                        help="实时绘图: 显示 P(hit) 动态曲线及双阈值")
    parser.add_argument("--calibrate", action="store_true",
                        help="校准模式: 采集真实 IMU 数据, 显示概率分布, "
                             "不触发提示音, 用于选择合适的 --prob-high/--prob-low 阈值")

    args = parser.parse_args()

    print("=" * 55)
    print("  实时踩踏检测")
    print("=" * 55)

    # 1. 加载模型
    model, mean, std, window_frames = load_model(args.model)

    # 2. 创建检测器
    detector = RealtimeDetector(
        model, mean, std, window_frames,
        prob_high=args.prob_high,
        prob_low=args.prob_low,
        cooldown_ms=args.cooldown,
        sound_freq=args.freq,
        sound_dur=args.sound_ms,
    )

    # 3. 启动 OSC 监听
    osc_thread = threading.Thread(
        target=osc_listener, args=(detector, args.port, args.debug), daemon=True,
    )
    osc_thread.start()
    time.sleep(0.5)

    if not detector.running:
        print("\nOSC failed, check:")
        print("   - ZigSim is sending?")
        print("   - UDP port 8888 free?")
        return

    # 4. 校准模式 or 正常检测
    if args.calibrate:
        run_calibrate(detector, CALIBRATE_SEC, args.prob_high, args.prob_low)
    else:
        # 实时绘图线程
        plot_thread = None
        if args.plot:
            plot_thread = threading.Thread(
                target=run_plot, args=(detector, args.prob_high, args.prob_low), daemon=True)
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
            print(f"  Max P(hit): {detector.max_prob:.3f}")


# ==================== 校准模式 ====================

def run_calibrate(detector: RealtimeDetector, duration_sec: float,
                  prob_high: float, prob_low: float):
    """
    采集真实 IMU 数据, 不触发, 只统计概率分布。
    用于帮用户选择合适的 --prob-high/--prob-low 阈值。
    """
    print(f"\n  CALIBRATE MODE ({duration_sec}s)")
    print(f"  Collecting live IMU data, no triggers...")
    print(f"  Move naturally (walk, stand, shift weight) but DON'T stomp yet.\n")

    probs = []
    last_frame_count = 0
    t0 = time.time()

    while detector.running and (time.time() - t0) < duration_sec:
        time.sleep(0.001)

        with detector.lock:
            cur_count = detector.frame_count
            buf_len = len(detector.buffer)

        if buf_len < detector.window_frames:
            if time.time() - t0 > 3:
                print(f"  Waiting for IMU... {buf_len}/{detector.window_frames}")
            continue

        if cur_count == last_frame_count:
            continue
        last_frame_count = cur_count

        prob = detector.predict()
        probs.append(prob)

    detector.stop()

    if len(probs) < 50:
        print(f"\n  Not enough data ({len(probs)} frames). Check OSC stream.")
        return

    probs_arr = np.array(probs)

    print(f"\n  Results ({len(probs)} frames, {len(probs)/60:.1f}s @60Hz):")
    print(f"  {'='*50}")
    print(f"  P(hit) distribution (softmax probability, CE-calibrated):")
    print(f"    Mean:  {np.mean(probs_arr):.3f}")
    print(f"    Std:   {np.std(probs_arr):.3f}")
    print(f"    P90:   {np.percentile(probs_arr, 90):.3f}")
    print(f"    P95:   {np.percentile(probs_arr, 95):.3f}")
    print(f"    P99:   {np.percentile(probs_arr, 99):.3f}")
    print(f"    Max:   {np.max(probs_arr):.3f}")

    # 建议阈值: high 取 P99 × 1.5 (留余量), low 取 P95 (安全释放线)
    p99 = np.percentile(probs_arr, 99)
    p95 = np.percentile(probs_arr, 95)
    recommended_high = min(max(p99 * 1.5, 0.7), 0.95)
    recommended_low = min(max(p95 * 0.8, 0.2), 0.6)

    print()
    print(f"  Suggested thresholds (background-based):")
    print(f"    --prob-high {recommended_high:.2f}  (P99 × 1.5, clamped [0.7, 0.95])")
    print(f"    --prob-low  {recommended_low:.2f}   (P95 × 0.8, clamped [0.2, 0.6])")
    print(f"  Current: --prob-high {prob_high} --prob-low {prob_low}")

    if prob_high <= p99:
        print(f"\n  !! Current --prob-high ({prob_high}) is BELOW or AT P99 background ({p99:.3f})!")
        print(f"     Background noise may cause false triggers.")

    # 统计超越阈值频率
    above_high = np.mean(probs_arr > prob_high) * 100
    above_low = np.mean(probs_arr > prob_low) * 100
    print(f"\n  With current thresholds:")
    print(f"    Frames above --prob-high ({prob_high}): {above_high:.1f}%")
    print(f"    Frames above --prob-low  ({prob_low}):  {above_low:.1f}%")
    print(f"\n  Suggested command:")
    print(f"    python realtime_detect.py --prob-high {recommended_high:.2f} --prob-low {recommended_low:.2f}")

    # 画图
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.hist(probs_arr, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=0.5, color="gray", ls="--", lw=1, label="0.5 = decision boundary")
        ax.axvline(x=recommended_high, color="green", ls="-", lw=2,
                    label=f"recommended --prob-high={recommended_high:.2f}")
        ax.axvline(x=recommended_low, color="orange", ls="-", lw=1.5,
                    label=f"recommended --prob-low={recommended_low:.2f}")
        ax.set_xlabel("P(hit)")
        ax.set_ylabel("Count")
        ax.set_title("Probability Distribution (Background)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("Calibration — P(hit) Distribution", fontsize=14)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\n  (Plot failed: {e})")


# ==================== 实时绘图 ====================

def run_plot(detector: RealtimeDetector, prob_high: float, prob_low: float):
    """后台线程: matplotlib 实时显示 P(hit) 曲线及双阈值."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle("Real-time Detection Monitor", fontsize=13)

    line_prob, = ax.plot([], [], "b-", lw=1.2)
    ax.axhline(y=0.5, color="gray", ls="--", lw=0.8, alpha=0.6, label="decision boundary")
    ax.axhline(y=prob_high, color="red", ls="--", lw=1, alpha=0.7, label=f"high={prob_high}")
    ax.axhline(y=prob_low, color="orange", ls="--", lw=0.8, alpha=0.7, label=f"low={prob_low}")
    ax.set_ylabel("P(hit)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    while detector.running:
        with detector.lock:
            probs = list(detector.plot_prob)

        if len(probs) > 1:
            t = np.arange(len(probs)) / 60.0  # frames → seconds
            line_prob.set_data(t, probs)
            ax.set_xlim(t[0], t[-1] + 0.1)

        try:
            plt.pause(0.1)
        except Exception:
            break

    plt.close()


if __name__ == "__main__":
    main()
