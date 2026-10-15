#!/usr/bin/env python
"""直接测量 LowLatencyDrumPlayer 的输出延迟 (play → DAC)。

不依赖回环录音, 而是利用 PortAudio 回调提供的 outputBufferDacTime
(样本实际送达 DAC 的流时间), 精确计算 play() 调用到出声的延迟,
避免 loopback 捕获缓冲的污染。

用法:
  python scripts/measure_player_latency.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FS = 48000
CLICK_FRAMES = 480          # 10ms click
N_CLICKS = 12
INTERVAL = 0.35
SETTLE_S = 0.4


def find_wasapi_output() -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and "WASAPI" in sd.query_hostapis(d["hostapi"])["name"]:
            return i
    return None


def main() -> int:
    buf = np.zeros((FS * 8, 2), dtype=np.float32)
    lock = threading.Lock()
    read_idx = 0
    results: list[tuple[float, float]] = []
    pending_play: float | None = None

    def callback(outdata, frames, time_info, status):
        nonlocal read_idx, pending_play
        with lock:
            idx = read_idx
            end = idx + frames
            if end <= len(buf):
                outdata[:] = buf[idx:end]
                buf[idx:end] = 0.0
            else:
                n1 = len(buf) - idx
                outdata[:n1] = buf[idx:]
                outdata[n1:] = buf[:end - len(buf)]
                buf[idx:] = 0.0
                buf[:end - len(buf)] = 0.0
            read_idx = end % len(buf)
            has_click = np.any(outdata != 0)
            if has_click and pending_play is not None:
                dac_stream = float(time_info.outputBufferDacTime)
                cur_stream = float(time_info.currentTime)
                dac_perf = time.perf_counter() + (dac_stream - cur_stream)
                results.append((pending_play, dac_perf))
                pending_play = None

    device = find_wasapi_output()
    stream = sd.OutputStream(
        samplerate=FS, channels=2, dtype="float32", blocksize=256,
        latency="low", device=device, callback=callback,
    )
    stream.start()
    print(f"流延迟 (stream.latency): {stream.latency * 1000:.1f} ms")
    time.sleep(SETTLE_S)

    click = np.ones((CLICK_FRAMES, 2), dtype=np.float32) * 0.5
    for _ in range(N_CLICKS):
        play_perf = time.perf_counter()
        with lock:
            idx = read_idx
            buf[idx:idx + CLICK_FRAMES] += click
        pending_play = play_perf
        time.sleep(INTERVAL)

    time.sleep(SETTLE_S)
    stream.stop()
    stream.close()

    if not results:
        print("✗ 未检测到 click 输出")
        return 1

    lat = np.array([(dac - play) * 1000.0 for play, dac in results])
    print(f"样本数: {len(lat)}")
    print("实测 play → DAC 延迟:")
    print(f"  均值: {lat.mean():.1f} ms")
    print(f"  最小: {lat.min():.1f} ms")
    print(f"  最大: {lat.max():.1f} ms")
    print(f"  标准差: {lat.std():.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
