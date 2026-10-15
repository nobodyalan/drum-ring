#!/usr/bin/env python
"""测量 play() 调用 → 实际出声 的音频链路延迟（WASAPI loopback 法）。

相比 WDM-KS 回环, WASAPI loopback 缓冲小(数 ms)、时间戳准确, 是 Windows 上
测量音频输出延迟的可靠方法。

原理:
  在已知 perf_counter 时刻向输出流写入短促 click, 同时通过 WASAPI 回环
  设备录制系统输出, 检测 click 起音点, 两者之差即为「播放→回环出声」延迟。
  该延迟 ≈ 输出缓冲 + 捕获缓冲 (回环捕获缓冲很小, 约 3ms)。

用法:
  python scripts/measure_audio_latency_wasapi.py --list-devices
  python scripts/measure_audio_latency_wasapi.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyaudiowpatch as pyaudio

DEFAULT_FS = 48000
CLICK_FREQ = 1000.0
CLICK_DURATION = 0.01
CLICK_AMP = 0.9
BLOCK = 512
SETTLE_S = 0.4
DEFAULT_N_CLICKS = 12
DEFAULT_INTERVAL = 0.35


def make_click(fs: int = DEFAULT_FS, freq: float = CLICK_FREQ,
               duration: float = CLICK_DURATION, amp: float = CLICK_AMP) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    envelope = np.exp(-t * 400.0)
    x = np.sin(2 * np.pi * freq * t) * envelope
    x = x / max(np.max(np.abs(x)), 1e-6) * amp
    return x.astype(np.float32)


def get_wasapi_devices(p: pyaudio.PyAudio) -> tuple[dict, dict]:
    """返回 (默认输出设备, 匹配的 loopback 输入设备)。"""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    loopback = None
    if not default_out.get("isLoopbackDevice"):
        for lb in p.get_loopback_device_info_generator():
            # 优先匹配默认输出设备的 loopback
            if lb["name"].replace(" [Loopback]", "") == default_out["name"]:
                loopback = lb
                break
            loopback = lb if loopback is None else loopback
    return default_out, loopback


def detect_onsets(mono: np.ndarray, fs: int, n_clicks: int) -> list[int]:
    win = max(int(fs * 0.002), 16)
    hop = win // 2
    n_windows = (len(mono) - win) // hop + 1
    rms = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        seg = mono[i * hop:i * hop + win]
        rms[i] = np.sqrt(np.mean(seg * seg))
    noise_floor = float(np.median(rms))
    threshold = max(noise_floor * 8.0, 1e-4)
    onsets: list[int] = []
    active = False
    min_sep = int(fs * 0.1)
    for i in range(n_windows):
        above = rms[i] > threshold
        if above and not active:
            onsets.append(i * hop)
            active = True
        elif not above:
            active = False
        if len(onsets) >= 2 and onsets[-1] - onsets[-2] < min_sep:
            onsets.pop()
    return onsets[:n_clicks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--fs", type=int, default=DEFAULT_FS)
    parser.add_argument("--n-clicks", type=int, default=DEFAULT_N_CLICKS)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    args = parser.parse_args()

    fs = int(args.fs)
    click = make_click(fs=fs)
    click_stereo = np.column_stack([click, click])
    n_clicks = int(args.n_clicks)

    p = pyaudio.PyAudio()
    default_out, loopback = get_wasapi_devices(p)

    if args.list_devices:
        print(f"默认输出: [{default_out['index']}] {default_out['name']} "
              f"out_lat={default_out['defaultLowOutputLatency']*1000:.1f}ms")
        if loopback:
            print(f"回环输入: [{loopback['index']}] {loopback['name']} "
                  f"in_lat={loopback['defaultLowInputLatency']*1000:.1f}ms")
        else:
            print("未找到 WASAPI loopback 设备")
        p.terminate()
        return 0

    if loopback is None:
        print("✗ 未找到 WASAPI loopback 设备")
        p.terminate()
        return 1

    print(f"✓ 输出设备: [{default_out['index']}] {default_out['name']} "
          f"(上报输出延迟 {default_out['defaultLowOutputLatency']*1000:.1f}ms)")
    print(f"✓ 回环设备: [{loopback['index']}] {loopback['name']} "
          f"(上报输入延迟 {loopback['defaultLowInputLatency']*1000:.1f}ms)")

    blocks: list[tuple[float, int, np.ndarray]] = []

    def input_callback(in_data, frame_count, time_info, status):
        arr = np.frombuffer(in_data, dtype=np.float32).reshape(-1, 2).copy()
        blocks.append((time.perf_counter(), int(frame_count), arr))
        return (None, pyaudio.paContinue)

    in_stream = p.open(
        format=pyaudio.paFloat32, channels=2, rate=fs,
        input=True, input_device_index=loopback["index"],
        frames_per_buffer=BLOCK, stream_callback=input_callback,
    )
    out_stream = p.open(
        format=pyaudio.paFloat32, channels=2, rate=fs,
        output=True, output_device_index=default_out["index"],
        frames_per_buffer=BLOCK,
    )

    in_stream.start_stream()
    out_stream.start_stream()
    time.sleep(SETTLE_S)

    play_times: list[float] = []
    for _ in range(n_clicks):
        play_times.append(time.perf_counter())
        out_stream.write(click_stereo.tobytes())
        time.sleep(args.interval)

    time.sleep(SETTLE_S)
    out_stream.stop_stream()
    in_stream.stop_stream()
    out_stream.close()
    in_stream.close()
    p.terminate()

    if not blocks:
        print("✗ 录音为空")
        return 1

    audio = np.concatenate([b[2] for b in blocks], axis=0)
    block_frames = blocks[0][1]
    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-3:
        print(f"✗ 录音几乎无声 (peak={peak:.5f})")
        return 1

    onsets = detect_onsets(mono, fs, n_clicks)
    print(f"  检测到 {len(onsets)} 个起音点 (预期 {n_clicks})")

    # sample → 绝对时间: 第 b 个 block 在 blocks[b][0] 交付, 最后一帧约在该时刻被捕获
    def sample_abs_time(s: int) -> float:
        b = min(s // block_frames, len(blocks) - 1)
        j = s % block_frames
        return blocks[b][0] - (block_frames - 1 - j) / fs

    onset_abs = [sample_abs_time(s) for s in onsets]
    latencies: list[float] = []
    onset_iter = 0
    for play_t in play_times:
        while onset_iter < len(onset_abs) and onset_abs[onset_iter] < play_t:
            onset_iter += 1
        if onset_iter >= len(onset_abs):
            break
        latencies.append((onset_abs[onset_iter] - play_t) * 1000.0)
        onset_iter += 1

    if not latencies:
        print("✗ 未能配对播放与起音")
        return 1

    lat_arr = np.asarray(latencies)
    in_lat = float(loopback["defaultLowInputLatency"]) * 1000.0
    print("\n=== 实测音频链路延迟 (play → 回环出声) ===")
    print(f"  样本数: {len(lat_arr)}")
    print(f"  均值: {lat_arr.mean():.1f} ms")
    print(f"  最小: {lat_arr.min():.1f} ms")
    print(f"  最大: {lat_arr.max():.1f} ms")
    print(f"  标准差: {lat_arr.std():.1f} ms")
    print(f"\n  估计纯输出延迟 (播放→出声) ≈ {lat_arr.mean() - in_lat:.1f} ms "
          f"(扣除回环捕获 {in_lat:.1f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
