"""低延迟鼓音播放器。

用持久 sounddevice OutputStream + 环形混合缓冲替代 QSoundEffect, 消除
Qt 音频引擎的大缓冲, 显著降低「播放 → 出声」延迟。

设计要点:
- 持久输出流: 打开一次, 持续运行, 避免每次播放的流启动延迟。
- 环形混合缓冲: play() 把样本叠加到「当前播放位置」之后, 回调逐帧取走并清零,
  天然支持多音重叠 (复音)。
- 非阻塞: play() 只在锁内做一次 numpy 叠加, 音频由 PortAudio 回调线程输出。
"""

from __future__ import annotations

import threading
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly

FS = 48000
CHANNELS = 2
BLOCKSIZE = 256            # ~5.3ms
BUFFER_SECONDS = 8.0       # 音色已截短至 8s，缓冲回落到原值（play_samples 含截断保护）
MAX_SOUND_SECONDS = 1.5    # 加载音色时截短尾部, 减少快速打击时的重叠削波


def _load_wav(path: str | Path, fs: int = FS, channels: int = CHANNELS) -> np.ndarray:
    rate, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    if int(rate) != int(fs):
        g = gcd(int(rate), int(fs))
        x = resample_poly(x, int(fs) // g, int(rate) // g, axis=0).astype(np.float32)
    if x.ndim == 1:
        x = np.column_stack([x, x])
    elif x.shape[1] == 1:
        x = np.repeat(x, channels, axis=1)
    else:
        x = x[:, :channels]
    # 截短音色尾部: 减少快速打击时长尾音在环形缓冲里的重叠削波
    max_frames = int(fs * MAX_SOUND_SECONDS)
    if x.shape[0] > max_frames:
        x = x[:max_frames].copy()
    return np.ascontiguousarray(x, dtype=np.float32)


def _soft_clip(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """软限幅: |x| ≤ threshold 保持原值; 超过部分用 tanh 平滑压缩, 输出收敛到 1。"""
    peak = float(max(x.max(), -x.min()))
    if peak <= threshold:
        return x
    y = x.copy()
    over = np.abs(y) > threshold
    excess = np.abs(y[over]) - threshold
    y[over] = np.sign(y[over]) * (
        threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold))
    )
    return y


def find_wasapi_output_device() -> int | None:
    """查找 Windows WASAPI 输出设备编号 (低延迟); 找不到返回 None。

    注意: WASAPI 信息在宿主 API 名称里, 不在设备名称里。
    """
    try:
        for index, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) > 0:
                hostapi = sd.query_hostapis(dev["hostapi"])["name"]
                if "WASAPI" in hostapi:
                    return index
    except Exception:
        pass
    return None


class LowLatencyDrumPlayer:
    def __init__(
        self,
        fs: int = FS,
        channels: int = CHANNELS,
        blocksize: int = BLOCKSIZE,
        device: int | None = None,
        buffer_seconds: float = BUFFER_SECONDS,
    ) -> None:
        self.fs = int(fs)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.buf_size = int(self.fs * buffer_seconds)
        self.buf = np.zeros((self.buf_size, self.channels), dtype=np.float32)
        self._lock = threading.Lock()
        self._read_idx = 0
        self._sounds: dict[int, np.ndarray] = {}

        if device is None:
            device = find_wasapi_output_device()
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            latency="low",
            device=device,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        with self._lock:
            idx = self._read_idx
            end = idx + frames
            if end <= self.buf_size:
                outdata[:] = self.buf[idx:end]
                self.buf[idx:end] = 0.0
            else:
                n1 = self.buf_size - idx
                outdata[:n1] = self.buf[idx:]
                outdata[n1:] = self.buf[:end - self.buf_size]
                self.buf[idx:] = 0.0
                self.buf[:end - self.buf_size] = 0.0
            self._read_idx = end % self.buf_size

    def load_sound(self, class_id: int, path: str | Path) -> None:
        self._sounds[int(class_id)] = _load_wav(path, self.fs, self.channels)

    def play(self, class_id: int, gain: float = 1.0) -> None:
        samples = self._sounds.get(int(class_id))
        if samples is None:
            return
        self.play_samples(samples, gain)

    def play_samples(self, samples: np.ndarray, gain: float = 1.0) -> None:
        """叠加播放任意样本 (测量用)。samples: (n, channels) float32。"""
        samples = np.asarray(samples, dtype=np.float32)
        if gain != 1.0:
            samples = (samples * float(gain)).astype(np.float32, copy=False)
        with self._lock:
            idx = self._read_idx
            n = len(samples)
            # 样本比环形缓冲长时只取前 buf_size 帧，避免环绕越界/形状不匹配
            if n > self.buf_size:
                samples = samples[: self.buf_size]
                n = self.buf_size

            # ── 重叠衰减 (防止密集打击叠加成满幅糊墙 → 电子音) ──
            # 若新样本将覆盖的区域已有前音在响, 等比降低新音增益, 使叠加峰值
            # 稳定在 TARGET 附近: 单击响度不变, 密集打击自然衰减。
            end = idx + n
            if end <= self.buf_size:
                existing = self.buf[idx:end]
                existing_peak = float(max(existing.max(), -existing.min())) if n else 0.0
            else:
                n1 = self.buf_size - idx
                e1 = float(max(self.buf[idx:].max(), -self.buf[idx:].min()))
                e2 = float(max(self.buf[: end - self.buf_size].max(), -self.buf[: end - self.buf_size].min()))
                existing_peak = max(e1, e2)
            if existing_peak > 0.15:
                sp = float(max(samples.max(), -samples.min()))
                if sp > 1e-6:
                    headroom = 0.85 - existing_peak
                    scale = min(1.0, headroom / sp)
                    if scale < 1.0:
                        samples = (samples * max(0.05, scale)).astype(np.float32, copy=False)

            if end <= self.buf_size:
                self.buf[idx:end] += samples
                self.buf[idx:end] = _soft_clip(self.buf[idx:end])
            else:
                n1 = self.buf_size - idx
                self.buf[idx:] += samples[:n1]
                self.buf[idx:] = _soft_clip(self.buf[idx:])
                self.buf[:end - self.buf_size] += samples[n1:]
                self.buf[:end - self.buf_size] = _soft_clip(self.buf[:end - self.buf_size])

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
