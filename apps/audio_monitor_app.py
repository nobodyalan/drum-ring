import argparse
import base64
import os
import struct
import sys
import time
import wave

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET

try:
    from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

    HAS_AUDIO_PLAYBACK = True
except Exception:
    HAS_AUDIO_PLAYBACK = False

STYLE_SHEET = APP_STYLE_SHEET


class AdpcmDecoder:
    index_table = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]

    step_size_table =  [7,8,9,10,11,12,13,14,16,17,
                        19,21,23,25,28,31,34,37,41,45,
                        50,55,60,66,73,80,88,97,107,118,
                        130,143,157,173,190,209,230,253,279,307,
                        337,371,408,449,494,544,598,658,724,796,
                        876,963,1060,1166,1282,1411,1552,1707,1878,2066,
                        2272,2499,2749,3024,3327,3660,4026,4428,4871,5358,
                        5894,6484,7132,7845,8630,9493,10442,11487,12635,13899,
                        15289,16818,18500,20350,22385,24623,27086,29794,32767]

    def __init__(self):
        self.predsample = 0
        self.pcm_index = 0

    def decode(self, code: int) -> int:
        step = self.step_size_table[self.pcm_index]
        diff = step >> 3
        if code & 4:
            diff += step
        if code & 2:
            diff += step >> 1
        if code & 1:
            diff += step >> 2

        if code & 8:
            diff = -diff
        self.predsample += diff

        if self.predsample > 32767:
            self.predsample = 32767
        elif self.predsample < -32768:
            self.predsample = -32768

        self.pcm_index += self.index_table[code]
        if self.pcm_index < 0:
            self.pcm_index = 0
        if self.pcm_index > 88:
            self.pcm_index = 88
        return self.predsample

    def decode_packet(self, data: bytes) -> bytes:
        pcm = bytearray()
        for value in data:
            pcm.extend(struct.pack("<h", self.decode(value & 0xF)))
            pcm.extend(struct.pack("<h", self.decode((value >> 4) & 0xF)))
        return bytes(pcm)


class AudioPlayer:
    def __init__(self, sample_rate: int = 16000):
        self.enabled = HAS_AUDIO_PLAYBACK
        self._sink = None
        self._io_device = None
        if not self.enabled:
            return

        fmt = QAudioFormat()
        fmt.setSampleRate(sample_rate)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        output = QMediaDevices.defaultAudioOutput()
        self._sink = QAudioSink(output, fmt)
        # Keep latency low without forcing aggressive underruns.
        self._sink.setBufferSize(6144)
        self._io_device = self._sink.start()

    def available_bytes(self) -> int:
        if not self.enabled or self._sink is None:
            return 0
        return max(0, int(self._sink.bytesFree()))

    def write(self, pcm_bytes: bytes) -> tuple[int, int]:
        if not self.enabled or self._io_device is None:
            return 0, len(pcm_bytes)
        if not pcm_bytes:
            return 0, 0

        writable = int(self._sink.bytesFree()) & ~1
        if writable <= 0:
            return 0, len(pcm_bytes)
        chunk = pcm_bytes[:writable]
        dropped = len(pcm_bytes) - len(chunk)

        written = self._io_device.write(chunk)
        written = max(0, int(written))
        if written < len(chunk):
            dropped += len(chunk) - written
        return written, dropped

    def close(self):
        if self._sink is not None:
            self._sink.stop()


class AudioMonitorWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("音频监控应用")
        self.resize(900, 620)

        self.decoder = AdpcmDecoder()
        self.player = AudioPlayer(sample_rate=16000)

        self.decoded_pcm = bytearray()
        self.packet_count = 0
        self.sample_count = 0
        self.play_drop_count = 0
        self.play_drop_bytes = 0
        self.play_compress_count = 0
        self.play_compress_saved_samples = 0
        self.last_play_sample: int = 0

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")
        self.metric_label = QLabel("音频包: 0 | 样本数: 0")

        self._build_ui()
        self.set_stream_status_label(self.status_label)
        self._bind_signals()
        self.client.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        panel = QFrame()
        panel.setProperty("class", "panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(10)

        title = QLabel("音频流监控")
        title.setProperty("class", "title")
        desc = QLabel("可按戒指订阅音频，实时解码播放，并可选保存 WAV。")
        desc.setProperty("class", "sub-text")

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.status_label)

        if not HAS_AUDIO_PLAYBACK:
            warn = QLabel("当前环境不可用实时播放，仅支持保存。")
            warn.setProperty("class", "sub-text")
            panel_layout.addWidget(warn)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择订阅戒指（不勾选表示接收全部）",
            max_height=160,
        )

        options = QHBoxLayout()
        self.play_checkbox = QCheckBox("实时播放")
        self.play_checkbox.setChecked(HAS_AUDIO_PLAYBACK)
        self.play_checkbox.setEnabled(HAS_AUDIO_PLAYBACK)

        self.save_checkbox = QCheckBox("缓存并可保存")
        self.save_checkbox.setChecked(True)

        options.addWidget(self.play_checkbox)
        options.addWidget(self.save_checkbox)
        options.addStretch()
        panel_layout.addLayout(options)

        actions = QHBoxLayout()
        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")

        self.save_btn = QPushButton("保存 WAV")
        self.save_btn.setProperty("class", "btn-success")

        self.clear_btn = QPushButton("清空缓存")
        self.clear_btn.setProperty("class", "btn-primary")

        actions.addWidget(self.apply_btn)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.clear_btn)
        actions.addStretch()
        panel_layout.addLayout(actions)

        panel_layout.addWidget(self.metric_label)

        root.addWidget(panel)

    def _bind_signals(self):
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.save_btn.clicked.connect(self.save_audio)
        self.clear_btn.clicked.connect(self.clear_audio_cache)

        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def handle_stream_message(self, message: dict):
        if message.get("type") != "audio":
            return

        audio_type = int(message.get("audio_type", -1))
        if audio_type != 1:
            return

        data_b64 = message.get("data_b64", "")
        if not isinstance(data_b64, str) or not data_b64:
            return

        try:
            raw_audio = base64.b64decode(data_b64)
        except Exception:
            return

        pcm = self.decoder.decode_packet(raw_audio)

        if self.play_checkbox.isChecked():
            pcm_for_play = pcm
            bytes_free = self.player.available_bytes()
            if bytes_free > 0 and len(pcm_for_play) > bytes_free:
                compressed = self.time_compress_to_fit(pcm_for_play, bytes_free)
                if compressed is not None:
                    orig_samples = len(pcm_for_play) // 2
                    new_samples = len(compressed) // 2
                    self.play_compress_count += 1
                    self.play_compress_saved_samples += max(0, orig_samples - new_samples)
                    pcm_for_play = compressed
                else:
                    # Extreme congestion: skip this packet as a last resort.
                    self.play_drop_count += 1
                    self.play_drop_bytes += len(pcm_for_play)
                    pcm_for_play = b""

            pcm_for_play = self.apply_declick(pcm_for_play)
            _, dropped_bytes = self.player.write(pcm_for_play)
            if dropped_bytes > 0:
                self.play_drop_count += 1
                self.play_drop_bytes += dropped_bytes

        if self.save_checkbox.isChecked():
            self.decoded_pcm.extend(pcm)

        self.packet_count += 1
        self.sample_count += len(pcm) // 2
        self.metric_label.setText(
            " | ".join(
                [
                    f"音频包: {self.packet_count}",
                    f"样本数: {self.sample_count}",
                    f"压缩次数: {self.play_compress_count}",
                    f"压缩节省样本: {self.play_compress_saved_samples}",
                    f"播放丢弃: {self.play_drop_count}",
                    f"丢弃字节: {self.play_drop_bytes}",
                    f"缓存: {len(self.decoded_pcm)} bytes",
                ]
            )
        )

    def apply_subscription(self):
        self.send_subscription(["audio"])

    def clear_audio_cache(self):
        self.decoded_pcm.clear()
        self.packet_count = 0
        self.sample_count = 0
        self.play_drop_count = 0
        self.play_drop_bytes = 0
        self.play_compress_count = 0
        self.play_compress_saved_samples = 0
        self.last_play_sample = 0
        self.decoder = AdpcmDecoder()
        self.metric_label.setText("音频包: 0 | 样本数: 0")

    def save_audio(self):
        if not self.decoded_pcm:
            self.status_label.setText("无可保存音频缓存")
            return

        os.makedirs("audio", exist_ok=True)
        filename = os.path.join("audio", f"result{int(time.time())}.wav")
        try:
            with wave.open(filename, "wb") as wavfile:
                wavfile.setnchannels(1)
                wavfile.setsampwidth(2)
                wavfile.setframerate(16000)
                wavfile.writeframes(bytes(self.decoded_pcm))
            self.status_label.setText(f"保存成功: {filename}")
        except Exception as exc:
            self.status_label.setText(f"保存失败: {exc}")

    def closeEvent(self, event):
        self.client.stop()
        self.client.wait(1500)
        self.player.close()
        event.accept()

    def apply_declick(self, pcm_bytes: bytes, ramp_samples: int = 64) -> bytes:
        if len(pcm_bytes) < 2:
            return pcm_bytes
        sample_count = len(pcm_bytes) // 2
        n = min(ramp_samples, sample_count)
        out = bytearray(pcm_bytes)
        for i in range(n):
            idx = i * 2
            current = struct.unpack("<h", out[idx : idx + 2])[0]
            alpha = (i + 1) / n
            blended = int(self.last_play_sample * (1.0 - alpha) + current * alpha)
            out[idx : idx + 2] = struct.pack("<h", blended)
        self.last_play_sample = struct.unpack("<h", out[-2:])[0]
        return bytes(out)

    def time_compress_to_fit(self, pcm_bytes: bytes, max_bytes: int) -> bytes | None:
        target_bytes = max_bytes & ~1
        if target_bytes < 4:
            return None
        if target_bytes >= len(pcm_bytes):
            return pcm_bytes

        src = np.frombuffer(pcm_bytes, dtype=np.int16)
        src_n = int(src.shape[0])
        target_n = target_bytes // 2
        if src_n < 4 or target_n < 4:
            return None

        ratio = target_n / src_n
        if ratio < 0.35:
            # Compression too aggressive, quality would be poor.
            return None

        x_old = np.arange(src_n, dtype=np.float64)
        x_new = np.linspace(0, src_n - 1, num=target_n, dtype=np.float64)
        y_new = np.interp(x_new, x_old, src.astype(np.float64))
        y_new = np.clip(y_new, -32768, 32767).astype(np.int16)
        return y_new.tobytes()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: AudioMonitorWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
