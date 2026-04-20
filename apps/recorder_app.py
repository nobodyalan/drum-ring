import argparse
import base64
import io
import os
import re
import struct
import sys
import time
import wave
from collections import deque

import numpy as np
from PyQt6.QtCore import QTimer, Qt
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
from apps.shared.file_saver import stash

STYLE_SHEET = APP_STYLE_SHEET

class AdpcmDecoder:
    index_table = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]
    step_size_table = [
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
        19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
        50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
        130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
        337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
        876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
        2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
        5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
        15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
    ]

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


class RecorderWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("录制中心")
        self.resize(940, 680)

        self.recording = False
        self.record_start_ts = 0
        self.record_start_time = 0.0
        self.last_record_elapsed = 0.0
        self.imu_records: dict[str, list[list[float]]] = {}
        self.audio_pcm: dict[str, bytearray] = {}
        self.audio_decoders: dict[str, AdpcmDecoder] = {}
        self.imu_frame_count = 0
        self.audio_frame_count = 0
        self.imu_recent_frames: deque[tuple[float, int]] = deque()
        self.audio_recent_frames: deque[tuple[float, int]] = deque()

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")
        self.metric_label = QLabel("就绪")
        self.realtime_fps_label = QLabel("实时帧率：--")
        self.overall_fps_label = QLabel("总体帧率：--")
        self.duration_label = QLabel("录制时长：00:00:00")
        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(250)
        self.metrics_timer.timeout.connect(self.refresh_metrics)

        self._build_ui()
        self.set_stream_status_label(self.status_label)
        self._bind_signals()
        self.refresh_metrics()
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

        title = QLabel("多戒指录制应用")
        title.setProperty("class", "title")
        desc = QLabel("可选择某几个戒指，并同时录制 IMU 或音频数据。")
        desc.setProperty("class", "sub-text")

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.status_label)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择要录制的戒指（不勾选表示全部）",
            max_height=180,
        )

        stream_row = QHBoxLayout()
        self.chk_imu = QCheckBox("录制 IMU")
        self.chk_imu.setChecked(True)
        self.chk_audio = QCheckBox("录制音频")
        self.chk_audio.setChecked(True)
        stream_row.addWidget(self.chk_imu)
        stream_row.addWidget(self.chk_audio)
        stream_row.addStretch()
        panel_layout.addLayout(stream_row)

        actions = QHBoxLayout()
        self.btn_start = QPushButton("开始录制")
        self.btn_start.setProperty("class", "btn-success")
        self.btn_stop = QPushButton("停止并保存")
        self.btn_stop.setProperty("class", "btn-danger")
        self.btn_stop.setEnabled(False)
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_stop)
        actions.addStretch()
        panel_layout.addLayout(actions)

        panel_layout.addWidget(self.metric_label)
        panel_layout.addWidget(self.realtime_fps_label)
        panel_layout.addWidget(self.overall_fps_label)
        panel_layout.addWidget(self.duration_label)
        root.addWidget(panel)

    def _bind_signals(self):
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)

        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.send_subscription(["imu"], empty_behavior="none")

    def handle_stream_message(self, message: dict):
        msg_type = message.get("type")
        if not self.recording:
            return

        if msg_type == "imu" and self.chk_imu.isChecked():
            self.on_imu_message(message)
        elif msg_type == "audio" and self.chk_audio.isChecked():
            self.on_audio_message(message)

    def selected_streams(self) -> list[str]:
        streams = []
        if self.chk_imu.isChecked():
            streams.append("imu")
        if self.chk_audio.isChecked():
            streams.append("audio")
        return streams

    def start_recording(self):
        streams = self.selected_streams()
        if not streams:
            self.metric_label.setText("请至少选择一种录制数据（IMU 或音频）")
            return

        rings = self.build_subscription_rings()
        self.client.send_command({"cmd": "subscribe", "rings": rings, "streams": streams})

        self.record_start_ts = int(time.time())
        self.record_start_time = time.perf_counter()
        self.last_record_elapsed = 0.0
        self.recording = True
        self.imu_records.clear()
        self.audio_pcm.clear()
        self.audio_decoders.clear()
        self.imu_frame_count = 0
        self.audio_frame_count = 0
        self.imu_recent_frames.clear()
        self.audio_recent_frames.clear()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.metric_label.setText(
            f"录制中... rings={len(rings) if rings else 'ALL'} streams={','.join(streams)}"
        )
        self.refresh_metrics()
        self.metrics_timer.start()

    def stop_recording(self):
        if not self.recording:
            return
        stop_time = time.perf_counter()
        self.last_record_elapsed = max(stop_time - self.record_start_time, 0.0)
        self.recording = False
        self.metrics_timer.stop()

        self.send_subscription(["imu"], empty_behavior="none")

        saved = self.save_records()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.metric_label.setText(f"录制完成，保存文件数: {saved}")
        self.refresh_metrics()

    def on_imu_message(self, message: dict):
        address = str(message.get("address", ""))
        if not address:
            return
        acc = message.get("acc")
        gyr = message.get("gyr")
        if not isinstance(acc, list) or not isinstance(gyr, list):
            return
        if len(acc) != 3 or len(gyr) != 3:
            return
        local_ts = float(message.get("local_timestamp", 0.0) or 0.0)
        host_ts = float(message.get("host_timestamp", 0.0) or 0.0)
        row = [local_ts, host_ts] + [float(v) for v in acc] + [float(v) for v in gyr]
        self.imu_records.setdefault(address, []).append(row)
        self._register_frame("imu", 1)

    def on_audio_message(self, message: dict):
        address = str(message.get("address", ""))
        if not address:
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

        decoder = self.audio_decoders.setdefault(address, AdpcmDecoder())
        pcm = decoder.decode_packet(raw_audio)
        self.audio_pcm.setdefault(address, bytearray()).extend(pcm)
        self._register_frame("audio", 1)

    def save_records(self) -> int:
        count = 0
        start_ts = self.record_start_ts or int(time.time())

        if self.imu_records:
            os.makedirs("data", exist_ok=True)
            for address, rows in self.imu_records.items():
                if not rows:
                    continue
                ring_name = str(self.ring_info.get(address, {}).get("name", "Unknown"))
                safe_name = self._safe_name(ring_name)
                safe_address = self._safe_name(address)
                plain_name = f"{safe_name}_{safe_address}_{start_ts}.npy"
                filename = f"data/{plain_name}.enc"
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(rows, dtype=np.float64))
                stash(
                    filename,
                    buffer.getvalue(),
                    original_name=plain_name,
                )
                count += 1

        if self.audio_pcm:
            os.makedirs("data", exist_ok=True)
            for address, pcm in self.audio_pcm.items():
                if not pcm:
                    continue
                ring_name = str(self.ring_info.get(address, {}).get("name", "Unknown"))
                safe_name = self._safe_name(ring_name)
                safe_address = self._safe_name(address)
                plain_name = f"{safe_name}_{safe_address}_{start_ts}.wav"
                filename = f"data/{plain_name}.enc"
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wavfile:
                    wavfile.setnchannels(1)
                    wavfile.setsampwidth(2)
                    wavfile.setframerate(16000)
                    wavfile.writeframes(bytes(pcm))
                stash(
                    filename,
                    buffer.getvalue(),
                    original_name=plain_name,
                )
                count += 1

        return count

    def _safe_name(self, text: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        return sanitized or "ring"

    def _register_frame(self, stream: str, count: int) -> None:
        now = time.perf_counter()
        if stream == "imu":
            self.imu_frame_count += count
            self.imu_recent_frames.append((now, count))
        elif stream == "audio":
            self.audio_frame_count += count
            self.audio_recent_frames.append((now, count))
        self._trim_recent_frames(now)

    def _trim_recent_frames(self, now: float) -> None:
        window_seconds = 1.0
        cutoff = now - window_seconds
        while self.imu_recent_frames and self.imu_recent_frames[0][0] < cutoff:
            self.imu_recent_frames.popleft()
        while self.audio_recent_frames and self.audio_recent_frames[0][0] < cutoff:
            self.audio_recent_frames.popleft()

    def _format_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_fps_group(self, imu_fps: float, audio_fps: float) -> str:
        parts: list[str] = []
        if self.chk_imu.isChecked():
            parts.append(f"IMU {imu_fps:.1f}Hz")
        if self.chk_audio.isChecked():
            parts.append(f"音频 {audio_fps:.1f}Hz")
        return " | ".join(parts) if parts else "--"

    def refresh_metrics(self) -> None:
        if self.recording and self.record_start_time > 0:
            now = time.perf_counter()
            elapsed = max(now - self.record_start_time, 1e-6)
            self._trim_recent_frames(now)
            realtime_window = min(elapsed, 1.0)
            realtime_imu_fps = sum(count for _, count in self.imu_recent_frames) / realtime_window
            realtime_audio_fps = sum(count for _, count in self.audio_recent_frames) / realtime_window
            overall_imu_fps = self.imu_frame_count / elapsed
            overall_audio_fps = self.audio_frame_count / elapsed

            self.realtime_fps_label.setText(
                f"实时帧率：{self._format_fps_group(realtime_imu_fps, realtime_audio_fps)}"
            )
            self.overall_fps_label.setText(
                f"总体帧率：{self._format_fps_group(overall_imu_fps, overall_audio_fps)}"
            )
            self.duration_label.setText(f"录制时长：{self._format_duration(elapsed)}")
            return

        if self.record_start_time > 0 and self.last_record_elapsed > 0:
            overall_imu_fps = self.imu_frame_count / self.last_record_elapsed
            overall_audio_fps = self.audio_frame_count / self.last_record_elapsed
            self.realtime_fps_label.setText("实时帧率：--")
            self.overall_fps_label.setText(
                f"总体帧率：{self._format_fps_group(overall_imu_fps, overall_audio_fps)}"
            )
            self.duration_label.setText(
                f"录制时长：{self._format_duration(self.last_record_elapsed)}"
            )
            return

        self.realtime_fps_label.setText("实时帧率：--")
        self.overall_fps_label.setText("总体帧率：--")
        self.duration_label.setText("录制时长：00:00:00")

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: RecorderWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
