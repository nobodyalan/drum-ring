import argparse
import base64
import os
import re
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

from common.stream_window import RingStreamWindowBase

STYLE_SHEET = """
QMainWindow { background-color: #f6f8fb; }
QFrame.panel {
    background-color: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 12px;
}
QPushButton {
    border-radius: 8px;
    padding: 8px 14px;
    border: none;
    font-weight: 600;
    color: white;
    min-height: 34px;
}
QPushButton.btn-primary { background-color: #3182ce; }
QPushButton.btn-primary:hover { background-color: #2b6cb0; }
QPushButton.btn-success { background-color: #38a169; }
QPushButton.btn-danger { background-color: #e53e3e; }
QLabel { font-size: 13px; color: #334155; }
QLabel.title { font-size: 18px; font-weight: 700; color: #0f172a; }
QLabel.sub-text { font-size: 12px; color: #64748b; }
"""

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
        self.imu_records: dict[str, list[list[float]]] = {}
        self.audio_pcm: dict[str, bytearray] = {}
        self.audio_decoders: dict[str, AdpcmDecoder] = {}

        self.status_label = QLabel("数据流状态：未连接")
        self.metric_label = QLabel("就绪")

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
        self.chk_audio.setChecked(False)
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
        self.recording = True
        self.imu_records.clear()
        self.audio_pcm.clear()
        self.audio_decoders.clear()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.metric_label.setText(
            f"录制中... rings={len(rings) if rings else 'ALL'} streams={','.join(streams)}"
        )

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False

        self.send_subscription(["imu"], empty_behavior="none")

        saved = self.save_records()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.metric_label.setText(f"录制完成，保存文件数: {saved}")

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
                filename = f"data/{safe_name}_{address}_{start_ts}.npy"
                np.save(filename, np.asarray(rows, dtype=np.float64))
                count += 1

        if self.audio_pcm:
            os.makedirs("data", exist_ok=True)
            for address, pcm in self.audio_pcm.items():
                if not pcm:
                    continue
                ring_name = str(self.ring_info.get(address, {}).get("name", "Unknown"))
                safe_name = self._safe_name(ring_name)
                filename = f"data/{safe_name}_{address}_{start_ts}.wav"
                with wave.open(filename, "wb") as wavfile:
                    wavfile.setnchannels(1)
                    wavfile.setsampwidth(2)
                    wavfile.setframerate(16000)
                    wavfile.writeframes(bytes(pcm))
                count += 1

        return count

    def _safe_name(self, text: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        return sanitized or "ring"

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17888)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = RecorderWindow(host=args.host, port=args.port)
    window.show()
    sys.exit(app.exec())
