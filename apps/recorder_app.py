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
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QIODevice, QTimer, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET
from apps.shared.file_saver import stash

STYLE_SHEET = APP_STYLE_SHEET


class LiveRecorderPlotWidget(QWidget):
    def __init__(
        self,
        imu_points: deque[tuple[float, float, float]],
        serial_points: deque[tuple[float, float]],
    ):
        super().__init__()
        self.imu_points = imu_points
        self.serial_points = serial_points
        self.window_seconds = 6.0
        self.setMinimumHeight(260)

    def paintEvent(self, event):
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        margin = 10
        gap = 10
        panel_h = max(90, (self.height() - margin * 2 - gap) // 2)
        imu_rect = (margin, margin, self.width() - margin * 2, panel_h)
        serial_rect = (
            margin,
            margin + panel_h + gap,
            self.width() - margin * 2,
            panel_h,
        )

        now = self._latest_elapsed()
        start = max(0.0, now - self.window_seconds)
        self._draw_imu_panel(painter, imu_rect, start, now)
        self._draw_serial_panel(painter, serial_rect, start, now)

    def _latest_elapsed(self) -> float:
        latest = 0.0
        if self.imu_points:
            latest = max(latest, self.imu_points[-1][0])
        if self.serial_points:
            latest = max(latest, self.serial_points[-1][0])
        return latest

    def _draw_imu_panel(
        self,
        painter: QPainter,
        rect: tuple[int, int, int, int],
        start: float,
        end: float,
    ) -> None:
        points = [point for point in self.imu_points if point[0] >= start]
        self._draw_frame(painter, rect, "IMU |acc| / |gyr|")
        if len(points) < 2:
            self._draw_empty(painter, rect, "等待 IMU 数据")
            return

        ts = np.asarray([p[0] for p in points], dtype=np.float64)
        acc = np.asarray([p[1] for p in points], dtype=np.float32)
        gyr = np.asarray([p[2] for p in points], dtype=np.float32)
        y_min, y_max = self._range(np.concatenate([acc, gyr]))
        x, y, w, h = self._plot_rect(rect)
        painter.setPen(QPen(QColor("#E53E3E"), 1))
        self._draw_series(painter, x, y, w, h, ts, acc, start, end, y_min, y_max)
        painter.setPen(QPen(QColor("#2563EB"), 1))
        self._draw_series(painter, x, y, w, h, ts, gyr, start, end, y_min, y_max)

    def _draw_serial_panel(
        self,
        painter: QPainter,
        rect: tuple[int, int, int, int],
        start: float,
        end: float,
    ) -> None:
        points = [point for point in self.serial_points if point[0] >= start]
        self._draw_frame(painter, rect, "Serial value")
        if len(points) < 2:
            self._draw_empty(painter, rect, "等待可解析串口数值")
            return

        ts = np.asarray([p[0] for p in points], dtype=np.float64)
        values = np.asarray([p[1] for p in points], dtype=np.float32)
        y_min, y_max = self._range(values)
        x, y, w, h = self._plot_rect(rect)
        painter.setPen(QPen(QColor("#059669"), 1))
        self._draw_series(painter, x, y, w, h, ts, values, start, end, y_min, y_max)

    def _draw_frame(self, painter: QPainter, rect: tuple[int, int, int, int], title: str) -> None:
        x, y, w, h = rect
        painter.setPen(QColor("#e2e8f0"))
        painter.drawRect(x, y, w, h)
        painter.setPen(QColor("#64748b"))
        painter.drawText(x + 8, y + 18, title)

    def _draw_empty(self, painter: QPainter, rect: tuple[int, int, int, int], text: str) -> None:
        x, y, w, h = rect
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(x + 56, y + h // 2 + 4, text)

    def _plot_rect(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = rect
        axis_w = 46
        return x + axis_w, y + 22, max(10, w - axis_w - 6), max(10, h - 28)

    def _range(self, values: np.ndarray) -> tuple[float, float]:
        ymin = float(np.min(values))
        ymax = float(np.max(values))
        pad = (ymax - ymin) * 0.12 + 1e-3
        if ymax - ymin < 1e-6:
            pad = max(pad, 1.0)
        return ymin - pad, ymax + pad

    def _draw_series(
        self,
        painter: QPainter,
        x: int,
        y: int,
        w: int,
        h: int,
        ts: np.ndarray,
        data: np.ndarray,
        ts_min: float,
        ts_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        denom_x = max(1e-6, ts_max - ts_min)
        denom_y = max(1e-6, y_max - y_min)
        x_vals = x + ((ts - ts_min) / denom_x) * w
        y_vals = y + h - ((data - y_min) / denom_y) * h

        last_x = int(x_vals[0])
        last_y = int(y_vals[0])
        for idx in range(1, x_vals.size):
            px = int(x_vals[idx])
            py = int(y_vals[idx])
            painter.drawLine(last_x, last_y, px, py)
            last_x, last_y = px, py


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
        self.serial_records: list[list[float | str]] = []
        self.serial_line_buffer = bytearray()
        self.preview_imu_points: deque[tuple[float, float, float]] = deque(maxlen=2400)
        self.preview_serial_points: deque[tuple[float, float]] = deque(maxlen=2400)
        self.falling_count = 0
        self.rising_count = 0
        self.imu_frame_count = 0
        self.audio_frame_count = 0
        self.serial_line_count = 0
        self.imu_recent_frames: deque[tuple[float, int]] = deque()
        self.audio_recent_frames: deque[tuple[float, int]] = deque()
        self.serial_recent_lines: deque[tuple[float, int]] = deque()

        self.serial = QSerialPort(self)
        self.serial.setBaudRate(115200)
        self.serial.readyRead.connect(self.on_serial_ready_read)
        self.serial.errorOccurred.connect(self.on_serial_error)

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")
        self.serial_status_label = QLabel("串口状态：未连接")
        self.serial_status_label.setObjectName("statusTag")
        self.metric_label = QLabel("就绪")
        self.realtime_fps_label = QLabel("实时帧率：--")
        self.overall_fps_label = QLabel("总体帧率：--")
        self.duration_label = QLabel("录制时长：00:00:00")
        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(250)
        self.metrics_timer.timeout.connect(self.refresh_metrics)
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)
        self.preview_timer.timeout.connect(self.refresh_preview)

        self._build_ui()
        self.set_stream_status_label(self.status_label)
        self._bind_signals()
        self.refresh_serial_ports()
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
        desc = QLabel("可选择某几个戒指，并同时录制 IMU、音频或 Arduino 串口数据。")
        desc.setProperty("class", "sub-text")

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.serial_status_label)

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
        self.chk_serial = QCheckBox("录制串口")
        self.chk_serial.setChecked(True)
        stream_row.addWidget(self.chk_imu)
        stream_row.addWidget(self.chk_audio)
        stream_row.addWidget(self.chk_serial)
        stream_row.addStretch()
        panel_layout.addLayout(stream_row)

        serial_row = QHBoxLayout()
        self.serial_combo = QComboBox()
        self.serial_combo.setMinimumWidth(300)
        self.btn_refresh_serial = QPushButton("刷新串口")
        self.btn_refresh_serial.setProperty("class", "btn-gray")
        self.btn_connect_serial = QPushButton("连接串口")
        self.btn_connect_serial.setProperty("class", "btn-success")
        serial_row.addWidget(QLabel("Arduino"))
        serial_row.addWidget(self.serial_combo)
        serial_row.addWidget(self.btn_refresh_serial)
        serial_row.addWidget(self.btn_connect_serial)
        serial_row.addStretch()
        panel_layout.addLayout(serial_row)

        name_row = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("可选：输入文件命名后缀")
        self.name_input.setMinimumWidth(300)
        name_row.addWidget(QLabel("文件命名"))
        name_row.addWidget(self.name_input)
        name_row.addStretch()
        panel_layout.addLayout(name_row)

        actions = QHBoxLayout()
        self.btn_start = QPushButton("开始录制")
        self.btn_start.setProperty("class", "btn-success")
        self.btn_stop = QPushButton("停止并保存")
        self.btn_stop.setProperty("class", "btn-danger")
        self.btn_stop.setEnabled(False)
        self.btn_delete_latest = QPushButton("删除最新录制")
        self.btn_delete_latest.setProperty("class", "btn-gray")
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_stop)
        actions.addWidget(self.btn_delete_latest)
        actions.addStretch()
        panel_layout.addLayout(actions)

        panel_layout.addWidget(self.metric_label)
        panel_layout.addWidget(self.realtime_fps_label)
        panel_layout.addWidget(self.overall_fps_label)
        panel_layout.addWidget(self.duration_label)
        root.addWidget(panel)

        preview_panel = QFrame()
        preview_panel.setProperty("class", "panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        preview_layout.setSpacing(10)
        preview_title = QLabel("实时预览")
        preview_title.setProperty("class", "title")
        preview_layout.addWidget(preview_title)
        self.serial_edge_count_label = QLabel("串口事件：FALLING 0 | RISING 0")
        self.serial_edge_count_label.setObjectName("valuePill")
        preview_layout.addWidget(self.serial_edge_count_label)
        self.preview_plot = LiveRecorderPlotWidget(
            imu_points=self.preview_imu_points,
            serial_points=self.preview_serial_points,
        )
        preview_layout.addWidget(self.preview_plot)
        preview_layout.addWidget(QLabel("串口消息"))
        self.serial_log = QListWidget()
        self.serial_log.setMaximumHeight(150)
        self.serial_log.setAlternatingRowColors(True)
        preview_layout.addWidget(self.serial_log)
        root.addWidget(preview_panel, stretch=1)

    def _bind_signals(self):
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_delete_latest.clicked.connect(self.delete_latest_recording)
        self.btn_refresh_serial.clicked.connect(self.refresh_serial_ports)
        self.btn_connect_serial.clicked.connect(self.toggle_serial_connection)

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
        if not streams and not self.chk_serial.isChecked():
            self.metric_label.setText("请至少选择一种录制数据（IMU、音频或串口）")
            return
        if self.chk_serial.isChecked() and not self.serial.isOpen():
            self.metric_label.setText("录制串口前请先连接 Arduino 串口")
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
        self.serial_records.clear()
        self.serial_line_buffer.clear()
        self.preview_imu_points.clear()
        self.preview_serial_points.clear()
        self.falling_count = 0
        self.rising_count = 0
        self.update_serial_edge_count_label()
        self.serial_log.clear()
        self.imu_frame_count = 0
        self.audio_frame_count = 0
        self.serial_line_count = 0
        self.imu_recent_frames.clear()
        self.audio_recent_frames.clear()
        self.serial_recent_lines.clear()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_delete_latest.setEnabled(False)
        self.metric_label.setText(
            f"录制中... rings={len(rings) if rings else 'ALL'} streams={','.join(streams) or 'none'}"
            + (" + serial" if self.chk_serial.isChecked() else "")
        )
        self.refresh_metrics()
        self.metrics_timer.start()
        self.preview_timer.start()

    def stop_recording(self):
        if not self.recording:
            return
        stop_time = time.perf_counter()
        self.last_record_elapsed = max(stop_time - self.record_start_time, 0.0)
        self.recording = False
        self.metrics_timer.stop()
        self.preview_timer.stop()

        self.send_subscription(["imu"], empty_behavior="none")

        saved = self.save_records()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_delete_latest.setEnabled(True)
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
        elapsed = (
            time.perf_counter() - self.record_start_time
            if self.record_start_time > 0
            else 0.0
        )
        acc_norm = float(np.linalg.norm(np.asarray(acc, dtype=np.float32)))
        gyr_norm = float(np.linalg.norm(np.asarray(gyr, dtype=np.float32)))
        self.preview_imu_points.append((elapsed, acc_norm, gyr_norm))
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

    def refresh_serial_ports(self) -> None:
        previous = self.serial_combo.currentData()
        self.serial_combo.clear()
        for info in QSerialPortInfo.availablePorts():
            port_name = info.systemLocation()
            description = info.description() or info.portName()
            self.serial_combo.addItem(f"{port_name} | {description}", port_name)

        if self.serial_combo.count() == 0:
            self.serial_combo.addItem("未发现串口", "")
            self.serial_status_label.setText("串口状态：未发现串口")
            return

        preferred = str(previous or "")
        for index in range(self.serial_combo.count()):
            port_name = str(self.serial_combo.itemData(index) or "")
            if preferred and port_name == preferred:
                self.serial_combo.setCurrentIndex(index)
                return
        for index in range(self.serial_combo.count()):
            port_name = str(self.serial_combo.itemData(index) or "")
            if "usbserial" in port_name.lower() or "wch" in port_name.lower():
                self.serial_combo.setCurrentIndex(index)
                return

    def toggle_serial_connection(self) -> None:
        if self.serial.isOpen():
            self.serial.close()
            self.btn_connect_serial.setText("连接串口")
            self.serial_status_label.setText("串口状态：已断开")
            return

        port_name = str(self.serial_combo.currentData() or "")
        if not port_name:
            self.serial_status_label.setText("串口状态：未选择有效串口")
            return

        self.serial.setPortName(port_name)
        self.serial.setBaudRate(115200)
        if not self.serial.open(QIODevice.OpenModeFlag.ReadOnly):
            self.serial_status_label.setText(f"串口状态：连接失败（{self.serial.errorString()}）")
            return
        self.btn_connect_serial.setText("断开串口")
        self.serial_status_label.setText(f"串口状态：已连接 {port_name} @ 115200")

    def on_serial_ready_read(self) -> None:
        chunk = bytes(self.serial.readAll())
        if not chunk:
            return
        self.serial_line_buffer.extend(chunk)
        while b"\n" in self.serial_line_buffer:
            raw_line, _, rest = self.serial_line_buffer.partition(b"\n")
            self.serial_line_buffer = bytearray(rest)
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not self.recording or not self.chk_serial.isChecked():
                continue
            now_wall = time.time()
            elapsed = (
                time.perf_counter() - self.record_start_time
                if self.record_start_time > 0
                else 0.0
            )
            self.serial_records.append([now_wall, elapsed, line])
            self._append_serial_preview(elapsed, line)
            self._register_frame("serial", 1)

    def on_serial_error(self, error) -> None:
        if error == QSerialPort.SerialPortError.NoError:
            return
        self.serial_status_label.setText(f"串口状态：异常（{self.serial.errorString()}）")

    def _append_serial_preview(self, elapsed: float, line: str) -> None:
        if "FALLING" in line:
            self.falling_count += 1
            self.update_serial_edge_count_label()
        elif "RISING" in line:
            self.rising_count += 1
            self.update_serial_edge_count_label()

        self.serial_log.insertItem(0, f"{elapsed:7.3f}s | {line}")
        while self.serial_log.count() > 120:
            self.serial_log.takeItem(self.serial_log.count() - 1)

        value = self._parse_serial_value(line)
        if value is not None:
            self.preview_serial_points.append((elapsed, value))

    def update_serial_edge_count_label(self) -> None:
        self.serial_edge_count_label.setText(
            f"串口事件：FALLING {self.falling_count} | RISING {self.rising_count}"
        )

    def _parse_serial_value(self, line: str) -> float | None:
        raw_match = re.search(r"\braw=(-?\d+(?:\.\d+)?)", line)
        if raw_match:
            return float(raw_match.group(1))

        parts = [part.strip() for part in line.split(",")]
        numeric_parts: list[float] = []
        for part in parts:
            if not part:
                continue
            try:
                numeric_parts.append(float(part))
            except ValueError:
                continue
        if not numeric_parts:
            return None
        return numeric_parts[-1]

    def save_records(self) -> int:
        count = 0
        start_ts = self.record_start_ts or int(time.time())
        suffix = self._recording_suffix(start_ts)

        if self.imu_records:
            os.makedirs("data", exist_ok=True)
            for address, rows in self.imu_records.items():
                if not rows:
                    continue
                ring_name = str(self.ring_info.get(address, {}).get("name", "Unknown"))
                safe_name = self._safe_name(ring_name)
                safe_address = self._safe_name(address)
                plain_name = f"{safe_name}_{safe_address}_{suffix}.npy"
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
                plain_name = f"{safe_name}_{safe_address}_{suffix}.wav"
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

        if self.serial_records:
            os.makedirs("data", exist_ok=True)
            plain_name = f"serial_{suffix}.csv"
            filename = f"data/{plain_name}.enc"
            buffer = io.StringIO()
            buffer.write("wall_time,elapsed_s,line\n")
            for wall_time, elapsed, line in self.serial_records:
                escaped = str(line).replace('"', '""')
                buffer.write(f'{float(wall_time):.6f},{float(elapsed):.6f},"{escaped}"\n')
            stash(
                filename,
                buffer.getvalue().encode("utf-8"),
                original_name=plain_name,
            )
            count += 1

        return count

    def _recording_suffix(self, start_ts: int) -> str:
        raw_tag = self.name_input.text().strip()
        if not raw_tag:
            return str(start_ts)
        tag = self._safe_name(raw_tag)
        return f"{start_ts}_{tag}"

    def delete_latest_recording(self) -> None:
        if self.recording:
            self.metric_label.setText("正在录制，不能删除最新结果")
            return

        latest_batch = self._latest_recording_batch()
        if latest_batch is None:
            self.metric_label.setText("没有找到可删除的录制结果")
            return

        paths = self._recording_paths_for_batch(latest_batch)
        deleted = 0
        for path in paths:
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                self.metric_label.setText(f"删除失败: {path.name} ({exc})")
                return

        self.metric_label.setText(f"已删除最新录制 batch={latest_batch}，文件数: {deleted}")

    def _latest_recording_batch(self) -> int | None:
        batches: list[int] = []
        data_dir = Path("data")
        if not data_dir.exists():
            return None
        for path in data_dir.iterdir():
            batch = self._batch_from_recording_path(path)
            if batch is not None:
                batches.append(batch)
        return max(batches) if batches else None

    def _recording_paths_for_batch(self, batch: int) -> list[Path]:
        data_dir = Path("data")
        if not data_dir.exists():
            return []
        return sorted(
            path
            for path in data_dir.iterdir()
            if self._batch_from_recording_path(path) == batch
        )

    def _batch_from_recording_path(self, path: Path) -> int | None:
        name = path.name
        patterns = [
            r".+_(\d+)(?:_[A-Za-z0-9._-]+)?\.npy\.enc$",
            r".+_(\d+)(?:_[A-Za-z0-9._-]+)?\.wav\.enc$",
            r"serial_(\d+)(?:_[A-Za-z0-9._-]+)?\.csv\.enc$",
            r"imu_serial_overlay_(\d+)(?:_[A-Za-z0-9._-]+)?\.png$",
        ]
        for pattern in patterns:
            match = re.fullmatch(pattern, name)
            if match:
                return int(match.group(1))
        return None

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
        elif stream == "serial":
            self.serial_line_count += count
            self.serial_recent_lines.append((now, count))
        self._trim_recent_frames(now)

    def _trim_recent_frames(self, now: float) -> None:
        window_seconds = 1.0
        cutoff = now - window_seconds
        while self.imu_recent_frames and self.imu_recent_frames[0][0] < cutoff:
            self.imu_recent_frames.popleft()
        while self.audio_recent_frames and self.audio_recent_frames[0][0] < cutoff:
            self.audio_recent_frames.popleft()
        while self.serial_recent_lines and self.serial_recent_lines[0][0] < cutoff:
            self.serial_recent_lines.popleft()

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
        if self.chk_serial.isChecked():
            parts.append(f"串口 {self.serial_line_count} 行")
        return " | ".join(parts) if parts else "--"

    def refresh_metrics(self) -> None:
        if self.recording and self.record_start_time > 0:
            now = time.perf_counter()
            elapsed = max(now - self.record_start_time, 1e-6)
            self._trim_recent_frames(now)
            realtime_window = min(elapsed, 1.0)
            realtime_imu_fps = sum(count for _, count in self.imu_recent_frames) / realtime_window
            realtime_audio_fps = sum(count for _, count in self.audio_recent_frames) / realtime_window
            realtime_serial_fps = sum(count for _, count in self.serial_recent_lines) / realtime_window
            overall_imu_fps = self.imu_frame_count / elapsed
            overall_audio_fps = self.audio_frame_count / elapsed

            self.realtime_fps_label.setText(
                f"实时帧率：{self._format_fps_group(realtime_imu_fps, realtime_audio_fps)}"
                + (f" | 串口 {realtime_serial_fps:.1f}行/s" if self.chk_serial.isChecked() else "")
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

    def refresh_preview(self) -> None:
        self.preview_plot.update()

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        if self.serial.isOpen():
            self.serial.close()
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
