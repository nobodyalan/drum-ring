import argparse
import base64
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET

STYLE_SHEET = APP_STYLE_SHEET

ACC_AXIS_COLORS = [QColor("#E53E3E"), QColor("#F59E0B"), QColor("#10B981")]
GYR_AXIS_COLORS = [QColor("#2563EB"), QColor("#7C3AED"), QColor("#EC4899")]


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
        self.predsample = max(-32768, min(32767, self.predsample))
        self.pcm_index += self.index_table[code]
        self.pcm_index = max(0, min(88, self.pcm_index))
        return self.predsample

    def decode_packet(self, data: bytes) -> bytes:
        pcm = bytearray()
        for value in data:
            pcm.extend(struct.pack("<h", self.decode(value & 0xF)))
            pcm.extend(struct.pack("<h", self.decode((value >> 4) & 0xF)))
        return bytes(pcm)


@dataclass
class RingBuffer:
    capacity: int = 12000

    def __post_init__(self):
        self.timestamps = np.zeros(self.capacity, dtype=np.float64)
        self.acc = np.zeros((self.capacity, 3), dtype=np.float32)
        self.gyr = np.zeros((self.capacity, 3), dtype=np.float32)
        self.write_idx = 0
        self.count = 0

    def append(self, timestamp: float, acc: np.ndarray, gyr: np.ndarray) -> None:
        idx = self.write_idx
        self.timestamps[idx] = timestamp
        self.acc[idx] = acc
        self.gyr[idx] = gyr
        self.write_idx = (idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def latest_timestamp(self) -> float:
        if self.count == 0:
            return 0.0
        latest_idx = (self.write_idx - 1) % self.capacity
        return float(self.timestamps[latest_idx])

    def recent(self, since_ts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        if self.count < self.capacity:
            ts = self.timestamps[: self.count]
            acc = self.acc[: self.count]
            gyr = self.gyr[: self.count]
        else:
            ts = np.concatenate(
                [self.timestamps[self.write_idx :], self.timestamps[: self.write_idx]],
                axis=0,
            )
            acc = np.concatenate(
                [self.acc[self.write_idx :], self.acc[: self.write_idx]],
                axis=0,
            )
            gyr = np.concatenate(
                [self.gyr[self.write_idx :], self.gyr[: self.write_idx]],
                axis=0,
            )

        mask = ts >= since_ts
        return ts[mask], acc[mask], gyr[mask]


@dataclass
class AudioLevelBuffer:
    capacity: int = 4096

    def __post_init__(self):
        self.timestamps = np.zeros(self.capacity, dtype=np.float64)
        self.rms = np.zeros(self.capacity, dtype=np.float32)
        self.peak = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx = 0
        self.count = 0

    def append(self, timestamp: float, rms: float, peak: float) -> None:
        idx = self.write_idx
        self.timestamps[idx] = timestamp
        self.rms[idx] = rms
        self.peak[idx] = peak
        self.write_idx = (idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def latest_timestamp(self) -> float:
        if self.count == 0:
            return 0.0
        latest_idx = (self.write_idx - 1) % self.capacity
        return float(self.timestamps[latest_idx])

    def recent(self, since_ts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        if self.count < self.capacity:
            ts = self.timestamps[: self.count]
            rms = self.rms[: self.count]
            peak = self.peak[: self.count]
        else:
            ts = np.concatenate(
                [self.timestamps[self.write_idx :], self.timestamps[: self.write_idx]],
                axis=0,
            )
            rms = np.concatenate(
                [self.rms[self.write_idx :], self.rms[: self.write_idx]],
                axis=0,
            )
            peak = np.concatenate(
                [self.peak[self.write_idx :], self.peak[: self.write_idx]],
                axis=0,
            )

        mask = ts >= since_ts
        return ts[mask], rms[mask], peak[mask]


class RingMiniPlotWidget(QWidget):
    def __init__(
        self,
        address: str,
        ring_name: str,
        buffer_ref: RingBuffer,
        audio_buffer_ref: AudioLevelBuffer,
    ):
        super().__init__()
        self.address = address
        self.ring_name = ring_name
        self.buffer_ref = buffer_ref
        self.audio_buffer_ref = audio_buffer_ref
        self.window_seconds = 5.0
        self.auto_scale = True
        self.acc_scale = 20.0
        self.gyr_scale = 1200.0
        self.setMinimumHeight(320)

    def set_window_seconds(self, value: float) -> None:
        self.window_seconds = value

    def set_scale_mode(self, auto_scale: bool, acc_scale: float, gyr_scale: float) -> None:
        self.auto_scale = auto_scale
        self.acc_scale = acc_scale
        self.gyr_scale = gyr_scale

    def paintEvent(self, event):
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        latest_ts = self.buffer_ref.latest_timestamp()
        latest_audio_ts = self.audio_buffer_ref.latest_timestamp()
        window_end = max(latest_ts, latest_audio_ts, time.time())
        since_ts = window_end - self.window_seconds
        ts, acc, gyr = self.buffer_ref.recent(since_ts)
        audio_ts, audio_rms, audio_peak = self.audio_buffer_ref.recent(since_ts)

        margin = 10
        top_space = 4
        total_h = self.height() - top_space - margin * 4
        imu_panel_h = int(total_h * 0.38)
        audio_panel_h = max(56, total_h - imu_panel_h * 2)
        acc_rect = (margin, top_space + margin, self.width() - margin * 2, imu_panel_h)
        gyr_rect = (
            margin,
            top_space + margin * 2 + imu_panel_h,
            self.width() - margin * 2,
            imu_panel_h,
        )
        audio_rect = (
            margin,
            top_space + margin * 3 + imu_panel_h * 2,
            self.width() - margin * 2,
            audio_panel_h,
        )

        self._draw_panel(
            painter,
            acc_rect,
            ts,
            acc,
            self.acc_scale,
            axis_colors=ACC_AXIS_COLORS,
        )
        self._draw_panel(
            painter,
            gyr_rect,
            ts,
            gyr,
            self.gyr_scale,
            axis_colors=GYR_AXIS_COLORS,
        )
        self._draw_audio_panel(painter, audio_rect, audio_ts, audio_rms, audio_peak)

    def _draw_panel(
        self,
        painter: QPainter,
        rect: tuple[int, int, int, int],
        ts: np.ndarray,
        values: np.ndarray,
        fixed_scale: float,
        axis_colors: list[QColor],
    ):
        x, y, w, h = rect
        painter.setPen(QColor("#e2e8f0"))
        painter.drawRect(x, y, w, h)

        finite_mask = np.isfinite(ts)
        if values.ndim == 2 and values.shape[1] == 3:
            finite_mask &= np.all(np.isfinite(values), axis=1)
        ts = ts[finite_mask]
        values = values[finite_mask]

        if ts.size < 2 or values.shape[0] < 2:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(x + 12, y + h // 2 + 4, "等待 IMU 数据")
            return

        y_min, y_max = self._range(values, fixed_scale)
        axis_w = 46
        plot_x = x + axis_w
        plot_w = max(10, w - axis_w - 4)
        self._draw_y_axis_labels(
            painter=painter,
            x=x,
            y=y,
            h=h,
            axis_w=axis_w,
            y_min=y_min,
            y_max=y_max,
        )

        for axis in range(3):
            painter.setPen(QPen(axis_colors[axis], 1))
            self._draw_series(
                painter,
                x=plot_x,
                y=y,
                w=plot_w,
                h=h,
                ts=ts,
                data=values[:, axis],
                ts_min=ts[0],
                ts_max=ts[-1],
                y_min=y_min,
                y_max=y_max,
            )

    def _draw_y_axis_labels(
        self,
        *,
        painter: QPainter,
        x: int,
        y: int,
        h: int,
        axis_w: int,
        y_min: float,
        y_max: float,
    ) -> None:
        painter.setPen(QColor("#94a3b8"))
        axis_x = x + axis_w
        painter.drawLine(axis_x, y, axis_x, y + h)

        mid = (y_min + y_max) * 0.5
        labels = [
            (y + 12, y_max),
            (y + h // 2 + 4, mid),
            (y + h - 4, y_min),
        ]
        for py, value in labels:
            painter.drawLine(axis_x - 4, py, axis_x, py)
            painter.drawText(x + 2, py + 4, f"{value:.2f}")

    def _range(self, values: np.ndarray, fixed_scale: float) -> tuple[float, float]:
        if not self.auto_scale:
            return -fixed_scale, fixed_scale

        ymin = float(np.min(values))
        ymax = float(np.max(values))
        pad = (ymax - ymin) * 0.12 + 1e-3
        if ymax - ymin < 1e-6:
            pad = max(pad, 1.0)
        return ymin - pad, ymax + pad

    def _draw_audio_panel(
        self,
        painter: QPainter,
        rect: tuple[int, int, int, int],
        ts: np.ndarray,
        rms: np.ndarray,
        peak: np.ndarray,
    ) -> None:
        x, y, w, h = rect
        painter.setPen(QColor("#e2e8f0"))
        painter.drawRect(x, y, w, h)
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(x + 8, y + 16, "AUDIO")

        if ts.size < 2 or rms.size < 2 or peak.size < 2:
            painter.drawText(x + 56, y + h // 2 + 4, "音频未传输")
            return

        axis_w = 34
        plot_x = x + axis_w
        plot_w = max(10, w - axis_w - 6)
        painter.drawLine(plot_x, y, plot_x, y + h)

        self._draw_audio_fill(
            painter,
            x=plot_x,
            y=y,
            w=plot_w,
            h=h,
            ts=ts,
            data=peak,
            ts_min=ts[0],
            ts_max=ts[-1],
            color=QColor(245, 158, 11, 70),
        )
        painter.setPen(QPen(QColor("#0EA5E9"), 1))
        self._draw_series(
            painter,
            x=plot_x,
            y=y,
            w=plot_w,
            h=h,
            ts=ts,
            data=rms,
            ts_min=ts[0],
            ts_max=ts[-1],
            y_min=0.0,
            y_max=1.0,
        )

    def _draw_audio_fill(
        self,
        painter: QPainter,
        *,
        x: int,
        y: int,
        w: int,
        h: int,
        ts: np.ndarray,
        data: np.ndarray,
        ts_min: float,
        ts_max: float,
        color: QColor,
    ) -> None:
        if data.size < 2:
            return

        max_points = max(80, w // 2)
        stride = max(1, data.size // max_points)
        if stride > 1:
            ts = ts[::stride]
            data = data[::stride]

        denom_x = max(1e-6, ts_max - ts_min)
        x_vals = x + ((ts - ts_min) / denom_x) * w
        baseline = y + h
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        bar_w = max(1, w // max(1, data.size))
        for idx in range(data.size):
            bar_h = int(float(np.clip(data[idx], 0.0, 1.0)) * (h - 2))
            painter.drawRect(int(x_vals[idx]), baseline - bar_h, bar_w, bar_h)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_series(
        self,
        painter: QPainter,
        *,
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
        if data.size < 2:
            return

        max_points = max(120, w)
        stride = max(1, data.size // max_points)
        if stride > 1:
            ts = ts[::stride]
            data = data[::stride]

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


class RingPlotCard(QFrame):
    def __init__(
        self,
        address: str,
        ring_name: str,
        buffer_ref: RingBuffer,
        audio_buffer_ref: AudioLevelBuffer,
    ):
        super().__init__()
        self.setProperty("class", "panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        header = QLabel(f"{ring_name} | {address}")
        header.setProperty("class", "title")
        layout.addWidget(header)

        legend_row = QHBoxLayout()
        legend_row.setSpacing(14)
        legend_row.addWidget(self._legend_label("ACC X", ACC_AXIS_COLORS[0]))
        legend_row.addWidget(self._legend_label("ACC Y", ACC_AXIS_COLORS[1]))
        legend_row.addWidget(self._legend_label("ACC Z", ACC_AXIS_COLORS[2]))
        legend_row.addSpacing(8)
        legend_row.addWidget(self._legend_label("GYR X", GYR_AXIS_COLORS[0]))
        legend_row.addWidget(self._legend_label("GYR Y", GYR_AXIS_COLORS[1]))
        legend_row.addWidget(self._legend_label("GYR Z", GYR_AXIS_COLORS[2]))
        legend_row.addStretch()
        layout.addLayout(legend_row)

        unit_row = QHBoxLayout()
        unit_row.setSpacing(18)
        unit_acc = QLabel("ACC (m/s^2)")
        unit_acc.setProperty("class", "sub-text")
        unit_gyro = QLabel("GYR (rad/s)")
        unit_gyro.setProperty("class", "sub-text")
        unit_audio = QLabel("AUDIO (peak + RMS)")
        unit_audio.setProperty("class", "sub-text")
        unit_row.addWidget(unit_acc)
        unit_row.addWidget(unit_gyro)
        unit_row.addWidget(unit_audio)
        unit_row.addStretch()
        layout.addLayout(unit_row)

        self.plot = RingMiniPlotWidget(
            address=address,
            ring_name=ring_name,
            buffer_ref=buffer_ref,
            audio_buffer_ref=audio_buffer_ref,
        )
        layout.addWidget(self.plot)

    def _legend_label(self, text: str, color: QColor) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {color.name()}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        return label


class ImuVisualizerWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("高性能 IMU/音频可视化")
        self.resize(1400, 1000)

        self.buffers: dict[str, RingBuffer] = {}
        self.audio_buffers: dict[str, AudioLevelBuffer] = {}
        self.audio_decoders: dict[str, AdpcmDecoder] = {}
        self.cards: dict[str, RingPlotCard] = {}
        self._subscription_applied = False
        self._max_swipe_log_entries = 120
        self.imu_message_count = 0
        self.audio_message_count = 0

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")
        self.stream_counter_label = QLabel("流消息：IMU 0 | AUDIO 0")
        self.stream_counter_label.setProperty("class", "sub-text")

        self._build_ui()
        self.set_stream_status_label(self.status_label)
        self._bind_signals()
        self.on_scale_controls_changed()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.refresh_plots)
        self.timer.start()

        self.client.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        top_panel = QFrame()
        top_panel.setProperty("class", "panel")
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(16, 14, 16, 14)
        top_layout.setSpacing(10)

        title = QLabel("多戒指 IMU/音频实时可视化")
        title.setProperty("class", "title")
        subtitle = QLabel(
            "每个戒指卡片同时显示 IMU 曲线，并在有音频流时绘制低开销的音频能量视图；"
            "收到 Swipe 手势事件时会显示在界面下方事件列表。"
        )
        subtitle.setProperty("class", "sub-text")
        subtitle.setWordWrap(True)

        top_layout.addWidget(title)
        top_layout.addWidget(subtitle)
        top_layout.addWidget(self.status_label)
        top_layout.addWidget(self.stream_counter_label)

        ring_scroll = self.add_ring_selection_panel(
            top_layout,
            title_text="选择订阅戒指",
            max_height=150,
        )
        self.ring_layout.setSpacing(8)
        self.ring_layout.setContentsMargins(8, 8, 8, 8)

        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(8)
        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        self.apply_btn.setMinimumWidth(120)
        controls.addWidget(self.apply_btn, 0, 0, 1, 1)

        controls.addWidget(QLabel("窗口(秒)"), 0, 1)
        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setRange(2, 20)
        self.window_slider.setValue(5)
        self.window_slider.setMinimumWidth(220)
        controls.addWidget(self.window_slider, 0, 2)
        self.window_label = QLabel("5s")
        self.window_label.setObjectName("valuePill")
        controls.addWidget(self.window_label, 0, 3)

        self.auto_scale_box = QCheckBox("自动 Y 轴")
        self.auto_scale_box.setChecked(True)
        controls.addWidget(self.auto_scale_box, 0, 4)

        controls.addWidget(QLabel("ACC scale"), 1, 1)
        self.acc_scale_input = QDoubleSpinBox()
        self.acc_scale_input.setRange(1.0, 200.0)
        self.acc_scale_input.setSingleStep(1.0)
        self.acc_scale_input.setDecimals(1)
        self.acc_scale_input.setValue(100.0)
        self.acc_scale_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.acc_scale_input.setFixedWidth(120)
        controls.addWidget(self.acc_scale_input, 1, 2)

        controls.addWidget(QLabel("GYR scale"), 1, 3)
        self.gyr_scale_input = QDoubleSpinBox()
        self.gyr_scale_input.setRange(0.01, 50.0)
        self.gyr_scale_input.setSingleStep(0.1)
        self.gyr_scale_input.setDecimals(2)
        self.gyr_scale_input.setValue(20.0)
        self.gyr_scale_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.gyr_scale_input.setFixedWidth(120)
        controls.addWidget(self.gyr_scale_input, 1, 4)
        controls.setColumnStretch(5, 1)
        top_layout.addLayout(controls)

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setStyleSheet("border: none;")
        self.cards_content = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_content)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cards_layout.setSpacing(12)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_scroll.setWidget(self.cards_content)
        self.empty_hint = QLabel("请先勾选戒指并点击“应用订阅”后显示可视化。")
        self.empty_hint.setProperty("class", "sub-text")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_layout.addWidget(self.empty_hint)

        root.addWidget(top_panel)
        self.cards_hint = QLabel("请先勾选戒指并点击“应用订阅”后显示可视化。")
        self.cards_hint.setProperty("class", "sub-text")
        self.cards_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_hint.setWordWrap(True)
        self.swipe_log = QListWidget()
        self.swipe_log.setMaximumHeight(180)
        self.swipe_log.setAlternatingRowColors(True)
        root.addWidget(self.cards_hint)
        root.addWidget(self.cards_scroll)
        root.addWidget(QLabel("Swipe 事件"))
        root.addWidget(self.swipe_log)
        self.cards_scroll.setVisible(False)

    def _bind_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.window_slider.valueChanged.connect(self.on_window_changed)
        self.auto_scale_box.stateChanged.connect(self.on_scale_controls_changed)
        self.acc_scale_input.valueChanged.connect(self.on_scale_controls_changed)
        self.gyr_scale_input.valueChanged.connect(self.on_scale_controls_changed)

        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        if self._subscription_applied:
            self.apply_subscription()
        self._refresh_cards()

    def on_window_changed(self, value: int) -> None:
        self.window_label.setText(f"{value}s")
        for card in self.cards.values():
            card.plot.set_window_seconds(float(value))

    def on_scale_controls_changed(self, _value=None) -> None:
        auto_mode = self.auto_scale_box.isChecked()
        self.acc_scale_input.setEnabled(not auto_mode)
        self.gyr_scale_input.setEnabled(not auto_mode)
        for card in self.cards.values():
            card.plot.set_scale_mode(
                auto_scale=auto_mode,
                acc_scale=float(self.acc_scale_input.value()),
                gyr_scale=float(self.gyr_scale_input.value()),
            )

    def handle_stream_message(self, message: dict) -> None:
        msg_type = message.get("type")
        if msg_type == "subscribed":
            self.status_label.setText(
                "数据流状态：已连接"
                + f" | rings={message.get('rings', [])}"
                + f" | streams={message.get('streams', [])}"
            )
            return
        if msg_type == "imu":
            self._handle_imu_message(message)
            return
        if msg_type == "audio":
            self._handle_audio_message(message)
            return
        if msg_type == "swipe":
            self._handle_swipe_message(message)

    def _handle_imu_message(self, message: dict) -> None:
        address = str(message.get("address", ""))
        if not address:
            return
        acc_raw = message.get("acc")
        gyr_raw = message.get("gyr")
        if not isinstance(acc_raw, list) or not isinstance(gyr_raw, list):
            return
        if len(acc_raw) != 3 or len(gyr_raw) != 3:
            return

        timestamp = self._plot_timestamp()
        acc = np.array(acc_raw, dtype=np.float32)
        gyr = np.array(gyr_raw, dtype=np.float32)
        self.imu_message_count += 1
        self._refresh_stream_counter()

        if address not in self.buffers:
            self.buffers[address] = RingBuffer()
        self.buffers[address].append(timestamp, acc, gyr)

    def _handle_audio_message(self, message: dict) -> None:
        address = str(message.get("address", ""))
        if not address:
            return

        audio_type = int(message.get("audio_type", -1))
        if audio_type not in (0, 1):
            return

        data_b64 = message.get("data_b64", "")
        if not isinstance(data_b64, str) or not data_b64:
            return

        try:
            raw_audio = base64.b64decode(data_b64)
        except Exception:
            return

        timestamp = self._plot_timestamp()
        decoder = self.audio_decoders.setdefault(address, AdpcmDecoder())
        pcm = decoder.decode_packet(raw_audio)
        if not pcm:
            return
        self.audio_message_count += 1
        self._refresh_stream_counter()

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return

        normalized = samples / 32768.0
        rms = float(np.sqrt(np.mean(normalized * normalized)))
        peak = float(np.max(np.abs(normalized)))

        if address not in self.audio_buffers:
            self.audio_buffers[address] = AudioLevelBuffer()
        self.audio_buffers[address].append(timestamp, rms, peak)

    def _handle_swipe_message(self, message: dict) -> None:
        address = str(message.get("address", ""))
        ring_name = str(message.get("name", "")).strip() or address or "Unknown"
        event = str(message.get("event", "")).strip() or "unknown"
        confidence = float(message.get("confidence", 0.0) or 0.0)
        source = str(message.get("source", "")).strip()
        total_count = message.get("total_count")
        count_text = "" if total_count is None else f" | total={int(total_count)}"
        source_text = "" if not source else f" | source={source}"
        timestamp = self._resolve_event_timestamp(
            float(message.get("host_timestamp", 0.0) or 0.0)
        )
        time_text = time.strftime("%H:%M:%S", time.localtime(timestamp))
        self.swipe_log.insertItem(
            0,
            f"{time_text} | {ring_name} | {event} | confidence={confidence:.3f}{source_text}{count_text}",
        )
        while self.swipe_log.count() > self._max_swipe_log_entries:
            self.swipe_log.takeItem(self.swipe_log.count() - 1)

    def _refresh_stream_counter(self) -> None:
        self.stream_counter_label.setText(
            f"流消息：IMU {self.imu_message_count} | AUDIO {self.audio_message_count}"
        )
        if (
            self._subscription_applied
            and self.cards_scroll.isVisible()
            and not self.cards
            and (self.imu_message_count > 0 or self.audio_message_count > 0)
        ):
            self.cards_hint.setText("正在收到数据，但当前没有可见卡片。请确认已勾选对应戒指并点击“应用订阅”。")
            self.cards_hint.setVisible(True)

    def _plot_timestamp(self) -> float:
        return time.time()

    def _resolve_event_timestamp(self, host_ts: float) -> float:
        now = time.time()
        cpu_time = time.perf_counter()
        host_ts = host_ts - cpu_time + now
        if not np.isfinite(host_ts) or host_ts <= 0:
            return now
        # Plot windows are based on wall-clock seconds; discard obviously invalid epochs.
        if host_ts < 946684800 or abs(host_ts - now) > 86400:
            return now
        return host_ts

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> None:
        _ = rings
        _ = removed_addresses
        _ = changed_connection_addresses
        self._refresh_cards()

    def _selected_addresses(self) -> list[str]:
        return self.selected_ring_addresses()

    def _effective_addresses(self) -> list[str]:
        return self._selected_addresses()

    def _refresh_cards(self) -> None:
        if not self._subscription_applied:
            self.cards_hint.setVisible(True)
            self.cards_scroll.setVisible(False)
            return

        self.cards_hint.setVisible(False)
        self.cards_scroll.setVisible(True)
        target = self._effective_addresses()

        for address in list(self.cards.keys()):
            if address not in target:
                card = self.cards.pop(address)
                card.setParent(None)

        for address in target:
            if not address:
                continue
            if address not in self.buffers:
                self.buffers[address] = RingBuffer()
            if address not in self.audio_buffers:
                self.audio_buffers[address] = AudioLevelBuffer()
            if address in self.cards:
                continue

            ring_meta = self.ring_info.get(address, {})
            ring_name = str(ring_meta.get("name", "Unknown"))
            card = RingPlotCard(
                address=address,
                ring_name=ring_name,
                buffer_ref=self.buffers[address],
                audio_buffer_ref=self.audio_buffers[address],
            )
            card.plot.set_window_seconds(float(self.window_slider.value()))
            card.plot.set_scale_mode(
                auto_scale=self.auto_scale_box.isChecked(),
                acc_scale=float(self.acc_scale_input.value()),
                gyr_scale=float(self.gyr_scale_input.value()),
            )
            self.cards[address] = card
            self.cards_layout.addWidget(card)

        self.empty_hint.setVisible(len(target) == 0)

    def apply_subscription(self) -> None:
        self._subscription_applied = True
        self.send_subscription(["imu", "audio", "swipe"], empty_behavior="none")
        self._refresh_cards()

    def refresh_plots(self) -> None:
        for card in self.cards.values():
            card.plot.update()

    def closeEvent(self, event):
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: ImuVisualizerWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
