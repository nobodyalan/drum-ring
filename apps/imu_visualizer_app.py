import argparse
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
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from common.stream_window import RingStreamWindowBase
from common.ui_style import APP_STYLE_SHEET

STYLE_SHEET = APP_STYLE_SHEET

ACC_AXIS_COLORS = [QColor("#E53E3E"), QColor("#F59E0B"), QColor("#10B981")]
GYR_AXIS_COLORS = [QColor("#2563EB"), QColor("#7C3AED"), QColor("#EC4899")]
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


class RingMiniPlotWidget(QWidget):
    def __init__(self, address: str, ring_name: str, buffer_ref: RingBuffer):
        super().__init__()
        self.address = address
        self.ring_name = ring_name
        self.buffer_ref = buffer_ref
        self.window_seconds = 5.0
        self.auto_scale = True
        self.acc_scale = 20.0
        self.gyr_scale = 1200.0
        self.setMinimumHeight(260)

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

        now = time.time()
        since_ts = now - self.window_seconds
        ts, acc, gyr = self.buffer_ref.recent(since_ts)

        margin = 10
        top_space = 4
        panel_h = (self.height() - top_space - margin * 3) // 2
        acc_rect = (margin, top_space + margin, self.width() - margin * 2, panel_h)
        gyr_rect = (
            margin,
            top_space + margin * 2 + panel_h,
            self.width() - margin * 2,
            panel_h,
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

        if ts.size < 2 or values.shape[0] < 2:
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
    def __init__(self, address: str, ring_name: str, buffer_ref: RingBuffer):
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
        unit_row.addWidget(unit_acc)
        unit_row.addWidget(unit_gyro)
        unit_row.addStretch()
        layout.addLayout(unit_row)

        self.plot = RingMiniPlotWidget(address=address, ring_name=ring_name, buffer_ref=buffer_ref)
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
        self.setWindowTitle("高性能 IMU 可视化")
        self.resize(1240, 800)

        self.buffers: dict[str, RingBuffer] = {}
        self.cards: dict[str, RingPlotCard] = {}
        self._subscription_applied = False

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")

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

        title = QLabel("多戒指 IMU 实时可视化")
        title.setProperty("class", "title")
        subtitle = QLabel("多戒指模式将生成多个小窗口卡片，每个卡片单独渲染一个戒指。")
        subtitle.setProperty("class", "sub-text")
        subtitle.setWordWrap(True)

        top_layout.addWidget(title)
        top_layout.addWidget(subtitle)
        top_layout.addWidget(self.status_label)

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
        root.addWidget(self.cards_hint)
        root.addWidget(self.cards_scroll)
        self.cards_scroll.setVisible(False)

    def _bind_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.window_slider.valueChanged.connect(self.on_window_changed)
        self.auto_scale_box.stateChanged.connect(self.on_scale_controls_changed)
        self.acc_scale_input.valueChanged.connect(self.on_scale_controls_changed)
        self.gyr_scale_input.valueChanged.connect(self.on_scale_controls_changed)

        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
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
        if message.get("type") != "imu":
            return

        address = str(message.get("address", ""))
        if not address:
            return
        acc_raw = message.get("acc")
        gyr_raw = message.get("gyr")
        host_ts = float(message.get("host_timestamp", 0.0) or 0.0)
        timestamp = host_ts if host_ts > 0 else time.time()
        if not isinstance(acc_raw, list) or not isinstance(gyr_raw, list):
            return
        if len(acc_raw) != 3 or len(gyr_raw) != 3:
            return

        acc = np.array(acc_raw, dtype=np.float32)
        gyr = np.array(gyr_raw, dtype=np.float32)

        if address not in self.buffers:
            self.buffers[address] = RingBuffer()
        self.buffers[address].append(timestamp, acc, gyr)

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
            if address in self.cards:
                continue

            ring_meta = self.ring_info.get(address, {})
            ring_name = str(ring_meta.get("name", "Unknown"))
            card = RingPlotCard(address=address, ring_name=ring_name, buffer_ref=self.buffers[address])
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
        self.send_subscription(["imu"], empty_behavior="none")
        self._refresh_cards()

    def refresh_plots(self) -> None:
        for card in self.cards.values():
            card.plot.update()

    def closeEvent(self, event):
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
    window = ImuVisualizerWindow(host=args.host, port=args.port)
    window.show()
    sys.exit(app.exec())
