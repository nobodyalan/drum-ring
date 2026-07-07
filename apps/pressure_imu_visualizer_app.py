import argparse
import re
import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QIODevice, QTimer, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
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
EVENT_COLORS = {
    "FALLING": QColor("#DC2626"),
    "RISING": QColor("#059669"),
}
PRESSURE_LINE_RE = re.compile(
    r"Sensor\s+(?P<sensor>\d+)\s+"
    r"(?P<edge>FALLING|RISING)\s+"
    r"count=(?P<count>-?\d+)\s+"
    r"raw=(?P<raw>-?\d+)\s+"
    r"delta=(?P<delta>-?\d+)"
)


@dataclass
class PressureEvent:
    timestamp: float
    sensor: int
    edge: str
    count: int
    raw: int
    delta: int


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


class PressureEventBuffer:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.events: list[PressureEvent] = []

    def append(self, event: PressureEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.capacity:
            self.events = self.events[-self.capacity :]

    def recent(self, since_ts: float) -> list[PressureEvent]:
        return [event for event in self.events if event.timestamp >= since_ts]

    def count(self) -> int:
        return len(self.events)


class PressureSerialReader:
    def __init__(
        self,
        serial: QSerialPort,
        event_buffer: PressureEventBuffer,
        on_event,
        on_status,
    ):
        self.serial = serial
        self.event_buffer = event_buffer
        self.on_event = on_event
        self.on_status = on_status
        self._line_buffer = bytearray()
        self.serial.readyRead.connect(self._read_available)
        self.serial.errorOccurred.connect(self._handle_error)

    def _read_available(self) -> None:
        chunk = bytes(self.serial.readAll())
        if not chunk:
            return
        self._line_buffer.extend(chunk)
        while b"\n" in self._line_buffer:
            raw_line, _, rest = self._line_buffer.partition(b"\n")
            self._line_buffer = bytearray(rest)
            self._handle_line(raw_line.decode("utf-8", errors="ignore").strip())

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        match = PRESSURE_LINE_RE.search(line)
        if match is None:
            if "Starting rising+falling edge detection" in line:
                self.on_status("压力串口：已连接，检测程序已启动")
            return

        event = PressureEvent(
            timestamp=time.time(),
            sensor=int(match.group("sensor")),
            edge=match.group("edge"),
            count=int(match.group("count")),
            raw=int(match.group("raw")),
            delta=int(match.group("delta")),
        )
        self.event_buffer.append(event)
        self.on_event(event)

    def _handle_error(self, error) -> None:
        if error == QSerialPort.SerialPortError.NoError:
            return
        self.on_status(f"压力串口：异常（{self.serial.errorString()}）")


class PressureImuPlotWidget(QWidget):
    def __init__(
        self,
        address: str,
        ring_name: str,
        buffer_ref: RingBuffer,
        event_buffer_ref: PressureEventBuffer,
    ):
        super().__init__()
        self.address = address
        self.ring_name = ring_name
        self.buffer_ref = buffer_ref
        self.event_buffer_ref = event_buffer_ref
        self.window_seconds = 5.0
        self.auto_scale = True
        self.acc_scale = 100.0
        self.gyr_scale = 20.0
        self.setMinimumHeight(300)

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

        latest_ts = max(self.buffer_ref.latest_timestamp(), time.time())
        since_ts = latest_ts - self.window_seconds
        ts, acc, gyr = self.buffer_ref.recent(since_ts)
        pressure_events = self.event_buffer_ref.recent(since_ts)

        margin = 10
        top_space = 4
        total_h = self.height() - top_space - margin * 3
        panel_h = max(90, total_h // 2)
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
            "ACC",
            ts,
            acc,
            self.acc_scale,
            ACC_AXIS_COLORS,
            pressure_events,
            since_ts,
            latest_ts,
        )
        self._draw_panel(
            painter,
            gyr_rect,
            "GYR",
            ts,
            gyr,
            self.gyr_scale,
            GYR_AXIS_COLORS,
            pressure_events,
            since_ts,
            latest_ts,
        )

    def _draw_panel(
        self,
        painter: QPainter,
        rect: tuple[int, int, int, int],
        label: str,
        ts: np.ndarray,
        values: np.ndarray,
        fixed_scale: float,
        axis_colors: list[QColor],
        pressure_events: list[PressureEvent],
        ts_min: float,
        ts_max: float,
    ) -> None:
        x, y, w, h = rect
        painter.setPen(QColor("#e2e8f0"))
        painter.drawRect(x, y, w, h)
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(x + 8, y + 16, label)

        axis_w = 46
        plot_x = x + axis_w
        plot_w = max(10, w - axis_w - 4)
        painter.drawLine(plot_x, y, plot_x, y + h)
        self._draw_pressure_events(
            painter,
            x=plot_x,
            y=y,
            w=plot_w,
            h=h,
            ts_min=ts_min,
            ts_max=ts_max,
            events=pressure_events,
        )

        finite_mask = np.isfinite(ts)
        if values.ndim == 2 and values.shape[1] == 3:
            finite_mask &= np.all(np.isfinite(values), axis=1)
        ts = ts[finite_mask]
        values = values[finite_mask]

        if ts.size < 2 or values.shape[0] < 2:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(plot_x + 12, y + h // 2 + 4, "等待 IMU 数据")
            return

        y_min, y_max = self._range(values, fixed_scale)
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
                ts_min=ts_min,
                ts_max=ts_max,
                y_min=y_min,
                y_max=y_max,
            )

    def _draw_pressure_events(
        self,
        painter: QPainter,
        *,
        x: int,
        y: int,
        w: int,
        h: int,
        ts_min: float,
        ts_max: float,
        events: list[PressureEvent],
    ) -> None:
        denom_x = max(1e-6, ts_max - ts_min)
        for event in events:
            px = int(x + ((event.timestamp - ts_min) / denom_x) * w)
            if px < x or px > x + w:
                continue
            color = EVENT_COLORS.get(event.edge, QColor("#475569"))
            painter.setPen(QPen(color, 2))
            painter.drawLine(px, y + 1, px, y + h - 1)
            painter.setPen(color)
            marker = "F" if event.edge == "FALLING" else "R"
            painter.drawText(px + 3, y + 14, marker)

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


class PressureImuCard(QFrame):
    def __init__(
        self,
        address: str,
        ring_name: str,
        buffer_ref: RingBuffer,
        event_buffer_ref: PressureEventBuffer,
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
        legend_row.addSpacing(8)
        legend_row.addWidget(self._legend_label("FALLING", EVENT_COLORS["FALLING"]))
        legend_row.addWidget(self._legend_label("RISING", EVENT_COLORS["RISING"]))
        legend_row.addStretch()
        layout.addLayout(legend_row)

        self.plot = PressureImuPlotWidget(
            address=address,
            ring_name=ring_name,
            buffer_ref=buffer_ref,
            event_buffer_ref=event_buffer_ref,
        )
        layout.addWidget(self.plot)

    def _legend_label(self, text: str, color: QColor) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {color.name()}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        return label


class PressureImuVisualizerWindow(RingStreamWindowBase):
    def __init__(
        self,
        host: str,
        port: int,
        serial_port: str,
        baudrate: int,
        auto_serial: bool,
    ):
        super().__init__(host=host, port=port)
        self.setWindowTitle("戒指 IMU + 压力边沿同步可视化")
        self.resize(1320, 900)

        self.serial_port_name = serial_port
        self.baudrate = baudrate
        self.auto_serial = auto_serial
        self.serial = QSerialPort(self)
        self.event_buffer = PressureEventBuffer()
        self.serial_reader = PressureSerialReader(
            serial=self.serial,
            event_buffer=self.event_buffer,
            on_event=self._handle_pressure_event,
            on_status=self._set_serial_status,
        )

        self.buffers: dict[str, RingBuffer] = {}
        self.cards: dict[str, PressureImuCard] = {}
        self._subscription_applied = False
        self.imu_message_count = 0
        self.pressure_event_count = 0

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")
        self.serial_status_label = QLabel("压力串口：未连接")
        self.serial_status_label.setObjectName("statusTag")
        self.counter_label = QLabel("流消息：IMU 0 | 压力事件 0")
        self.counter_label.setProperty("class", "sub-text")

        self._build_ui()
        self.set_stream_status_label(self.status_label)
        self._bind_signals()
        self.on_scale_controls_changed()
        self.refresh_serial_ports()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.refresh_plots)
        self.timer.start()

        self.client.start()
        if self.serial_port_name:
            self._select_serial_port(self.serial_port_name)
        if self.auto_serial:
            QTimer.singleShot(100, self.connect_selected_serial_if_available)

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

        title = QLabel("戒指 IMU + 压力边沿同步可视化")
        title.setProperty("class", "title")
        subtitle = QLabel(
            "订阅戒指 IMU 数据，并从 Arduino 串口读取 FALLING / RISING 压力边沿；"
            "两类数据按本机接收时间绘制在同一个滑动时间窗中。"
        )
        subtitle.setProperty("class", "sub-text")
        subtitle.setWordWrap(True)
        top_layout.addWidget(title)
        top_layout.addWidget(subtitle)
        top_layout.addWidget(self.status_label)
        top_layout.addWidget(self.serial_status_label)
        top_layout.addWidget(self.counter_label)

        self.add_ring_selection_panel(
            top_layout,
            title_text="选择订阅戒指",
            max_height=130,
        )
        self.ring_layout.setSpacing(8)
        self.ring_layout.setContentsMargins(8, 8, 8, 8)

        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(8)

        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        controls.addWidget(self.apply_btn, 0, 0)

        controls.addWidget(QLabel("串口"), 0, 1)
        self.serial_combo = QComboBox()
        self.serial_combo.setMinimumWidth(260)
        controls.addWidget(self.serial_combo, 0, 2)

        self.refresh_serial_btn = QPushButton("刷新")
        self.refresh_serial_btn.setProperty("class", "btn-gray")
        controls.addWidget(self.refresh_serial_btn, 0, 3)

        self.serial_connect_btn = QPushButton("连接串口")
        self.serial_connect_btn.setProperty("class", "btn-success")
        controls.addWidget(self.serial_connect_btn, 0, 4)

        controls.addWidget(QLabel("窗口(秒)"), 1, 1)
        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setRange(2, 20)
        self.window_slider.setValue(5)
        self.window_slider.setMinimumWidth(220)
        controls.addWidget(self.window_slider, 1, 2)
        self.window_label = QLabel("5s")
        self.window_label.setObjectName("valuePill")
        controls.addWidget(self.window_label, 1, 3)

        controls.addWidget(QLabel("ACC scale"), 2, 1)
        self.acc_scale_input = QDoubleSpinBox()
        self.acc_scale_input.setRange(1.0, 200.0)
        self.acc_scale_input.setSingleStep(1.0)
        self.acc_scale_input.setDecimals(1)
        self.acc_scale_input.setValue(100.0)
        self.acc_scale_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.acc_scale_input.setFixedWidth(120)
        controls.addWidget(self.acc_scale_input, 2, 2)

        controls.addWidget(QLabel("GYR scale"), 2, 3)
        self.gyr_scale_input = QDoubleSpinBox()
        self.gyr_scale_input.setRange(0.01, 50.0)
        self.gyr_scale_input.setSingleStep(0.1)
        self.gyr_scale_input.setDecimals(2)
        self.gyr_scale_input.setValue(20.0)
        self.gyr_scale_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.gyr_scale_input.setFixedWidth(120)
        controls.addWidget(self.gyr_scale_input, 2, 4)

        self.auto_scale_btn = QPushButton("自动 Y 轴")
        self.auto_scale_btn.setCheckable(True)
        self.auto_scale_btn.setChecked(True)
        controls.addWidget(self.auto_scale_btn, 2, 0)
        controls.setColumnStretch(5, 1)
        top_layout.addLayout(controls)
        root.addWidget(top_panel)

        self.cards_hint = QLabel("请先勾选戒指并点击“应用订阅”。")
        self.cards_hint.setProperty("class", "sub-text")
        self.cards_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_hint.setWordWrap(True)
        root.addWidget(self.cards_hint)

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setStyleSheet("border: none;")
        self.cards_content = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_content)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cards_layout.setSpacing(12)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_scroll.setWidget(self.cards_content)
        self.cards_scroll.setVisible(False)
        root.addWidget(self.cards_scroll)

        root.addWidget(QLabel("压力事件"))
        self.event_log = QListWidget()
        self.event_log.setMaximumHeight(160)
        self.event_log.setAlternatingRowColors(True)
        root.addWidget(self.event_log)

    def _bind_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.refresh_serial_btn.clicked.connect(self.refresh_serial_ports)
        self.serial_connect_btn.clicked.connect(self.toggle_serial_connection)
        self.window_slider.valueChanged.connect(self.on_window_changed)
        self.auto_scale_btn.toggled.connect(self.on_scale_controls_changed)
        self.acc_scale_input.valueChanged.connect(self.on_scale_controls_changed)
        self.gyr_scale_input.valueChanged.connect(self.on_scale_controls_changed)
        self.bind_stream_client_signals()

    def refresh_serial_ports(self) -> None:
        previous = self.serial_combo.currentData()
        self.serial_combo.clear()
        for info in QSerialPortInfo.availablePorts():
            port_name = info.systemLocation()
            description = info.description() or info.portName()
            self.serial_combo.addItem(f"{port_name} | {description}", port_name)

        if self.serial_combo.count() == 0:
            self.serial_combo.addItem("未发现串口", "")
            return

        preferred = self.serial_port_name or str(previous or "")
        if preferred:
            self._select_serial_port(preferred)
            return

        for index in range(self.serial_combo.count()):
            port_name = str(self.serial_combo.itemData(index) or "")
            if "usbserial" in port_name.lower() or "wch" in port_name.lower():
                self.serial_combo.setCurrentIndex(index)
                return

    def _select_serial_port(self, port_name: str) -> None:
        for index in range(self.serial_combo.count()):
            if self.serial_combo.itemData(index) == port_name:
                self.serial_combo.setCurrentIndex(index)
                return

    def toggle_serial_connection(self) -> None:
        if self.serial.isOpen():
            self.serial.close()
            self.serial_connect_btn.setText("连接串口")
            self.serial_connect_btn.setProperty("class", "btn-success")
            self.serial_connect_btn.style().unpolish(self.serial_connect_btn)
            self.serial_connect_btn.style().polish(self.serial_connect_btn)
            self._set_serial_status("压力串口：已断开")
            return

        port_name = str(self.serial_combo.currentData() or "")
        if not port_name:
            self._set_serial_status("压力串口：未选择有效串口")
            return

        self.serial.setPortName(port_name)
        self.serial.setBaudRate(self.baudrate)
        self.serial.setDataBits(QSerialPort.DataBits.Data8)
        self.serial.setParity(QSerialPort.Parity.NoParity)
        self.serial.setStopBits(QSerialPort.StopBits.OneStop)
        self.serial.setFlowControl(QSerialPort.FlowControl.NoFlowControl)
        if not self.serial.open(QIODevice.OpenModeFlag.ReadOnly):
            self._set_serial_status(f"压力串口：连接失败（{self.serial.errorString()}）")
            return

        self.serial_port_name = port_name
        self.serial_connect_btn.setText("断开串口")
        self.serial_connect_btn.setProperty("class", "btn-danger")
        self.serial_connect_btn.style().unpolish(self.serial_connect_btn)
        self.serial_connect_btn.style().polish(self.serial_connect_btn)
        self._set_serial_status(f"压力串口：已连接 {port_name} @ {self.baudrate}")

    def connect_selected_serial_if_available(self) -> None:
        if self.serial.isOpen():
            return
        port_name = str(self.serial_combo.currentData() or "")
        if not port_name:
            return
        if self.serial_port_name or "usbserial" in port_name.lower() or "wch" in port_name.lower():
            self.toggle_serial_connection()

    def on_stream_connected(self) -> None:
        if self._subscription_applied:
            self.apply_subscription()
        self._refresh_cards()

    def on_window_changed(self, value: int) -> None:
        self.window_label.setText(f"{value}s")
        for card in self.cards.values():
            card.plot.set_window_seconds(float(value))

    def on_scale_controls_changed(self, _value=None) -> None:
        auto_mode = self.auto_scale_btn.isChecked()
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
        if msg_type != "imu":
            return

        address = str(message.get("address", ""))
        if not address:
            return
        acc_raw = message.get("acc")
        gyr_raw = message.get("gyr")
        if not isinstance(acc_raw, list) or not isinstance(gyr_raw, list):
            return
        if len(acc_raw) != 3 or len(gyr_raw) != 3:
            return

        if address not in self.buffers:
            self.buffers[address] = RingBuffer()
        self.buffers[address].append(
            time.time(),
            np.array(acc_raw, dtype=np.float32),
            np.array(gyr_raw, dtype=np.float32),
        )
        self.imu_message_count += 1
        self._refresh_counter()

    def _handle_pressure_event(self, event: PressureEvent) -> None:
        self.pressure_event_count += 1
        self._refresh_counter()
        time_text = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        self.event_log.insertItem(
            0,
            (
                f"{time_text} | Sensor {event.sensor} | {event.edge} | "
                f"count={event.count} raw={event.raw} delta={event.delta}"
            ),
        )
        while self.event_log.count() > 120:
            self.event_log.takeItem(self.event_log.count() - 1)

    def _set_serial_status(self, text: str) -> None:
        self.serial_status_label.setText(text)

    def _refresh_counter(self) -> None:
        self.counter_label.setText(
            f"流消息：IMU {self.imu_message_count} | 压力事件 {self.pressure_event_count}"
        )

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

    def _refresh_cards(self) -> None:
        if not self._subscription_applied:
            self.cards_hint.setVisible(True)
            self.cards_scroll.setVisible(False)
            return

        self.cards_hint.setVisible(False)
        self.cards_scroll.setVisible(True)
        target = self.selected_ring_addresses()

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
            card = PressureImuCard(
                address=address,
                ring_name=ring_name,
                buffer_ref=self.buffers[address],
                event_buffer_ref=self.event_buffer,
            )
            card.plot.set_window_seconds(float(self.window_slider.value()))
            card.plot.set_scale_mode(
                auto_scale=self.auto_scale_btn.isChecked(),
                acc_scale=float(self.acc_scale_input.value()),
                gyr_scale=float(self.gyr_scale_input.value()),
            )
            self.cards[address] = card
            self.cards_layout.addWidget(card)

        if len(target) == 0:
            self.cards_hint.setText("请至少勾选一个戒指并点击“应用订阅”。")
            self.cards_hint.setVisible(True)
        else:
            self.cards_hint.setVisible(False)

    def apply_subscription(self) -> None:
        self._subscription_applied = True
        self.send_subscription(["imu"], empty_behavior="none")
        self._refresh_cards()

    def refresh_plots(self) -> None:
        for card in self.cards.values():
            card.plot.update()

    def closeEvent(self, event):
        if self.serial.isOpen():
            self.serial.close()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser(description="Ring IMU + Arduino pressure edge visualizer")
    parser.add_argument("--serial-port", default="")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--no-auto-serial",
        action="store_true",
        help="Do not automatically open the selected USB serial port on startup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: PressureImuVisualizerWindow(
            host=args.host,
            port=args.port,
            serial_port=args.serial_port,
            baudrate=args.baudrate,
            auto_serial=not args.no_auto_serial,
        ),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
