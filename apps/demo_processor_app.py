import argparse
import math
import sys
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from common.stream_client import JsonLineStreamClient

STYLE_SHEET = """
QMainWindow { background-color: #f2f3f5; }
QFrame.panel {
    background-color: #ffffff;
    border: 1px solid #dcdcdc;
    border-radius: 8px;
}
QPushButton {
    border-radius: 4px; padding: 6px 12px; border: none; font-weight: bold; color: white;
}
QPushButton.btn-primary { background-color: #3182ce; }
QPushButton.btn-primary:hover { background-color: #2b6cb0; }
QLabel { font-size: 13px; color: #2d3748; }
QLabel.title { font-size: 14px; font-weight: bold; }
QLabel.sub-text { font-size: 11px; color: #718096; }
"""


@dataclass
class ProcessState:
    packet_count: int = 0
    ema_acc_norm: float = 0.0
    ema_gyr_norm: float = 0.0


class BaseImuProcessor:
    """示范处理器基类，后续可在子类中实现业务逻辑。"""

    def on_sample(self, address: str, acc: list[float], gyr: list[float]) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, float]:
        raise NotImplementedError


class DemoImuProcessor(BaseImuProcessor):
    def __init__(self, alpha: float = 0.05):
        self._alpha = alpha
        self._state = ProcessState()

    def on_sample(self, address: str, acc: list[float], gyr: list[float]) -> None:
        _ = address
        acc_norm = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
        gyr_norm = math.sqrt(gyr[0] ** 2 + gyr[1] ** 2 + gyr[2] ** 2)

        self._state.packet_count += 1
        if self._state.packet_count == 1:
            self._state.ema_acc_norm = acc_norm
            self._state.ema_gyr_norm = gyr_norm
            return

        self._state.ema_acc_norm = (
            self._alpha * acc_norm + (1.0 - self._alpha) * self._state.ema_acc_norm
        )
        self._state.ema_gyr_norm = (
            self._alpha * gyr_norm + (1.0 - self._alpha) * self._state.ema_gyr_norm
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "packet_count": float(self._state.packet_count),
            "ema_acc_norm": self._state.ema_acc_norm,
            "ema_gyr_norm": self._state.ema_gyr_norm,
        }


class DemoProcessorWindow(QMainWindow):
    def __init__(self, host: str, port: int):
        super().__init__()
        self.setWindowTitle("示范应用 - 数据处理模板")
        self.resize(820, 560)

        self.client = JsonLineStreamClient(host=host, port=port)
        self.processor: BaseImuProcessor = DemoImuProcessor()
        self.ring_boxes: dict[str, QCheckBox] = {}

        self.status_text = QLabel("数据流状态：未连接")
        self.metrics_text = QLabel("包数: 0\nEMA(acc): --\nEMA(gyr): --")

        self._build_ui()
        self._bind_signals()
        self.client.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        panel = QFrame()
        panel.setProperty("class", "panel")
        panel_layout = QVBoxLayout(panel)

        title = QLabel("示范处理器")
        title.setProperty("class", "title")
        desc = QLabel("该应用仅演示独立进程接收戒指数据流与处理接口设计。")
        desc.setProperty("class", "sub-text")

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.status_text)

        ring_title = QLabel("选择要订阅的戒指（不勾选表示接收全部）")
        ring_title.setProperty("class", "sub-text")
        panel_layout.addWidget(ring_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.ring_layout = QVBoxLayout(scroll_content)
        self.ring_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(scroll_content)
        panel_layout.addWidget(scroll)

        control_row = QHBoxLayout()
        self.btn_apply = QPushButton("应用订阅")
        self.btn_apply.setProperty("class", "btn-primary")
        control_row.addWidget(self.btn_apply)
        control_row.addStretch()
        panel_layout.addLayout(control_row)

        panel_layout.addWidget(self.metrics_text)

        root.addWidget(panel)

    def _bind_signals(self) -> None:
        self.btn_apply.clicked.connect(self.apply_subscription)
        self.client.sig_connected.connect(self.on_connected_changed)
        self.client.sig_message.connect(self.on_message)
        self.client.sig_error.connect(self.on_error)

    def on_connected_changed(self, connected: bool) -> None:
        self.status_text.setText(f"数据流状态：{'已连接' if connected else '未连接'}")
        if connected:
            self.client.send_command({"cmd": "status"})
            self.apply_subscription()

    def on_error(self, message: str) -> None:
        self.status_text.setText(f"数据流状态：异常（{message}）")

    def on_message(self, message: dict) -> None:
        msg_type = message.get("type")
        if msg_type == "status":
            self._sync_rings(message.get("rings", []))
            return

        if msg_type != "imu":
            return

        acc = message.get("acc")
        gyr = message.get("gyr")
        address = message.get("address", "")
        if not isinstance(acc, list) or not isinstance(gyr, list) or len(acc) != 3 or len(gyr) != 3:
            return

        self.processor.on_sample(address, [float(v) for v in acc], [float(v) for v in gyr])
        snapshot = self.processor.snapshot()
        self.metrics_text.setText(
            "\n".join(
                [
                    f"包数: {int(snapshot['packet_count'])}",
                    f"EMA(acc): {snapshot['ema_acc_norm']:.4f}",
                    f"EMA(gyr): {snapshot['ema_gyr_norm']:.4f}",
                ]
            )
        )

    def _sync_rings(self, rings: list[dict]) -> None:
        ring_map = {str(item.get("address", "")): item for item in rings}

        for address in list(self.ring_boxes.keys()):
            if address not in ring_map:
                box = self.ring_boxes.pop(address)
                box.setParent(None)

        for address, ring in ring_map.items():
            if not address:
                continue
            if address in self.ring_boxes:
                box = self.ring_boxes[address]
            else:
                box = QCheckBox()
                box.setChecked(False)
                self.ring_boxes[address] = box
                self.ring_layout.addWidget(box)

            ring_name = ring.get("name", "Unknown")
            connected = bool(ring.get("connected", False))
            box.setText(f"{ring_name} | {address} | {'在线' if connected else '离线'}")

    def apply_subscription(self) -> None:
        selected = [addr for addr, box in self.ring_boxes.items() if box.isChecked()]
        self.client.send_command({"cmd": "subscribe", "rings": selected})

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
    win = DemoProcessorWindow(host=args.host, port=args.port)
    win.show()
    sys.exit(app.exec())
