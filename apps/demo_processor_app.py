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
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET

STYLE_SHEET = APP_STYLE_SHEET


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


class DemoProcessorWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("示范应用 - 数据处理模板")
        self.resize(820, 560)

        self.processor: BaseImuProcessor = DemoImuProcessor()

        self.status_text = QLabel("数据流状态：未连接")
        self.status_text.setObjectName("statusTag")
        self.metrics_text = QLabel("包数: 0\nEMA(acc): --\nEMA(gyr): --")

        self._build_ui()
        self.set_stream_status_label(self.status_text)
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

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择要订阅的戒指（不勾选表示接收全部）",
        )

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
        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def handle_stream_message(self, message: dict) -> None:
        if message.get("type") != "imu":
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

    def apply_subscription(self) -> None:
        self.send_subscription(["imu"])

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
        lambda: DemoProcessorWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
