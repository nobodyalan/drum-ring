from __future__ import annotations

"""Standalone touchpad app assembled from the shared touchpad runtime."""

import argparse
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET

from .runtime import (
    HAS_TOUCHPAD_CURSOR_CONTROL,
    TOUCHPAD_CURSOR_CONTROL_ERROR,
    TouchpadConfig,
    TouchpadWorker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TouchpadAppWindow(RingStreamWindowBase):
    """Thin UI shell around the shared touchpad worker."""

    def __init__(self, host: str, port: int, config: TouchpadConfig):
        super().__init__(host=host, port=port)
        self.setWindowTitle("触控板应用")
        self.resize(860, 560)

        self.worker_error = ""
        self.touchpad_worker: TouchpadWorker | None = None
        if not HAS_TOUCHPAD_CURSOR_CONTROL:
            self.worker_error = f"桌面控制不可用：{TOUCHPAD_CURSOR_CONTROL_ERROR}"
        else:
            try:
                self.touchpad_worker = TouchpadWorker(config)
                self.touchpad_worker.engine.set_cursor_control_enabled(True)
            except Exception as exc:
                self.worker_error = str(exc)

        self.status_text = QLabel("数据流状态：未连接")
        self.status_text.setObjectName("statusTag")
        self.runtime_text = QLabel(self.runtime_hint_text())
        self.runtime_text.setWordWrap(True)
        self.metrics_text = QLabel("包数: 0\n队列: 0\n活动戒指: 无")

        self._build_ui()
        self.set_stream_status_label(self.status_text)
        self._bind_signals()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_metrics)
        self.refresh_timer.start(200)

        self.client.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        panel = QFrame()
        panel.setProperty("class", "panel")
        panel_layout = QVBoxLayout(panel)

        title = QLabel("戒指触控板")
        title.setProperty("class", "title")
        desc = QLabel(
            "独立应用只负责组装界面和订阅，手势识别、位移估计、点击判定都在 touchpad feature 中复用。"
        )
        desc.setProperty("class", "sub-text")
        desc.setWordWrap(True)

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.status_text)
        panel_layout.addWidget(self.runtime_text)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择要作为触控板的戒指（不勾选表示订阅全部）",
        )

        row = QHBoxLayout()
        self.btn_apply = QPushButton("应用订阅")
        self.btn_apply.setProperty("class", "btn-primary")
        row.addWidget(self.btn_apply)
        row.addStretch()

        panel_layout.addLayout(row)
        panel_layout.addWidget(self.metrics_text)
        root.addWidget(panel)

    def _bind_signals(self) -> None:
        self.btn_apply.clicked.connect(self.apply_subscription)
        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def handle_stream_message(self, message: dict) -> None:
        if self.touchpad_worker is None or message.get("type") != "imu":
            return

        address = str(message.get("address", ""))
        acc_raw = message.get("acc")
        gyr_raw = message.get("gyr")
        if not address or not isinstance(acc_raw, list) or not isinstance(gyr_raw, list):
            return
        if len(acc_raw) != 3 or len(gyr_raw) != 3:
            return

        data = np.asarray(
            [
                float(acc_raw[0]),
                float(acc_raw[1]),
                float(acc_raw[2]),
                float(gyr_raw[0]),
                float(gyr_raw[1]),
                float(gyr_raw[2]),
            ],
            dtype=np.float32,
        )
        self.touchpad_worker.enqueue(address, data)

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> None:
        _ = rings
        if self.touchpad_worker is None:
            return
        reset_addresses = removed_addresses + changed_connection_addresses
        if reset_addresses:
            self.touchpad_worker.clear_states(reset_addresses)

    def apply_subscription(self) -> None:
        self.send_subscription(["imu"])

    def runtime_hint_text(self) -> str:
        if self.worker_error:
            return f"初始化失败：{self.worker_error}"
        return "逻辑保持与旧版一致：Down 手势进入移动态，Up 手势退出；在短时间内 Down 后 Up 会触发左键点击。"

    def refresh_metrics(self) -> None:
        if self.touchpad_worker is None:
            self.runtime_text.setText(self.runtime_hint_text())
            return

        snapshot = self.touchpad_worker.snapshot()
        active_rings = snapshot["active_rings"]
        active_text = "、".join(active_rings) if active_rings else "无"
        self.runtime_text.setText(
            "\n".join(
                [
                    self.runtime_hint_text(),
                    f"最近状态：{snapshot['last_status']}",
                ]
            )
        )
        self.metrics_text.setText(
            "\n".join(
                [
                    f"包数: {int(snapshot['packet_count'])}",
                    f"队列: {int(snapshot['pending_count'])}",
                    f"活动戒指: {active_text}",
                    f"已初始化戒指: {int(snapshot['ring_count'])}",
                ]
            )
        )

    def closeEvent(self, event) -> None:
        self.refresh_timer.stop()
        self.client.stop()
        self.client.wait(1500)
        if self.touchpad_worker is not None:
            self.touchpad_worker.stop()
        event.accept()


def create_touchpad_parser() -> argparse.ArgumentParser:
    parser = create_base_parser()
    parser.add_argument("--gesture-ckpt", default=str(PROJECT_ROOT / "checkpoint" / "tap-94.pth"))
    parser.add_argument("--motion-ckpt", default=str(PROJECT_ROOT / "checkpoint" / "tcnlstm.pt"))
    return parser


def run_touchpad_app() -> int:
    args = create_touchpad_parser().parse_args()
    config = TouchpadConfig(
        gesture_ckpt=Path(args.gesture_ckpt),
        motion_ckpt=Path(args.motion_ckpt),
    )
    return run_window(
        lambda: TouchpadAppWindow(host=args.host, port=args.port, config=config),
        style_sheet=APP_STYLE_SHEET,
    )
