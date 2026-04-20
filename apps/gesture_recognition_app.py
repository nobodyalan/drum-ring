import argparse
import time
from dataclasses import dataclass
from functools import partial

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from apps.features.gesture.pc_control import (
    HAS_PC_CONTROL,
    PC_CONTROL_ERROR,
    RingPCController,
)
from apps.features.gesture.runtime import (
    GestureConfig,
    GestureInferenceWorker,
    GestureSnapshot,
    load_gesture_config,
)
from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET
from apps.shared.constants import GESTURE_CODE_TO_NAME

STYLE_SHEET = APP_STYLE_SHEET


class RingGestureCard(QFrame):
    def __init__(self, address: str, ring_name: str):
        super().__init__()
        self.address = address
        self.ring_name = ring_name
        self.setProperty("class", "panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title = QLabel(f"{ring_name} | {address}")
        self.title.setProperty("class", "title")
        self.lbl_latest = QLabel("最新手势: -")
        self.lbl_count = QLabel("累计识别: 0")
        self.lbl_time = QLabel("最近触发: -")
        self.lbl_time.setProperty("class", "sub-text")

        layout.addWidget(self.title)
        layout.addWidget(self.lbl_latest)
        layout.addWidget(self.lbl_count)
        layout.addWidget(self.lbl_time)

    def update_state(self, snapshot: GestureSnapshot):
        code_text = str(snapshot.last_code) if snapshot.last_code >= 0 else "-"
        self.lbl_latest.setText(f"最新手势: {snapshot.last_name} ({code_text})")
        self.lbl_count.setText(f"累计识别: {snapshot.detected_count}")
        if snapshot.last_ts > 0:
            self.lbl_time.setText(
                f"最近触发: {time.strftime('%H:%M:%S', time.localtime(snapshot.last_ts))}"
            )
        else:
            self.lbl_time.setText("最近触发: -")


class GestureRecognitionWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("手势识别")
        self.resize(980, 700)

        self.cfg = load_gesture_config()
        self.worker = GestureInferenceWorker(self.cfg)

        self.cards: dict[str, RingGestureCard] = {}
        self._stream_connected = False
        self._worker_ready = False
        self._worker_error = ""
        self._subscription_applied = False
        self._pc_control_enabled = False
        self.pc_controller = RingPCController() if HAS_PC_CONTROL else None

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")
        self.total_count_label = QLabel("总识别次数: 0")
        self.rate_label = QLabel("接收: --Hz | 识别: --Hz | 丢弃: 0")
        self.rate_label.setProperty("class", "sub-text")
        self.total_count = 0

        self._build_ui()
        self.set_stream_status_label(self.status_label)
        self._bind_signals()
        self.client.add_message_hook(self._handle_pc_control_stream_message)
        self.worker.start()
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

        title = QLabel("手势识别可视化（OpenVINO）")
        title.setProperty("class", "title")
        desc = QLabel("多模型投票 + 滑窗判定。")
        desc.setProperty("class", "sub-text")
        desc.setWordWrap(True)

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.total_count_label)
        panel_layout.addWidget(self.rate_label)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择订阅戒指（不勾选表示全部）",
            max_height=170,
        )

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        self.reset_btn = QPushButton("清空计数")
        self.reset_btn.setProperty("class", "btn-gray")
        self.pc_control_checkbox = QCheckBox("开启 PC 控制")
        self.pc_control_checkbox.setChecked(False)
        self.pc_control_checkbox.setEnabled(HAS_PC_CONTROL)
        action_row.addWidget(self.apply_btn)
        action_row.addWidget(self.reset_btn)
        action_row.addWidget(self.pc_control_checkbox)
        action_row.addStretch()
        panel_layout.addLayout(action_row)
        self.pc_control_hint = QLabel(self.pc_control_hint_text())
        self.pc_control_hint.setProperty("class", "sub-text")
        self.pc_control_hint.setWordWrap(True)
        panel_layout.addWidget(self.pc_control_hint)

        settings_panel = QFrame()
        settings_panel.setProperty("class", "settings-panel")
        settings_layout = QGridLayout(settings_panel)
        settings_layout.setContentsMargins(14, 12, 14, 12)
        settings_layout.setHorizontalSpacing(16)
        settings_layout.setVerticalSpacing(12)

        mode_title = QLabel("处理模式")
        mode_title.setProperty("class", "settings-title")
        mode_switch = QHBoxLayout()
        mode_switch.setSpacing(8)
        self.mode_btn_max = QPushButton("尽可能满帧率")
        self.mode_btn_max.setProperty("class", "segment")
        self.mode_btn_max.setCheckable(True)
        self.mode_btn_max.setChecked(True)
        self.mode_btn_target = QPushButton("指定帧率")
        self.mode_btn_target.setProperty("class", "segment")
        self.mode_btn_target.setCheckable(True)
        mode_switch.addWidget(self.mode_btn_max)
        mode_switch.addWidget(self.mode_btn_target)

        fps_title = QLabel("目标帧率")
        fps_title.setProperty("class", "settings-title")
        fps_desc = QLabel("仅在“指定帧率”模式下生效，用于限制推理刷新速度。")
        fps_desc.setProperty("class", "settings-desc")
        fps_desc.setWordWrap(True)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 240.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSuffix(" Hz")
        self.fps_spin.setEnabled(False)
        self.fps_spin.setMinimumWidth(140)
        self.fps_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.fps_spin.setObjectName("settingsField")

        mode_text = QVBoxLayout()
        mode_text.setSpacing(4)
        mode_text.addWidget(mode_title)

        fps_text = QVBoxLayout()
        fps_text.setSpacing(4)
        fps_text.addWidget(fps_title)
        fps_text.addWidget(fps_desc)

        settings_layout.addLayout(mode_text, 0, 0)
        settings_layout.addLayout(mode_switch, 0, 1)
        settings_layout.addLayout(fps_text, 1, 0)
        settings_layout.addWidget(self.fps_spin, 1, 1)
        settings_layout.setColumnStretch(0, 1)

        self.mode_hint = QLabel("识别线程会先缓存逐帧 IMU 数据；积压不大时逐帧处理，积压过大时批量追赶。")
        self.mode_hint.setProperty("class", "sub-text")
        self.mode_hint.setWordWrap(True)
        panel_layout.addWidget(settings_panel)
        panel_layout.addWidget(self.mode_hint)

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setStyleSheet("border: none;")
        self.cards_content = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_content)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cards_layout.setSpacing(10)
        self.cards_scroll.setWidget(self.cards_content)

        root.addWidget(panel)
        self.cards_hint = QLabel("订阅后才会显示各戒指的识别卡片。")
        self.cards_hint.setProperty("class", "sub-text")
        self.cards_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_hint.setWordWrap(True)
        root.addWidget(self.cards_hint)
        root.addWidget(self.cards_scroll)
        self.cards_scroll.setVisible(False)

        self.empty_hint = QLabel("当前订阅下暂无可显示的戒指。")
        self.empty_hint.setProperty("class", "sub-text")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_layout.addWidget(self.empty_hint)

    def _bind_signals(self):
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.reset_btn.clicked.connect(self.reset_counts)
        self.pc_control_checkbox.toggled.connect(self.on_pc_control_toggled)
        self.mode_btn_max.clicked.connect(
            partial(self._on_mode_button_clicked, GestureInferenceWorker.MODE_MAX)
        )
        self.mode_btn_target.clicked.connect(
            partial(self._on_mode_button_clicked, GestureInferenceWorker.MODE_TARGET)
        )
        self.fps_spin.valueChanged.connect(self.on_processing_mode_changed)

        self.bind_stream_client_signals()
        self.worker.sig_runtime.connect(self.on_worker_runtime)
        self.worker.sig_result.connect(self.on_worker_result)
        self.worker.sig_stats.connect(self.on_worker_stats)

    def _on_mode_button_clicked(self, mode: str, checked: bool) -> None:
        if checked:
            self.set_processing_mode_ui(mode)

    def on_connected_changed(self, connected: bool):
        self._stream_connected = connected
        if not connected and self._worker_ready:
            self.worker.clear_stream_state()
        self.refresh_status()
        if connected:
            self.client.send_command({"cmd": "status"})
            self.refresh_cards()

    def on_error(self, message: str):
        if self._worker_error:
            return
        self.set_stream_error(message)

    def on_worker_runtime(self, ready: bool, error_message: str):
        self._worker_ready = ready
        self._worker_error = error_message
        self.refresh_status()
        self.on_processing_mode_changed()

    def on_worker_result(self, address: str, state_payload: dict, code: int):
        if address not in self.cards:
            self.add_card(address)
        self.cards[address].update_state(
            GestureSnapshot(
                last_code=int(state_payload.get("last_code", -1)),
                last_name=str(state_payload.get("last_name", "-")),
                last_ts=float(state_payload.get("last_ts", 0.0) or 0.0),
                detected_count=int(state_payload.get("detected_count", 0)),
            )
        )
        if code >= 0:
            self.total_count += 1
            self.total_count_label.setText(f"总识别次数: {self.total_count}")
            if self._pc_control_enabled and self.pc_controller is not None:
                self.pc_controller.handle_gesture(address, code)

    def on_worker_stats(self, payload: dict):
        recv_fps = float(payload.get("recv_fps", 0.0) or 0.0)
        infer_fps = float(payload.get("infer_fps", 0.0) or 0.0)
        dropped = int(payload.get("dropped", 0) or 0)
        self.rate_label.setText(
            f"接收: {recv_fps:.1f}Hz | 识别: {infer_fps:.1f}Hz | 丢弃: {dropped}"
        )

    def refresh_status(self):
        if self._worker_error:
            self.status_label.setText(f"模型初始化失败: {self._worker_error}")
            return
        if not self._worker_ready:
            self.status_label.setText("模型初始化中...")
            return
        self.status_label.setText(f"数据流状态：{'已连接' if self._stream_connected else '未连接'}")

    def on_processing_mode_changed(self, *_args):
        mode = self.current_processing_mode()
        is_target = mode == GestureInferenceWorker.MODE_TARGET
        self.fps_spin.setEnabled(is_target and self._worker_ready)
        if self._worker_ready:
            self.worker.set_processing_mode(mode, self.fps_spin.value())

    def current_processing_mode(self) -> str:
        if self.mode_btn_target.isChecked():
            return GestureInferenceWorker.MODE_TARGET
        return GestureInferenceWorker.MODE_MAX

    def set_processing_mode_ui(self, mode: str) -> None:
        is_target = mode == GestureInferenceWorker.MODE_TARGET
        self.mode_btn_max.setChecked(not is_target)
        self.mode_btn_target.setChecked(is_target)
        self.on_processing_mode_changed()

    def handle_stream_message(self, message: dict):
        message_type = str(message.get("type", ""))
        if message_type == "touch":
            return

        if message_type != "imu":
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

        local_ts = float(message.get("local_timestamp", 0.0) or 0.0)
        event_ts = local_ts
        if event_ts <= 0:
            event_ts = float(message.get("host_timestamp", 0.0) or 0.0)
        if event_ts <= 0:
            event_ts = time.time()

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
        if not self._worker_ready:
            return
        self.worker.enqueue_reading(address, data, event_ts)

    def _handle_pc_control_stream_message(self, message: dict) -> None:
        if not self._pc_control_enabled or self.pc_controller is None:
            return

        message_type = str(message.get("type", ""))
        address = str(message.get("address", ""))
        if not address:
            return

        if message_type == "touch":
            self.pc_controller.handle_touch(
                address,
                int(message.get("action_code", -1) or -1),
            )
            return

        if message_type != "imu":
            return

        acc_raw = message.get("acc")
        gyr_raw = message.get("gyr")
        if not isinstance(acc_raw, list) or not isinstance(gyr_raw, list):
            return
        if len(acc_raw) != 3 or len(gyr_raw) != 3:
            return

        local_ts = float(message.get("local_timestamp", 0.0) or 0.0)
        event_ts = local_ts
        if event_ts <= 0:
            event_ts = float(message.get("host_timestamp", 0.0) or 0.0)
        if event_ts <= 0:
            event_ts = time.time()

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
        self.pc_controller.handle_imu(address, data, event_ts)

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ):
        _ = rings
        reset_addresses = changed_connection_addresses + removed_addresses
        if reset_addresses and self._worker_ready:
            self.worker.clear_stream_state(reset_addresses)
        if reset_addresses and self.pc_controller is not None:
            self.pc_controller.clear_states(reset_addresses)
        self.refresh_cards()

    def selected_addresses(self) -> list[str]:
        return self.selected_ring_addresses()

    def effective_addresses(self) -> list[str]:
        selected = self.selected_addresses()
        if selected:
            return selected
        return list(self.ring_info.keys())

    def add_card(self, address: str):
        ring_name = str(self.ring_info.get(address, {}).get("name", "Unknown"))
        card = RingGestureCard(address=address, ring_name=ring_name)
        self.cards[address] = card
        self.cards_layout.addWidget(card)

    def refresh_cards(self):
        if not self._subscription_applied:
            self.cards_hint.setVisible(True)
            self.cards_scroll.setVisible(False)
            return

        self.cards_hint.setVisible(False)
        self.cards_scroll.setVisible(True)
        target = self.effective_addresses()
        for address in list(self.cards.keys()):
            if address not in target:
                card = self.cards.pop(address)
                card.setParent(None)

        for address in target:
            if address and address not in self.cards:
                self.add_card(address)

        self.empty_hint.setVisible(len(target) == 0)

    def apply_subscription(self):
        self._subscription_applied = True
        self.send_subscription(self.selected_streams())
        self.refresh_cards()

    def selected_streams(self) -> list[str]:
        streams = ["imu"]
        if self._pc_control_enabled:
            streams.append("touch")
        return streams

    def on_pc_control_toggled(self, checked: bool) -> None:
        self._pc_control_enabled = bool(checked) and HAS_PC_CONTROL
        self.pc_control_hint.setText(self.pc_control_hint_text())
        if not self._pc_control_enabled and self.pc_controller is not None:
            self.pc_controller.disable()
        if self._subscription_applied:
            self.apply_subscription()

    def pc_control_hint_text(self) -> str:
        if not HAS_PC_CONTROL:
            return f"PC 控制不可用：{PC_CONTROL_ERROR}"
        return "开启后会订阅 touch 事件，并把手势识别结果与 IMU 姿态映射为鼠标/桌面控制。"

    def reset_counts(self):
        self.total_count = 0
        self.total_count_label.setText("总识别次数: 0")
        self.worker.reset_counts()
        for card in self.cards.values():
            card.update_state(GestureSnapshot())

    def closeEvent(self, event):
        self.client.stop()
        self.client.wait(1500)
        self.worker.stop()
        self.worker.wait(1500)
        if self.pc_controller is not None:
            self.pc_controller.stop()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: GestureRecognitionWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
