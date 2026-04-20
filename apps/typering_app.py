from __future__ import annotations

import argparse
import html
import multiprocessing as mp
import os
import queue
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from apps.features.typering import (
    MSG_ERROR,
    MSG_EVENT,
    MSG_IMU_FRAME,
    MSG_QUIT,
    MSG_READY,
    MSG_TOUCH_EVENT,
    TypeRingConfig,
    TypeRingEventLogEntry,
    TypeRingPaths,
    TypeRingSessionState,
    typering_worker_main,
)
from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET
from typering_core.decoder.typering_decoder import TypeRingIMUDecoder
from typering_core.decoder.typering_interface import TypeRingInterface

TYPE_RING_EXTRA_STYLE_SHEET = """
QMainWindow#typeRingWindow,
QWidget#typeRingRoot {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #f4f8fd,
        stop: 0.45 #edf4fb,
        stop: 1 #f8fbff
    );
}

QMainWindow#typeRingWindow QLabel.title {
    font-size: 22px;
    font-weight: 700;
    color: #1f3048;
}

QMainWindow#typeRingWindow QLabel.sub-text {
    font-size: 12px;
    color: #7287a2;
}

QFrame#topBar,
QFrame#candidatePanel,
QFrame#configPanel,
QFrame#compactConfigPanel {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #dde8f3;
    border-radius: 24px;
}

QFrame#heroPanel {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #ffffff,
        stop: 0.55 #f3f8ff,
        stop: 1 #edf6ff
    );
    border: 1px solid #d7e6f7;
    border-radius: 26px;
}

QFrame#configBlock {
    background: #f8fbff;
    border: 1px solid #e3edf7;
    border-radius: 18px;
}

QMainWindow#typeRingWindow QPushButton {
    min-height: 34px;
    border-radius: 12px;
    font-size: 14px;
    padding: 6px 16px;
}

QLabel#heroCaption {
    font-size: 12px;
    font-weight: 700;
    color: #7086a0;
}

QLabel#targetValue {
    font-size: 26px;
    font-weight: 700;
    color: #18263a;
    padding: 4px 0 6px 0;
}

QLabel#inputValue {
    font-size: 36px;
    font-weight: 700;
    color: #0f6fff;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid #dce8f7;
    border-radius: 18px;
    padding: 14px 18px;
}

QLabel#heroMeta {
    font-size: 12px;
    color: #6f849f;
    padding-top: 8px;
}

QLabel#sectionTitle {
    font-size: 15px;
    font-weight: 700;
    color: #22314a;
}

QLabel#sectionHint {
    font-size: 11px;
    color: #8093ab;
}

QLabel#fieldLabel {
    font-size: 12px;
    font-weight: 700;
    color: #60748d;
}

QLabel#deviceStatus {
    background: #f4f8fc;
    border: 1px solid #dde8f2;
    border-radius: 12px;
    padding: 8px 10px;
    color: #50657f;
    font-weight: 600;
}

QComboBox#settingsField,
QSpinBox#settingsField {
    min-height: 34px;
    border-radius: 12px;
    padding: 0 10px;
    font-size: 14px;
}

QComboBox#settingsField::drop-down {
    border: none;
    width: 30px;
}

"""

STYLE_SHEET = APP_STYLE_SHEET + "\n" + TYPE_RING_EXTRA_STYLE_SHEET


class TypeRingWindow(RingStreamWindowBase):
    CANDIDATE_COLUMNS = 4
    CANDIDATE_ROWS = 3
    MONO_FONT_FAMILIES = ["Consolas", "Menlo", "Monaco", "Courier New"]

    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setObjectName("typeRingWindow")
        self.setWindowTitle("TypeRing 戒指打字")
        self.resize(1180, 740)

        self.root_dir = Path(__file__).resolve().parent.parent
        self.paths = TypeRingPaths(self.root_dir)
        self.config = TypeRingConfig()
        self.state = TypeRingSessionState()
        self.session_running = False
        self.left_address = ""
        self.right_address = ""
        self.role_queues: dict[str, tuple[mp.Process, mp.Queue, mp.Queue]] = {}
        self.ring_touch_state: dict[str, bool] = {}
        self.pending_logs: deque[TypeRingEventLogEntry] = deque(maxlen=200)

        self.interface = TypeRingInterface(str(self.paths.text_phrases_path), n_phrase=self.config.n_phrase)
        self.decoder = TypeRingIMUDecoder(
            str(self.paths.char2finger_path),
            str(self.paths.corpus_path),
            str(self.paths.unigrams_path),
            str(self.paths.bigrams_path),
        )

        self.status_text = QLabel("数据流状态：未连接")
        self.status_text.setObjectName("statusTag")
        self.run_status = QLabel("等待开始")
        self.run_status.setObjectName("statusTag")

        self.worker_poll_timer = QTimer(self)
        self.worker_poll_timer.setInterval(20)
        self.worker_poll_timer.timeout.connect(self.poll_worker_messages)

        self._build_ui()
        self.set_stream_status_label(self.status_text)
        self.bind_stream_client_signals()
        self.client.start()
        self.refresh_ui_state()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("typeRingRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        mono_font = self._build_mono_font(15)

        # 初始化基类要求的戒指选择面板容器，避免 status 消息早到时崩溃
        ring_panel_host = QFrame()
        ring_panel_host.setVisible(False)
        ring_panel_layout = QVBoxLayout(ring_panel_host)
        ring_panel_layout.setContentsMargins(0, 0, 0, 0)
        ring_panel_layout.setSpacing(0)
        self.add_ring_selection_panel(
            ring_panel_layout,
            title_text="隐藏戒指列表同步容器",
            max_height=1,
        )
        root.addWidget(ring_panel_host)

        header = QFrame()
        header.setObjectName("topBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("TypeRing 双戒指打字")
        title.setProperty("class", "title")
        desc = QLabel("主显示优先，候选词和配置收到底部。")
        desc.setProperty("class", "sub-text")
        title_block.addWidget(title)
        title_block.addWidget(desc)
        header_layout.addLayout(title_block, 1)

        status_block = QVBoxLayout()
        status_block.setSpacing(8)
        status_block.addWidget(self.status_text)
        status_block.addWidget(self.run_status)
        header_layout.addLayout(status_block)
        root.addWidget(header)

        self.left_combo = QComboBox()
        self.left_combo.setObjectName("settingsField")
        self.right_combo = QComboBox()
        self.right_combo.setObjectName("settingsField")
        self.model_combo = QComboBox()
        self.model_combo.addItem(self.config.model_name)
        self.phrase_spin = QSpinBox()
        self.phrase_spin.setObjectName("settingsField")
        self.phrase_spin.setRange(1, 200)
        self.phrase_spin.setValue(self.config.n_phrase)

        self.left_status = QLabel("左手：未绑定")
        self.left_status.setObjectName("deviceStatus")
        self.right_status = QLabel("右手：未绑定")
        self.right_status.setObjectName("deviceStatus")

        hero_panel = QFrame()
        hero_panel.setObjectName("heroPanel")
        hero_layout = QVBoxLayout(hero_panel)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        hero_layout.setSpacing(4)

        phrase_caption = QLabel("要输入的文本")
        phrase_caption.setObjectName("heroCaption")
        self.phrase_label = QLabel()
        self.phrase_label.setObjectName("targetValue")
        self.phrase_label.setWordWrap(True)
        self.phrase_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.phrase_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.phrase_label.setMinimumHeight(48)

        input_caption = QLabel("正在输入")
        input_caption.setObjectName("heroCaption")
        self.input_label = QLabel()
        self.input_label.setObjectName("inputValue")
        self.input_label.setWordWrap(True)
        self.input_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.input_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.input_label.setMinimumHeight(120)

        self.progress_label = QLabel()
        self.progress_label.setObjectName("heroMeta")

        hero_layout.addWidget(phrase_caption)
        hero_layout.addWidget(self.phrase_label)
        hero_layout.addSpacing(8)
        hero_layout.addWidget(input_caption)
        hero_layout.addWidget(self.input_label)
        hero_layout.addWidget(self.progress_label)
        root.addWidget(hero_panel)

        lower_row = QHBoxLayout()
        lower_row.setSpacing(10)

        candidate_panel = QFrame()
        candidate_panel.setObjectName("candidatePanel")
        candidate_layout = QVBoxLayout(candidate_panel)
        candidate_layout.setContentsMargins(16, 14, 16, 14)
        candidate_layout.setSpacing(8)

        candidate_title = QLabel("候选词矩阵")
        candidate_title.setObjectName("sectionTitle")
        candidate_hint = QLabel("3 × 4，不改交互。")
        candidate_hint.setObjectName("sectionHint")
        candidate_layout.addWidget(candidate_title)
        candidate_layout.addWidget(candidate_hint)

        self.candidate_grid = QGridLayout()
        self.candidate_grid.setContentsMargins(0, 6, 0, 0)
        self.candidate_grid.setHorizontalSpacing(8)
        self.candidate_grid.setVerticalSpacing(8)
        self.candidate_cells: list[QLabel] = []
        candidate_layout.addLayout(self.candidate_grid)
        lower_row.addWidget(candidate_panel, 3)

        self._build_candidate_grid()

        config_panel = QFrame()
        config_panel.setObjectName("compactConfigPanel")
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(16, 14, 16, 14)
        config_layout.setSpacing(8)
        config_title = QLabel("设备与配置")
        config_title.setObjectName("sectionTitle")
        config_hint = QLabel("紧凑模式")
        config_hint.setObjectName("sectionHint")
        config_layout.addWidget(config_title)
        config_layout.addWidget(config_hint)

        config_block = QFrame()
        config_block.setObjectName("configBlock")
        binding_layout = QGridLayout(config_block)
        binding_layout.setContentsMargins(12, 12, 12, 12)
        binding_layout.setHorizontalSpacing(8)
        binding_layout.setVerticalSpacing(8)
        left_label = QLabel("左手戒指")
        left_label.setObjectName("fieldLabel")
        right_label = QLabel("右手戒指")
        right_label.setObjectName("fieldLabel")
        binding_layout.addWidget(left_label, 0, 0)
        binding_layout.addWidget(right_label, 0, 1)
        binding_layout.addWidget(self.left_combo, 1, 0)
        binding_layout.addWidget(self.right_combo, 1, 1)
        binding_layout.addWidget(self.left_status, 2, 0)
        binding_layout.addWidget(self.right_status, 2, 1)
        phrase_label = QLabel("短语数量")
        phrase_label.setObjectName("fieldLabel")
        binding_layout.addWidget(phrase_label, 3, 0)
        binding_layout.addWidget(self.phrase_spin, 4, 0)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.btn_start = QPushButton("开始打字")
        self.btn_start.setProperty("class", "btn-primary")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setProperty("class", "btn-danger")
        self.btn_reset = QPushButton("重置")
        self.btn_reset.setProperty("class", "btn-gray")
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_stop)
        button_row.addWidget(self.btn_reset)
        binding_layout.addLayout(button_row, 4, 1)
        config_layout.addWidget(config_block)
        lower_row.addWidget(config_panel, 2)

        root.addLayout(lower_row)
        root.setStretch(0, 0)
        root.setStretch(1, 0)
        root.setStretch(2, 0)
        root.setStretch(3, 1)

        self.btn_start.clicked.connect(self.start_typing)
        self.btn_stop.clicked.connect(self.stop_typing)
        self.btn_reset.clicked.connect(self.reset_session)
        self.left_combo.currentIndexChanged.connect(self.on_role_binding_changed)
        self.right_combo.currentIndexChanged.connect(self.on_role_binding_changed)

        self.model_combo.hide()

    def _build_candidate_grid(self) -> None:
        for row in range(self.CANDIDATE_ROWS):
            for col in range(self.CANDIDATE_COLUMNS):
                label = QLabel("")
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                label.setMinimumHeight(36)
                label.setMaximumHeight(36)
                label.setMinimumWidth(0)
                label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                label.setFont(self._build_mono_font(13))
                self.candidate_grid.addWidget(label, row, col)
                self.candidate_cells.append(label)

    def _build_mono_font(self, point_size: int) -> QFont:
        font = QFont()
        for family in self.MONO_FONT_FAMILIES:
            font.setFamilies([family])
            if font.exactMatch():
                break
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(point_size)
        return font

    def on_stream_connected(self) -> None:
        self.client.send_command({"cmd": "status"})
        self.update_subscription()

    def on_rings_updated(self, *, rings, removed_addresses, changed_connection_addresses) -> None:
        _ = removed_addresses
        _ = changed_connection_addresses
        self._sync_ring_combos(rings)
        self.refresh_binding_status()
        self.update_subscription()

    def _sync_ring_combos(self, rings: list[dict]) -> None:
        items = [("", "请选择戒指")] + [
            (str(r.get("address", "")), self.format_ring_text(r))
            for r in rings
            if str(r.get("address", ""))
        ]
        for combo, current in ((self.left_combo, self.left_address), (self.right_combo, self.right_address)):
            combo.blockSignals(True)
            combo.clear()
            for address, text in items:
                combo.addItem(text, address)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def on_role_binding_changed(self) -> None:
        self.left_address = str(self.left_combo.currentData() or "")
        self.right_address = str(self.right_combo.currentData() or "")
        self.refresh_binding_status()
        self.update_subscription()

    def refresh_binding_status(self) -> None:
        self._update_binding_label(self.left_status, "左手", self.left_address)
        self._update_binding_label(self.right_status, "右手", self.right_address)

    def _update_binding_label(self, label: QLabel, side: str, address: str) -> None:
        ring = self.ring_info.get(address) if address else None
        if not address:
            label.setText(f"{side} · 未绑定")
            label.setStyleSheet(
                "background: #f4f8fc; border: 1px solid #dde8f2; border-radius: 14px; "
                "padding: 10px 12px; color: #6f839b; font-weight: 600;"
            )
            return
        if not ring or not bool(ring.get("connected")):
            label.setText(f"{side} · {self.describe_ring(address)}")
            label.setStyleSheet(
                "background: #fdf5f4; border: 1px solid #f0d8d4; border-radius: 14px; "
                "padding: 10px 12px; color: #9d5e58; font-weight: 600;"
            )
            return
        label.setText(f"{side} · {self.describe_ring(address)}")
        label.setStyleSheet(
            "background: #eff8f3; border: 1px solid #d5eadc; border-radius: 14px; "
            "padding: 10px 12px; color: #47735d; font-weight: 600;"
        )

    def describe_ring(self, address: str) -> str:
        if not address:
            return "未绑定"
        ring = self.ring_info.get(address)
        if not ring:
            return f"{address}（不可用）"
        suffix = "在线" if ring.get("connected") else "离线"
        return f"{ring.get('name', 'Unknown')} | {suffix}"

    def update_subscription(self) -> None:
        selected = []
        if self.left_address:
            selected.append(self.left_address)
        if self.right_address and self.right_address not in selected:
            selected.append(self.right_address)
        rings = selected if selected else ["__none__"]
        self.client.send_command({"cmd": "subscribe", "rings": rings, "streams": ["imu", "touch"]})

    def handle_stream_message(self, message: dict) -> None:
        msg_type = message.get("type")
        if msg_type == "subscribed":
            return
        if msg_type == "imu":
            self.handle_imu_message(message)
            return
        if msg_type == "touch":
            self.handle_touch_message(message)
            return

    def handle_touch_message(self, message: dict) -> None:
        address = str(message.get("address", ""))
        if address not in {self.left_address, self.right_address}:
            return
        action_code = int(message.get("action_code", -1))
        self.ring_touch_state[address] = action_code != 0
        if not self.session_running:
            return
        role = self.role_for_address(address)
        if role is None or role not in self.role_queues:
            return
        _proc, in_q, _out_q = self.role_queues[role]
        tap_timestamp = message.get("tap_timestamp")
        release_timestamp = message.get("release_timestamp")
        if tap_timestamp is not None:
            in_q.put(
                {
                    "type": MSG_TOUCH_EVENT,
                    "action": "press",
                    "timestamp": float(tap_timestamp),
                }
            )
        if release_timestamp is not None:
            in_q.put(
                {
                    "type": MSG_TOUCH_EVENT,
                    "action": "release",
                    "timestamp": float(release_timestamp),
                }
            )

    def handle_imu_message(self, message: dict) -> None:
        if not self.session_running:
            return
        address = str(message.get("address", ""))
        role = self.role_for_address(address)
        if role is None or role not in self.role_queues:
            return
        process, in_q, _out_q = self.role_queues[role]
        _ = process
        acc = message.get("acc")
        gyr = message.get("gyr")
        if not isinstance(acc, list) or not isinstance(gyr, list) or len(acc) != 3 or len(gyr) != 3:
            return
        timestamp = float(message.get("host_timestamp") or message.get("local_timestamp") or 0.0)
        in_q.put(
            {
                "type": MSG_IMU_FRAME,
                "acc": [float(v) for v in acc],
                "gyr": [float(v) for v in gyr],
                "timestamp": timestamp,
            }
        )

    def role_for_address(self, address: str) -> str | None:
        if address == self.left_address:
            return "left"
        if address == self.right_address:
            return "right"
        return None

    def start_typing(self) -> None:
        if self.session_running:
            return
        validation = self.validate_before_start()
        if validation:
            self.run_status.setText(f"无法开始：{validation}")
            return

        self.config.model_name = self.model_combo.currentText().strip() or self.config.model_name
        self.config.n_phrase = int(self.phrase_spin.value())
        self.config.n_candidate = self.CANDIDATE_COLUMNS * self.CANDIDATE_ROWS
        self.interface = TypeRingInterface(str(self.paths.text_phrases_path), n_phrase=self.config.n_phrase)
        self.decoder = TypeRingIMUDecoder(
            str(self.paths.char2finger_path),
            str(self.paths.corpus_path),
            str(self.paths.unigrams_path),
            str(self.paths.bigrams_path),
        )
        self.pending_logs.clear()
        self.spawn_workers()
        self.session_running = True
        self.worker_poll_timer.start()
        self.run_status.setText("运行中：等待左右手事件")
        self.refresh_ui_state()

    def validate_before_start(self) -> str:
        if not self.left_address or not self.right_address:
            return "请先选择左右手戒指"
        if self.left_address == self.right_address:
            return "左右手不能绑定同一枚戒指"
        left_ring = self.ring_info.get(self.left_address)
        right_ring = self.ring_info.get(self.right_address)
        if not left_ring or not bool(left_ring.get("connected")):
            return "左手戒指未在线"
        if not right_ring or not bool(right_ring.get("connected")):
            return "右手戒指未在线"
        missing = self.paths.validate(self.model_combo.currentText().strip() or self.config.model_name)
        if missing:
            return f"缺少资源：{Path(missing[0]).name}"
        return ""

    def spawn_workers(self) -> None:
        self.stop_workers()
        ctx = mp.get_context("spawn")
        config_dict = asdict(self.config)
        for role in ("left", "right"):
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            proc = ctx.Process(
                target=typering_worker_main,
                args=(role, str(self.root_dir), config_dict, in_q, out_q),
                daemon=True,
            )
            proc.start()
            self.role_queues[role] = (proc, in_q, out_q)

    def poll_worker_messages(self) -> None:
        for role, (proc, _in_q, out_q) in list(self.role_queues.items()):
            if not proc.is_alive() and proc.exitcode not in (None, 0):
                self.run_status.setText(f"{role} worker 异常退出：{proc.exitcode}")
                self.stop_typing()
                return
            while True:
                try:
                    message = out_q.get_nowait()
                except queue.Empty:
                    break
                self.handle_worker_message(message)

    def handle_worker_message(self, message: dict) -> None:
        msg_type = message.get("type")
        role = str(message.get("role", ""))
        if msg_type == MSG_READY:
            self.pending_logs.append(TypeRingEventLogEntry(time.time(), role, "worker ready", ""))
            return
        if msg_type == MSG_ERROR:
            self.run_status.setText(f"{role} worker 错误：{message.get('message', '')}")
            self.stop_typing()
            return
        if msg_type != MSG_EVENT:
            return

        event_name = str(message.get("event", ""))
        send_ts = float(message.get("timestamp", 0.0))
        event = (f"{role} {event_name}", send_ts, time.perf_counter())
        try:
            result = self.interface.decode(event, (float("nan"), float("nan")))
        except IndexError:
            self._repair_interface_phrase_overflow()
            result = self.interface.decode(event, (float("nan"), float("nan")))
        action_text = ""
        if result is not None and result.get("type") == "data":
            candidates = self.decoder.decode(
                result.get("prefix"),
                result.get("fingers", []),
                self.config.n_candidate,
            )
            if candidates is not None:
                self.interface.update_candidates(candidates)
                action_text = f"候选 {', '.join(candidates[:3])}"
        elif result is not None and result.get("type") == "quit":
            action_text = "已完成全部短语"
            self._append_session_metrics_log()
            self.stop_typing()

        self.pending_logs.append(TypeRingEventLogEntry(send_ts, role, event_name, action_text))
        self.refresh_ui_state()

    def _repair_interface_phrase_overflow(self) -> None:
        current_phrase = getattr(self.interface, "current_phrase", "")
        current_input = getattr(self.interface, "current_input", "")
        if not current_phrase or len(current_input) <= len(current_phrase):
            raise

        overflow = len(current_input) - len(current_phrase)
        if overflow <= 0:
            raise

        self.interface.current_input = current_input[:-overflow]
        if hasattr(self.interface, "fingers"):
            self.interface.fingers = self.interface.fingers[:-overflow]
        if hasattr(self.interface, "positions"):
            self.interface.positions = self.interface.positions[:-overflow]
        if hasattr(self.interface, "candidates"):
            self.interface.candidates.clear()
        if hasattr(self.interface, "candidate_idx"):
            self.interface.candidate_idx = 0

        self.pending_logs.append(
            TypeRingEventLogEntry(
                time.time(),
                "system",
                "repair",
                f"截断超出目标短语的 {overflow} 个字符",
            )
        )

    def refresh_ui_state(self) -> None:
        current_phrase = getattr(self.interface, "current_phrase", "")
        current_input = getattr(self.interface, "current_input", "")
        phrase_idx = int(getattr(self.interface, "phrase_idx", 0))
        phrase_total = int(getattr(self.interface, "n_phrase", self.phrase_spin.value()))

        self.phrase_label.setText(self._render_phrase_text(current_phrase, current_input))
        self.input_label.setText(self._render_input_text(current_input, current_phrase))
        self.progress_label.setText(
            f"短语 {min(phrase_idx + 1, max(phrase_total, 1))}/{max(phrase_total, 1)}"
            f" · 候选区保持 {self.CANDIDATE_ROWS} × {self.CANDIDATE_COLUMNS}"
        )
        candidates = list(getattr(self.interface, "candidates", []))
        candidate_index = int(getattr(self.interface, "candidate_idx", 0))
        total_cells = self.CANDIDATE_COLUMNS * self.CANDIDATE_ROWS
        for index in range(total_cells):
            label = self.candidate_cells[index]
            if index < len(candidates):
                prefix = "▶ " if index == candidate_index else "   "
                label.setText(f"{prefix}{candidates[index]}")
                if index == candidate_index:
                    label.setStyleSheet(
                        "background: #0f6fff; border: 1px solid #0f6fff; border-radius: 14px; "
                        "color: #ffffff; font-weight: 700; padding: 0 12px;"
                    )
                else:
                    label.setStyleSheet(
                        "background: #f3f7fc; border: 1px solid #e1eaf3; border-radius: 14px; "
                        "color: #4a617c; font-weight: 600; padding: 0 12px;"
                    )
            else:
                label.setText("")
                label.setStyleSheet(
                    "background: #fbfdff; border: 1px dashed #e7eef6; border-radius: 14px; "
                    "color: #9cafc2; padding: 0 12px;"
                )
        self.btn_start.setEnabled(not self.session_running)
        self.btn_stop.setEnabled(self.session_running)
        self.refresh_binding_status()

    def _render_phrase_text(self, phrase: str, current_input: str) -> str:
        if not phrase:
            return '<span style="color:#8aa0b8;">等待加载短语...</span>'

        matched_len = 0
        for expected, actual in zip(phrase, current_input):
            if expected != actual:
                break
            matched_len += 1

        matched = html.escape(phrase[:matched_len]).replace(" ", "&nbsp;")
        pending = html.escape(phrase[matched_len:]).replace(" ", "&nbsp;")
        return (
            f'<span style="color:#1b2a40;">{matched}</span>'
            f'<span style="color:#8ea2b8;">{pending}</span>'
        )

    def _render_input_text(self, current_input: str, current_phrase: str) -> str:
        if not current_input:
            return '<span style="color:#90a3b8;">等待手势输入...</span>'

        matched_len = 0
        for expected, actual in zip(current_phrase, current_input):
            if expected != actual:
                break
            matched_len += 1

        matched = html.escape(current_input[:matched_len]).replace(" ", "&nbsp;")
        mismatch = html.escape(current_input[matched_len:]).replace(" ", "&nbsp;")
        if mismatch:
            return (
                f'<span style="color:#0f6fff;">{matched}</span>'
                f'<span style="color:#d64545;">{mismatch}</span>'
                f'<span style="color:#9db0c4;">&nbsp;|</span>'
            )
        return (
            f'<span style="color:#0f6fff;">{matched}</span>'
            f'<span style="color:#9db0c4;">&nbsp;|</span>'
        )

    def _append_session_metrics_log(self) -> None:
        if getattr(self.interface, "toc", None) is None or getattr(self.interface, "tic", None) is None:
            return
        try:
            metrics = self.interface.save_data()
        except Exception:
            return
        summary = (
            f"WPM={metrics.get('wpm', 0.0):.2f}, "
            f"UER={metrics.get('uer', 0.0):.3f}, "
            f"CER={metrics.get('cer', 0.0):.3f}, "
            f"delete={metrics.get('delete_cnt', 0)}, "
            f"switch={metrics.get('switch_cnt', 0)}"
        )
        self.pending_logs.append(
            TypeRingEventLogEntry(
                time.time(),
                "metrics",
                "session summary",
                summary,
            )
        )

    def stop_typing(self) -> None:
        self.session_running = False
        self.worker_poll_timer.stop()
        self.stop_workers()
        self.run_status.setText("已停止")
        self.refresh_ui_state()

    def stop_workers(self) -> None:
        for role, (proc, in_q, _out_q) in list(self.role_queues.items()):
            try:
                in_q.put({"type": MSG_QUIT})
            except Exception:
                pass
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
            del self.role_queues[role]

    def reset_session(self) -> None:
        if self.session_running:
            self.stop_typing()
        self.interface = TypeRingInterface(str(self.paths.text_phrases_path), n_phrase=self.phrase_spin.value())
        self.pending_logs.clear()
        self.run_status.setText("等待开始")
        self.refresh_ui_state()

    def closeEvent(self, event) -> None:
        self.stop_typing()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    if sys.platform.startswith("darwin"):
        mp.set_start_method("spawn", force=True)
    args = parse_args()
    return run_window(
        lambda: TypeRingWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
        style="Fusion",
    )


if __name__ == "__main__":
    raise SystemExit(main())
