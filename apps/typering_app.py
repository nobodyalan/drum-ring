from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from common.stream_window import RingStreamWindowBase
from common.ui_style import APP_STYLE_SHEET
from typering_core.decoder.typering_decoder import TypeRingIMUDecoder
from typering_core.decoder.typering_interface import TypeRingInterface

from apps.typering_support.config import TypeRingConfig, TypeRingPaths
from apps.typering_support.state import TypeRingEventLogEntry, TypeRingSessionState
from apps.typering_support.worker import (
    MSG_ERROR,
    MSG_EVENT,
    MSG_IMU_FRAME,
    MSG_QUIT,
    MSG_READY,
    MSG_TOUCH_EVENT,
    typering_worker_main,
)

STYLE_SHEET = APP_STYLE_SHEET


class TypeRingWindow(RingStreamWindowBase):
    CANDIDATE_COLUMNS = 4
    CANDIDATE_ROWS = 3
    MONO_FONT_FAMILIES = ["Consolas", "Menlo", "Monaco", "Courier New"]

    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("TypeRing 戒指打字")
        self.resize(1100, 760)

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
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
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
        header.setProperty("class", "panel")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(6)
        title = QLabel("TypeRing 双戒指打字")
        title.setProperty("class", "title")
        desc = QLabel("从主程序连接两枚戒指，在此绑定左右手并可视化打字过程。")
        desc.setProperty("class", "sub-text")
        header_layout.addWidget(title)
        header_layout.addWidget(desc)
        header_layout.addWidget(self.status_text)
        header_layout.addWidget(self.run_status)
        root.addWidget(header)

        body = QGridLayout()
        body.setHorizontalSpacing(10)
        body.setVerticalSpacing(10)

        config_panel = QFrame()
        config_panel.setProperty("class", "panel")
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(16, 14, 16, 14)
        config_layout.setSpacing(8)
        config_layout.addWidget(QLabel("设备绑定"))

        self.left_combo = QComboBox()
        self.right_combo = QComboBox()
        self.model_combo = QComboBox()
        self.model_combo.addItem(self.config.model_name)
        self.phrase_spin = QSpinBox()
        self.phrase_spin.setRange(1, 200)
        self.phrase_spin.setValue(self.config.n_phrase)

        config_layout.addWidget(QLabel("左手戒指"))
        config_layout.addWidget(self.left_combo)
        config_layout.addWidget(QLabel("右手戒指"))
        config_layout.addWidget(self.right_combo)
        config_layout.addWidget(QLabel("短语数量"))
        config_layout.addWidget(self.phrase_spin)

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
        config_layout.addLayout(button_row)

        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(8)
        status_grid.setVerticalSpacing(6)
        self.left_status = QLabel("左手：未绑定")
        self.right_status = QLabel("右手：未绑定")
        status_grid.addWidget(self.left_status, 0, 0)
        status_grid.addWidget(self.right_status, 1, 0)
        config_layout.addLayout(status_grid)

        body.addWidget(config_panel, 0, 0)

        text_panel = QFrame()
        text_panel.setProperty("class", "panel")
        text_layout = QVBoxLayout(text_panel)
        text_layout.setContentsMargins(16, 14, 16, 14)
        text_layout.setSpacing(4)

        phrase_row = QHBoxLayout()
        phrase_row.setSpacing(10)
        phrase_row.setContentsMargins(0, 0, 0, 0)
        phrase_title = QLabel("目标短语")
        self.phrase_label = QLabel()
        self.phrase_label.setWordWrap(False)
        self.phrase_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.phrase_label.setFont(mono_font)
        self.phrase_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        phrase_row.addWidget(phrase_title)
        phrase_row.addWidget(self.phrase_label, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        input_row.setContentsMargins(0, 0, 0, 0)
        input_title = QLabel("当前输入")
        self.input_label = QLabel()
        self.input_label.setWordWrap(False)
        self.input_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.input_label.setFont(mono_font)
        self.input_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        input_row.addWidget(input_title)
        input_row.addWidget(self.input_label, 1)

        self.committed_label = QLabel()
        self.committed_label.setWordWrap(True)
        self.committed_label.setFont(mono_font)
        self.committed_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.committed_label.setMinimumHeight(72)
        self.committed_label.setMaximumHeight(120)
        self.candidate_grid = QGridLayout()
        self.candidate_grid.setContentsMargins(0, 0, 0, 0)
        self.candidate_grid.setHorizontalSpacing(4)
        self.candidate_grid.setVerticalSpacing(0)
        self.candidate_cells: list[QLabel] = []
        text_layout.addLayout(phrase_row)
        text_layout.addLayout(input_row)
        candidate_title = QLabel("候选词（3 × 4）")
        candidate_title.setMaximumHeight(22)
        committed_title = QLabel("已提交文本")
        committed_title.setMaximumHeight(22)
        text_layout.addWidget(candidate_title)
        text_layout.addLayout(self.candidate_grid)
        text_layout.addWidget(committed_title)
        text_layout.addWidget(self.committed_label)
        body.addWidget(text_panel, 0, 1)

        self._build_candidate_grid()

        log_panel = QFrame()
        log_panel.setProperty("class", "panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(16, 14, 16, 14)
        log_layout.setSpacing(6)
        log_layout.addWidget(QLabel("事件流"))
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setFont(mono_font)
        self.event_log.setMinimumHeight(260)
        log_layout.addWidget(self.event_log)
        body.addWidget(log_panel, 1, 0, 1, 2)

        body.setRowStretch(0, 0)
        body.setRowStretch(1, 1)
        body.setColumnStretch(1, 2)
        root.addLayout(body)

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
                label.setMinimumHeight(20)
                label.setMaximumHeight(20)
                label.setMinimumWidth(0)
                label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                label.setFont(self._build_mono_font(14))
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
        self.left_status.setText(f"左手：{self.describe_ring(self.left_address)}")
        self.right_status.setText(f"右手：{self.describe_ring(self.right_address)}")

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
        self.event_log.clear()
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
            self.render_logs()
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
        self.render_logs()

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
        committed = getattr(self.interface, "committed_inputs", [])
        self.phrase_label.setText(getattr(self.interface, "current_phrase", ""))
        self.input_label.setText(getattr(self.interface, "current_input", ""))
        self.committed_label.setText("\n".join(committed[-3:]) if committed else "")
        candidates = list(getattr(self.interface, "candidates", []))
        candidate_index = int(getattr(self.interface, "candidate_idx", 0))
        total_cells = self.CANDIDATE_COLUMNS * self.CANDIDATE_ROWS
        for index in range(total_cells):
            label = self.candidate_cells[index]
            if index < len(candidates):
                prefix = "▶ " if index == candidate_index else "  "
                label.setText(f"{prefix}{candidates[index]}")
                if index == candidate_index:
                    label.setStyleSheet(
                        "color: #1d4ed8; font-weight: 700; padding: 0 2px; background: transparent;"
                    )
                else:
                    label.setStyleSheet(
                        "color: #475569; font-weight: 500; padding: 0 2px; background: transparent;"
                    )
            else:
                label.setText("")
                label.setStyleSheet(
                    "color: #94a3b8; padding: 0 2px; background: transparent;"
                )
        self.btn_start.setEnabled(not self.session_running)
        self.btn_stop.setEnabled(self.session_running)
        self.refresh_binding_status()

    def render_logs(self) -> None:
        lines = []
        for entry in list(self.pending_logs)[-50:]:
            ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp)) if entry.timestamp > 1000 else f"{entry.timestamp:.3f}"
            suffix = f" | {entry.action}" if entry.action else ""
            lines.append(f"[{ts}] {entry.source}: {entry.event}{suffix}")
        self.event_log.setPlainText("\n".join(lines))
        cursor = self.event_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.event_log.setTextCursor(cursor)

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
        self.event_log.clear()
        self.run_status.setText("等待开始")
        self.refresh_ui_state()

    def closeEvent(self, event) -> None:
        self.stop_typing()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17888)
    return parser.parse_args()


if __name__ == "__main__":
    if sys.platform.startswith("darwin"):
        mp.set_start_method("spawn", force=True)
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    win = TypeRingWindow(host=args.host, port=args.port)
    win.show()
    sys.exit(app.exec())
