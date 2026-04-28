from __future__ import annotations

import argparse
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_client import JsonLineStreamClient
from apps.framework.ui_style import APP_STYLE_SHEET
from whisper.whisper_with_pretrain_recognizer import WhisperWithPretrainRecognizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change this single value to switch the demo UI language: "en" or "zh".
UI_LANGUAGE = "en"

UI_TEXT = {
    "en": {
        "window_title": "FingerScribe Demo",
        "waiting": "Waiting",
        "recording": "Recording",
        "digits": "Digits",
        "letters": "Words",
        "latency": "Model latency",
        "confidence": "Confidence",
        "history": "History",
        "history_empty": " ",
    },
    "zh": {
        "window_title": "手写识别展示",
        "waiting": "等待输入",
        "recording": "正在输入",
        "digits": "数字",
        "letters": "单词",
        "latency": "模型耗时",
        "confidence": "置信度",
        "history": "记录",
        "history_empty": " ",
    },
}

if UI_LANGUAGE not in UI_TEXT:
    raise ValueError(f"Unsupported UI_LANGUAGE: {UI_LANGUAGE!r}")


def tr(key: str) -> str:
    return UI_TEXT[UI_LANGUAGE][key]


@dataclass(frozen=True)
class RecognitionMode:
    key: str
    label: str
    checkpoint_dir: Path


@dataclass(frozen=True)
class InferenceJob:
    job_id: int
    mode_key: str
    values: np.ndarray
    sample_count: int
    capture_ms: float


@dataclass(frozen=True)
class InferenceResult:
    job_id: int
    mode_key: str
    text: str
    confidence: float
    inference_ms: float
    sample_count: int
    capture_ms: float
    error: str = ""


MODES = (
    RecognitionMode(
        key="digits",
        label=tr("digits"),
        checkpoint_dir=PROJECT_ROOT / "checkpoint" / "whisper-number-exp",
    ),
    RecognitionMode(
        key="letters",
        label=tr("letters"),
        checkpoint_dir=PROJECT_ROOT / "checkpoint" / "whisper-word-exp",
    ),
)


DEMO_STYLE = (
    APP_STYLE_SHEET
    + """
QMainWindow, QWidget {
    background-color: #f7f9fc;
}
QFrame#demoShell {
    background-color: #ffffff;
    border: 1px solid #e4ebf3;
    border-radius: 24px;
}
QLabel#stateText {
    font-size: 30px;
    font-weight: 800;
    color: #26384d;
}
QLabel#resultText {
    font-size: 84px;
    font-weight: 900;
    color: #1f2d3d;
}
QLabel#metricLabel {
    font-size: 13px;
    font-weight: 700;
    color: #8190a1;
}
QLabel#metricValue {
    font-size: 30px;
    font-weight: 800;
    color: #26384d;
}
QLabel#historyTitle {
    font-size: 13px;
    font-weight: 800;
    color: #8190a1;
}
QLabel#historySentence {
    background: #f7f9fc;
    border: 1px solid #e6edf5;
    border-radius: 16px;
    padding: 16px 18px;
    font-size: 26px;
    font-weight: 800;
    color: #2e4055;
}
QPushButton.modeButton {
    min-height: 44px;
    border-radius: 22px;
    padding: 0 28px;
    font-size: 16px;
    font-weight: 800;
    color: #526579;
    background: #f2f5f9;
    border: 1px solid #dfe7f0;
}
QPushButton.modeButton:checked {
    color: #ffffff;
    background: #2f6f9f;
    border: 1px solid #2f6f9f;
}
"""
)


class DemoInferenceWorker(QThread):
    ready = pyqtSignal()
    status_changed = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, modes: tuple[RecognitionMode, ...]):
        super().__init__()
        self.modes = modes
        self._jobs: queue.Queue[InferenceJob | None] = queue.Queue()
        self._stopping = False

    def submit(self, job: InferenceJob) -> None:
        self._jobs.put(job)

    def stop(self) -> None:
        self._stopping = True
        self._jobs.put(None)
        self.wait(1500)

    def run(self) -> None:
        recognizers: dict[str, WhisperWithPretrainRecognizer] = {}
        try:
            for mode in self.modes:
                recognizers[mode.key] = WhisperWithPretrainRecognizer(mode.checkpoint_dir, device="cpu")
        except Exception as exc:
            self.fatal_error.emit(str(exc))
            return

        self.ready.emit()
        while not self._stopping:
            job = self._jobs.get()
            if job is None:
                return

            started = time.perf_counter()
            try:
                prediction = recognizers[job.mode_key].predict(job.values)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.result_ready.emit(
                    InferenceResult(
                        job_id=job.job_id,
                        mode_key=job.mode_key,
                        text=prediction.text,
                        confidence=prediction.average_confidence,
                        inference_ms=elapsed_ms,
                        sample_count=job.sample_count,
                        capture_ms=job.capture_ms,
                    )
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.result_ready.emit(
                    InferenceResult(
                        job_id=job.job_id,
                        mode_key=job.mode_key,
                        text="",
                        confidence=0.0,
                        inference_ms=elapsed_ms,
                        sample_count=job.sample_count,
                        capture_ms=job.capture_ms,
                        error=str(exc),
                    )
                )


class HandwritingDemoWindow(QMainWindow):
    def __init__(self, host: str, port: int):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.apply_half_screen_size()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.client = JsonLineStreamClient(host=host, port=port)
        self.client.sig_connected.connect(self.on_connected)
        self.client.sig_message.connect(self.on_stream_message)
        self.client.sig_error.connect(self.on_stream_error)

        self.worker = DemoInferenceWorker(MODES)
        self.worker.ready.connect(self.on_worker_ready)
        self.worker.result_ready.connect(self.on_inference_result)
        self.worker.fatal_error.connect(self.on_worker_error)

        self.rings: dict[str, dict[str, Any]] = {}
        self.active_ring_address = ""
        self.active_ring_name = ""
        self.mode_key = MODES[0].key
        self.recording = False
        self.runtime_ready = False
        self.current_samples: list[list[float]] = []
        self.started_at = 0.0
        self.latest_job_id = 0
        self.history_by_mode: dict[str, list[str]] = {mode.key: [] for mode in MODES}

        self.status_dot = QFrame()
        self.status_dot.setFixedSize(26, 26)
        self.state_label = QLabel(tr("waiting"))
        self.state_label.setObjectName("stateText")
        self.result_label = QLabel("-")
        self.result_label.setObjectName("resultText")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.elapsed_value = QLabel("-- ms")
        self.elapsed_value.setObjectName("metricValue")
        self.confidence_value = QLabel("--")
        self.confidence_value.setObjectName("metricValue")
        self.history_sentence_label = QLabel("")
        self.history_sentence_label.setObjectName("historySentence")
        self.history_sentence_label.setWordWrap(True)
        self.history_sentence_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}

        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(200)
        self.capture_timer.timeout.connect(self.refresh_capture_state)
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_capture)

        self._build_ui()
        self.set_waiting_state()
        self.worker.start()
        self.client.start()
        self.capture_timer.start()

    def apply_half_screen_size(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(760, 900)
            return
        available = screen.availableGeometry()
        width = max(640, available.width() // 2)
        height = max(720, available.height())
        self.resize(width, height)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(42, 36, 42, 36)
        root.setSpacing(0)

        shell = QFrame()
        shell.setObjectName("demoShell")
        root.addWidget(shell, 1)

        layout = QVBoxLayout(shell)
        layout.setContentsMargins(42, 36, 42, 36)
        layout.setSpacing(28)

        top_row = QHBoxLayout()
        top_row.setSpacing(14)
        top_row.addWidget(self.status_dot)
        top_row.addWidget(self.state_label)
        top_row.addStretch()

        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        for mode in MODES:
            button = QPushButton(mode.label)
            button.setCheckable(True)
            button.setProperty("class", "modeButton")
            button.clicked.connect(lambda _checked=False, key=mode.key: self.set_mode(key))
            self.mode_group.addButton(button)
            self.mode_buttons[mode.key] = button
            mode_row.addWidget(button)
        self.mode_buttons[self.mode_key].setChecked(True)
        top_row.addLayout(mode_row)
        layout.addLayout(top_row)

        layout.addStretch(1)
        layout.addWidget(self.result_label, 3)
        layout.addStretch(1)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(42)
        metrics.setVerticalSpacing(8)
        elapsed_label = QLabel(tr("latency"))
        elapsed_label.setObjectName("metricLabel")
        confidence_label = QLabel(tr("confidence"))
        confidence_label.setObjectName("metricLabel")
        metrics.addWidget(elapsed_label, 0, 0)
        metrics.addWidget(confidence_label, 0, 1)
        metrics.addWidget(self.elapsed_value, 1, 0)
        metrics.addWidget(self.confidence_value, 1, 1)
        layout.addLayout(metrics)

        history_title = QLabel(tr("history"))
        history_title.setObjectName("historyTitle")
        layout.addWidget(history_title)
        layout.addWidget(self.history_sentence_label, 2)
        self.refresh_history()

    def on_connected(self, connected: bool) -> None:
        if connected:
            self.client.send_command({"cmd": "status"})
            self.client.send_command({"cmd": "subscribe", "rings": [], "streams": ["imu"]})

    def on_stream_error(self, _message: str) -> None:
        if not self.recording:
            self.set_waiting_state()

    def on_stream_message(self, message: dict[str, Any]) -> None:
        message_type = str(message.get("type", ""))
        if message_type == "status":
            self.sync_online_ring(message.get("rings", []))
            return
        if message_type == "imu":
            self.handle_imu(message)

    def sync_online_ring(self, rings: Any) -> None:
        if not isinstance(rings, list):
            return
        self.rings = {
            str(item.get("address", "")): dict(item)
            for item in rings
            if str(item.get("address", "")) and bool(item.get("connected", False))
        }
        if self.active_ring_address in self.rings:
            self.active_ring_name = self.ring_name(self.active_ring_address)
            return
        self.active_ring_address = next(iter(self.rings), "")
        self.active_ring_name = self.ring_name(self.active_ring_address) if self.active_ring_address else ""

    def handle_imu(self, message: dict[str, Any]) -> None:
        if not self.recording:
            return
        address = str(message.get("address", ""))
        if self.active_ring_address and address != self.active_ring_address:
            return
        if not self.active_ring_address:
            self.active_ring_address = address
            self.active_ring_name = self.ring_name(address)

        acc = message.get("acc")
        gyr = message.get("gyr")
        if not isinstance(acc, list) or not isinstance(gyr, list) or len(acc) != 3 or len(gyr) != 3:
            return
        self.current_samples.append([float(value) for value in acc] + [float(value) for value in gyr])

    def toggle_capture(self) -> None:
        if self.recording:
            self.finish_capture()
        else:
            self.start_capture()

    def start_capture(self) -> None:
        if not self.active_ring_address and self.rings:
            self.active_ring_address = next(iter(self.rings), "")
            self.active_ring_name = self.ring_name(self.active_ring_address)
        self.recording = True
        self.current_samples.clear()
        self.started_at = time.time()
        self.result_label.setText("")
        self.elapsed_value.setText("-- ms")
        self.confidence_value.setText("--")
        self.set_recording_state()

    def finish_capture(self) -> None:
        self.recording = False
        values = np.asarray(self.current_samples, dtype=np.float32)
        sample_count = int(values.shape[0])
        capture_ms = max((time.time() - self.started_at) * 1000.0, 0.0)
        self.current_samples.clear()
        self.set_waiting_state()

        if sample_count <= 0 or not self.runtime_ready:
            self.result_label.setText("-")
            return

        self.latest_job_id += 1
        self.worker.submit(
            InferenceJob(
                job_id=self.latest_job_id,
                mode_key=self.mode_key,
                values=values,
                sample_count=sample_count,
                capture_ms=capture_ms,
            )
        )

    def on_worker_ready(self) -> None:
        self.runtime_ready = True

    def on_worker_error(self, _message: str) -> None:
        self.runtime_ready = False
        self.result_label.setText("-")

    def on_inference_result(self, result: InferenceResult) -> None:
        if result.job_id != self.latest_job_id:
            return
        self.result_label.setText(result.text or "-")
        self.elapsed_value.setText(f"{result.inference_ms:.0f} ms")
        self.confidence_value.setText(f"{result.confidence:.3f}")
        if result.text and not result.error:
            self.add_history_item(result.mode_key, result.text)

    def set_mode(self, mode_key: str) -> None:
        if self.recording:
            self.mode_buttons[self.mode_key].setChecked(True)
            return
        if mode_key == self.mode_key:
            self.refresh_history()
            return
        self.mode_key = mode_key
        self.clear_all_history()
        self.result_label.setText("-")
        self.elapsed_value.setText("-- ms")
        self.confidence_value.setText("--")
        self.refresh_history()

    def add_history_item(self, mode_key: str, text: str) -> None:
        history = self.history_by_mode.setdefault(mode_key, [])
        history.append(text)
        del history[:-18]
        if mode_key == self.mode_key:
            self.refresh_history()

    def refresh_history(self) -> None:
        items = self.history_by_mode.get(self.mode_key, [])
        if self.mode_key == "digits":
            text = "  ".join(items)
        else:
            text = " ".join(items)
        self.history_sentence_label.setText(text if text else tr("history_empty"))

    def clear_all_history(self) -> None:
        for history in self.history_by_mode.values():
            history.clear()

    def refresh_capture_state(self) -> None:
        if not self.recording:
            return
        self.set_recording_state()

    def set_waiting_state(self) -> None:
        self.status_dot.setStyleSheet(
            "background-color: #d64545; border-radius: 13px; border: 4px solid #f7dada;"
        )
        self.state_label.setText(tr("waiting"))

    def set_recording_state(self) -> None:
        self.status_dot.setStyleSheet(
            "background-color: #2f9e60; border-radius: 13px; border: 4px solid #d9f1e4;"
        )
        self.state_label.setText(tr("recording"))

    def ring_name(self, address: str) -> str:
        ring = self.rings.get(address)
        if ring is not None:
            name = str(ring.get("name", "")).strip()
            if name:
                return name
        return address[-8:] if address else ""

    def closeEvent(self, event) -> None:
        self.capture_timer.stop()
        self.worker.stop()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = create_base_parser("Clean handwriting recognition demo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: HandwritingDemoWindow(host=args.host, port=args.port),
        style_sheet=DEMO_STYLE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
