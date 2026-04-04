import argparse
import queue
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from common.constants import TOUCH_CODE_TO_NAME
    from common.stream_window import RingStreamWindowBase
    from common.ui_style import APP_STYLE_SHEET
except ModuleNotFoundError:
    from apps.common.constants import TOUCH_CODE_TO_NAME
    from apps.common.stream_window import RingStreamWindowBase
    from apps.common.ui_style import APP_STYLE_SHEET

try:
    from whisper.whisper_number_recognizer import WhisperNumberPrediction, WhisperNumberRecognizer

    HAS_WHISPER_RUNTIME = True
    WHISPER_RUNTIME_ERROR = ""
except Exception as exc:
    WhisperNumberPrediction = object
    WhisperNumberRecognizer = None
    HAS_WHISPER_RUNTIME = False
    WHISPER_RUNTIME_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET
CHECKPOINT_DIR = APP_ROOT / "checkpoint" / "whisper-number"


@dataclass
class CaptureState:
    ring_name: str
    recording: bool = False
    started_at: float = 0.0
    samples: list[list[float]] = field(default_factory=list)
    last_duration_ms: float = 0.0
    last_sample_count: int = 0
    last_result: str = "-"


@dataclass(frozen=True)
class InferenceJob:
    job_id: int
    address: str
    ring_name: str
    values: np.ndarray
    capture_ms: float
    sample_count: int


@dataclass(frozen=True)
class InferenceResult:
    job_id: int
    address: str
    ring_name: str
    capture_ms: float
    sample_count: int
    prediction: WhisperNumberPrediction
    elapsed_ms: float


@dataclass(frozen=True)
class InferenceFailure:
    job_id: int
    address: str
    ring_name: str
    error: str


class WhisperInferenceWorker(QThread):
    status_changed = pyqtSignal(str)
    ready = pyqtSignal()
    result_ready = pyqtSignal(object)
    job_failed = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self._jobs: queue.Queue[InferenceJob | None] = queue.Queue()
        self._stopping = False

    def submit(self, job: InferenceJob) -> None:
        self._jobs.put(job)

    def stop(self) -> None:
        self._stopping = True
        self._jobs.put(None)
        self.wait(1500)

    def run(self) -> None:
        if WhisperNumberRecognizer is None:
            self.fatal_error.emit(f"推理依赖不可用: {WHISPER_RUNTIME_ERROR}")
            return

        try:
            self.status_changed.emit("状态：正在从 checkpoint/whisper-number 加载 CPU 模型")
            recognizer = WhisperNumberRecognizer(self.checkpoint_dir, device="cpu")
        except Exception as exc:
            self.fatal_error.emit(f"模型加载失败: {exc}")
            return

        self.ready.emit()
        self.status_changed.emit("状态：模型已就绪，等待触摸按下")

        while not self._stopping:
            job = self._jobs.get()
            if job is None:
                return

            self.status_changed.emit(f"状态：{job.ring_name} 预处理中（{job.sample_count} 帧）")
            started = time.perf_counter()
            try:
                prediction = recognizer.predict(job.values)
            except Exception as exc:
                self.job_failed.emit(
                    InferenceFailure(
                        job_id=job.job_id,
                        address=job.address,
                        ring_name=job.ring_name,
                        error=str(exc),
                    )
                )
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.result_ready.emit(
                InferenceResult(
                    job_id=job.job_id,
                    address=job.address,
                    ring_name=job.ring_name,
                    capture_ms=job.capture_ms,
                    sample_count=job.sample_count,
                    prediction=prediction,
                    elapsed_ms=elapsed_ms,
                )
            )
            self.status_changed.emit(f"状态：{job.ring_name} 推理完成")


class WhisperNumberTouchWindow(RingStreamWindowBase):
    TOUCH_DOWN_CODES = {2}
    TOUCH_RELEASE_CODES = {5}

    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("触摸数字书写识别")
        self.resize(980, 760)

        self.states: dict[str, CaptureState] = {}
        self.latest_job_id = 0
        self.active_address = ""
        self.runtime_ready = False
        self.runtime_failed = not HAS_WHISPER_RUNTIME

        self.stream_status_label = QLabel("数据流状态：未连接")
        self.stream_status_label.setObjectName("statusTag")
        self.workflow_label = QLabel("状态：准备启动")
        self.workflow_label.setWordWrap(True)
        self.device_label = QLabel("推理设备：CPU | 模型：checkpoint/whisper-number | 加载中")
        self.capture_label = QLabel("当前录制：无")
        self.result_label = QLabel("--")
        self.result_label.setProperty("class", "title")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result_meta_label = QLabel("最近结果：无")
        self.result_meta_label.setWordWrap(True)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("识别记录会显示在这里")
        self.log_output.document().setMaximumBlockCount(200)

        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(200)
        self.capture_timer.timeout.connect(self.refresh_capture_status)

        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self._bind_signals()
        self.capture_timer.start()

        self.worker: WhisperInferenceWorker | None = None
        if HAS_WHISPER_RUNTIME:
            self.worker = WhisperInferenceWorker(CHECKPOINT_DIR)
            self.worker.status_changed.connect(self.set_runtime_status)
            self.worker.ready.connect(self.on_worker_ready)
            self.worker.result_ready.connect(self.on_inference_result)
            self.worker.job_failed.connect(self.on_inference_failed)
            self.worker.fatal_error.connect(self.on_worker_fatal_error)
            self.worker.start()
        else:
            self.device_label.setText("推理设备：CPU | 模型：不可用")
            self.set_runtime_status(f"状态：推理依赖不可用（{WHISPER_RUNTIME_ERROR}）")

        self.client.start()

    def _build_ui(self) -> None:
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

        title = QLabel("触摸数字书写识别")
        title.setProperty("class", "title")
        desc = QLabel("按下触摸按键开始缓存 IMU，松开时触发一次 CPU 推理，模型来自 checkpoint/whisper-number。")
        desc.setProperty("class", "sub-text")

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.stream_status_label)
        panel_layout.addWidget(self.workflow_label)
        panel_layout.addWidget(self.device_label)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择订阅戒指（不勾选表示接收全部）",
            max_height=160,
        )

        actions = QHBoxLayout()
        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        self.clear_btn = QPushButton("清空结果")
        self.clear_btn.setProperty("class", "btn-primary")
        actions.addWidget(self.apply_btn)
        actions.addWidget(self.clear_btn)
        actions.addStretch()
        panel_layout.addLayout(actions)

        panel_layout.addWidget(self.capture_label)

        result_panel = QFrame()
        result_panel.setProperty("class", "panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(14, 14, 14, 14)
        result_layout.setSpacing(8)

        result_title = QLabel("最新识别结果")
        result_title.setProperty("class", "title")
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_meta_label)
        result_layout.addWidget(self.log_output)

        root.addWidget(panel)
        root.addWidget(result_panel, 1)

    def _bind_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.clear_btn.clicked.connect(self.clear_results)
        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> None:
        _ = rings
        for address in set(removed_addresses + changed_connection_addresses):
            ring = self.ring_info.get(address)
            connected = bool(ring.get("connected", False)) if ring is not None else False
            if connected:
                continue
            state = self.states.get(address)
            if state is None or not state.recording:
                continue
            state.recording = False
            state.samples.clear()
            if self.active_address == address:
                self.active_address = ""
                self.capture_label.setText("当前录制：无")
            self.set_runtime_status(f"状态：{state.ring_name} 已离线，当前录制已取消")

    def handle_stream_message(self, message: dict) -> None:
        message_type = str(message.get("type", ""))
        if message_type == "imu":
            self.handle_imu_message(message)
        elif message_type == "touch":
            self.handle_touch_message(message)

    def handle_imu_message(self, message: dict) -> None:
        address = str(message.get("address", ""))
        if not address:
            return
        state = self.state_for(address)
        if not state.recording:
            return

        acc = message.get("acc")
        gyr = message.get("gyr")
        if not isinstance(acc, list) or not isinstance(gyr, list):
            return
        if len(acc) != 3 or len(gyr) != 3:
            return

        sample = [float(value) for value in acc] + [float(value) for value in gyr]
        state.samples.append(sample)
        state.last_sample_count = len(state.samples)

    def handle_touch_message(self, message: dict) -> None:
        address = str(message.get("address", ""))
        if not address:
            return

        action_code = int(message.get("action_code", -1) or -1)
        host_timestamp = float(message.get("host_timestamp", 0.0) or time.time())
        if action_code in self.TOUCH_DOWN_CODES:
            self.start_capture(address, host_timestamp)
            return
        if action_code in self.TOUCH_RELEASE_CODES:
            self.finish_capture(
                address,
                host_timestamp,
                TOUCH_CODE_TO_NAME.get(action_code, f"code={action_code}"),
            )

    def apply_subscription(self) -> None:
        self.send_subscription(["imu", "touch"])
        if self.runtime_failed:
            self.set_runtime_status("状态：已订阅 IMU + touch，但推理不可用")
        elif self.runtime_ready:
            self.set_runtime_status("状态：已订阅 IMU + touch，等待触摸按下")
        else:
            self.set_runtime_status("状态：已订阅 IMU + touch，等待模型加载")

    def start_capture(self, address: str, host_timestamp: float) -> None:
        state = self.state_for(address)
        state.recording = True
        state.started_at = host_timestamp if host_timestamp > 0 else time.time()
        state.samples.clear()
        state.last_sample_count = 0
        self.active_address = address
        self.capture_label.setText(f"当前录制：{state.ring_name} | 0 帧 | 0.00 s")
        self.set_runtime_status(f"状态：{state.ring_name} 触摸按下，开始缓存 IMU")
        self.append_log(f"{state.ring_name} touch-down -> 开始录制")

    def finish_capture(self, address: str, host_timestamp: float, release_name: str) -> None:
        state = self.state_for(address)
        if not state.recording:
            return

        state.recording = False
        end_timestamp = host_timestamp if host_timestamp > 0 else time.time()
        state.last_duration_ms = max((end_timestamp - state.started_at) * 1000.0, 0.0)
        state.last_sample_count = len(state.samples)
        self.active_address = ""

        capture_seconds = state.last_duration_ms / 1000.0
        self.capture_label.setText(
            f"最近录制：{state.ring_name} | {state.last_sample_count} 帧 | {capture_seconds:.2f} s"
        )

        if state.last_sample_count <= 0:
            self.set_runtime_status(f"状态：{state.ring_name} {release_name}，但没有采到 IMU")
            self.append_log(f"{state.ring_name} {release_name} -> 0 帧，已忽略")
            state.samples.clear()
            return

        if self.worker is None or self.runtime_failed:
            self.set_runtime_status(f"状态：{state.ring_name} {release_name}，但推理不可用")
            self.append_log(f"{state.ring_name} {release_name} -> 推理不可用")
            state.samples.clear()
            return

        self.latest_job_id += 1
        values = np.asarray(state.samples, dtype=np.float32)
        state.samples.clear()
        self.worker.submit(
            InferenceJob(
                job_id=self.latest_job_id,
                address=address,
                ring_name=state.ring_name,
                values=values,
                capture_ms=state.last_duration_ms,
                sample_count=state.last_sample_count,
            )
        )
        self.set_runtime_status(
            f"状态：{state.ring_name} {release_name}，已提交 {state.last_sample_count} 帧到 CPU 推理"
        )
        self.append_log(
            f"{state.ring_name} {release_name} -> 提交推理，{state.last_sample_count} 帧，{capture_seconds:.2f} s"
        )

    def on_worker_ready(self) -> None:
        self.runtime_ready = True
        self.device_label.setText("推理设备：CPU | 模型：checkpoint/whisper-number | 已就绪")

    def on_worker_fatal_error(self, message: str) -> None:
        self.runtime_failed = True
        self.runtime_ready = False
        self.device_label.setText("推理设备：CPU | 模型：加载失败")
        self.set_runtime_status(f"状态：{message}")
        self.append_log(message)

    def on_inference_result(self, result: InferenceResult) -> None:
        prediction_text = result.prediction.text or "(空结果)"
        resample_note = " | 已重采样" if result.prediction.was_resampled else ""

        state = self.state_for(result.address)
        state.last_result = prediction_text

        self.result_label.setText(prediction_text)
        self.result_meta_label.setText(
            (
                f"{result.ring_name} | {result.sample_count} 帧 | "
                f"录制 {result.capture_ms / 1000.0:.2f} s | 推理 {result.elapsed_ms:.1f} ms | "
                f"置信度 {result.prediction.average_confidence:.3f}{resample_note}"
            )
        )
        self.set_runtime_status(f"状态：{result.ring_name} 识别完成 -> {prediction_text}")
        self.append_log(
            (
                f"{result.ring_name} -> {prediction_text} | "
                f"frames={result.sample_count} infer={result.elapsed_ms:.1f}ms "
                f"conf={result.prediction.average_confidence:.3f}{resample_note}"
            )
        )

    def on_inference_failed(self, failure: InferenceFailure) -> None:
        self.set_runtime_status(f"状态：{failure.ring_name} 推理失败（{failure.error}）")
        self.append_log(f"{failure.ring_name} 推理失败: {failure.error}")

    def refresh_capture_status(self) -> None:
        if not self.active_address:
            return
        state = self.states.get(self.active_address)
        if state is None or not state.recording:
            return
        elapsed = max(time.time() - state.started_at, 0.0)
        self.capture_label.setText(
            f"当前录制：{state.ring_name} | {len(state.samples)} 帧 | {elapsed:.2f} s"
        )

    def clear_results(self) -> None:
        self.result_label.setText("--")
        self.result_meta_label.setText("最近结果：无")
        self.log_output.clear()
        self.capture_label.setText("当前录制：无")
        self.set_runtime_status("状态：结果已清空，等待触摸按下")

    def set_runtime_status(self, text: str) -> None:
        self.workflow_label.setText(text)

    def state_for(self, address: str) -> CaptureState:
        state = self.states.get(address)
        ring_name = self.ring_name(address)
        if state is None:
            state = CaptureState(ring_name=ring_name)
            self.states[address] = state
        elif state.ring_name != ring_name:
            state.ring_name = ring_name
        return state

    def ring_name(self, address: str) -> str:
        ring = self.ring_info.get(address)
        if ring is not None:
            name = str(ring.get("name", "")).strip()
            if name:
                return name
        if address:
            return address[-8:]
        return "Unknown"

    def append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
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
    win = WhisperNumberTouchWindow(host=args.host, port=args.port)
    win.show()
    sys.exit(app.exec())
