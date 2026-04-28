from __future__ import annotations
"""Shared touch-to-Whisper window runtime used by multiple apps."""

import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from apps.framework.stream_window import RingStreamWindowBase
from apps.shared.constants import TOUCH_CODE_TO_NAME


# Change this single value to switch this realtime handwriting UI: "zh" or "en".
UI_LANGUAGE = "en"

UI_TEXT = {
    "zh": {
        "runtime_unavailable": "推理依赖不可用: {error}",
        "app_window_title": "手写识别",
        "app_title": "手写识别",
        "app_description": "选择输出类型；按空格开始录制，再按空格结束并识别。",
        "mode_number": "数字",
        "mode_lowercase": "英文小写",
        "loading_mode_model": "状态：正在加载 {mode} 模型（{model}）",
        "model_load_failed": "模型加载失败: {error}",
        "ready_waiting": "状态：模型已就绪，等待空格开始",
        "preprocessing": "状态：{ring} | {mode} 预处理中（{samples} 帧）",
        "inference_done_status": "状态：{ring} | {mode} 推理完成",
        "stream_disconnected": "数据流状态：未连接",
        "stream_connected": "数据流状态：已连接",
        "stream_error": "数据流状态：异常（{message}）",
        "workflow_initial": "状态：准备启动",
        "device_loading": "推理设备：CPU | 模型：{model} | 加载中",
        "capture_none": "当前录制：无",
        "last_result_none": "最近结果：无",
        "log_placeholder": "识别记录会显示在这里",
        "model_unavailable": "推理设备：CPU | 模型：不可用",
        "runtime_unavailable_status": "状态：推理依赖不可用（{error}）",
        "mode_label": "输出类型",
        "ring_selection": "选择订阅戒指（不勾选表示接收全部）",
        "apply_subscription": "应用订阅",
        "clear_results": "清空结果",
        "latest_result": "最新识别结果",
        "ready": "已就绪",
        "loading": "加载中",
        "unavailable": "不可用",
        "mode_changed": "状态：已切换到 {mode}",
        "offline_cancelled": "状态：{ring} 已离线，当前录制已取消",
        "subscribed_unavailable": "状态：已订阅 IMU + touch，但推理不可用",
        "subscribed_ready": "状态：已订阅 IMU + touch，等待空格开始",
        "subscribed_loading": "状态：已订阅 IMU + touch，等待模型加载",
        "select_one_ring": "状态：请先勾选一个戒指，再按空格开始",
        "space_down": "Space",
        "space_up": "Space",
        "touch_down": "touch-down",
        "capture_started": "状态：{ring} | {mode} {trigger}，开始缓存 IMU",
        "capture_started_log": "{ring} | {mode} {trigger} -> 开始录制",
        "capture_current": "当前录制：{ring} | {samples} 帧 | {seconds:.2f} s",
        "capture_last": "最近录制：{ring} | {samples} 帧 | {seconds:.2f} s",
        "no_imu": "状态：{ring} {release}，但没有采到 IMU",
        "ignored_zero": "{ring} {release} -> 0 帧，已忽略",
        "inference_unavailable": "状态：{ring} {release}，但推理不可用",
        "inference_unavailable_log": "{ring} {release} -> 推理不可用",
        "submitted": "状态：{ring} | {mode} {release}，已提交 {samples} 帧到 CPU 推理",
        "submitted_log": "{ring} | {mode} {release} -> 提交推理，{samples} 帧，{seconds:.2f} s",
        "device_ready": "推理设备：CPU | 模型：{model} | 已就绪",
        "device_failed": "推理设备：CPU | 模型：加载失败",
        "empty_result": "(空结果)",
        "resampled": " | 已重采样",
        "result_meta": "{ring} | {mode} | {samples} 帧 | 录制 {capture_s:.2f} s | 推理 {infer_ms:.1f} ms | 置信度 {confidence:.3f}{resample_note}",
        "recognized": "状态：{ring} | {mode} 识别完成 -> {prediction}",
        "result_log": "{ring} | {mode} -> {prediction} | frames={samples} infer={infer_ms:.1f}ms conf={confidence:.3f}{resample_note}",
        "failed": "状态：{ring} | {mode} 推理失败（{error}）",
        "failed_log": "{ring} | {mode} 推理失败: {error}",
        "cleared": "状态：结果已清空，等待空格开始",
        "unknown": "Unknown",
        "ring_online": "在线",
        "ring_offline": "离线",
    },
    "en": {
        "runtime_unavailable": "Inference runtime unavailable: {error}",
        "app_window_title": "Handwriting Recognition",
        "app_title": "Handwriting Recognition",
        "app_description": "Choose output type. Press Space to start recording, then press Space again to recognize.",
        "mode_number": "Digits",
        "mode_lowercase": "Lowercase",
        "loading_mode_model": "Status: loading {mode} model ({model})",
        "model_load_failed": "Model load failed: {error}",
        "ready_waiting": "Status: ready, press Space to start",
        "preprocessing": "Status: {ring} | {mode} preprocessing ({samples} frames)",
        "inference_done_status": "Status: {ring} | {mode} inference complete",
        "stream_disconnected": "Stream: disconnected",
        "stream_connected": "Stream: connected",
        "stream_error": "Stream: error ({message})",
        "workflow_initial": "Status: starting",
        "device_loading": "Device: CPU | Model: {model} | loading",
        "capture_none": "Capture: none",
        "last_result_none": "Last result: none",
        "log_placeholder": "Recognition history",
        "model_unavailable": "Device: CPU | Model: unavailable",
        "runtime_unavailable_status": "Status: inference runtime unavailable ({error})",
        "mode_label": "Output",
        "ring_selection": "Rings (select one for Space capture)",
        "apply_subscription": "Apply rings",
        "clear_results": "Clear",
        "latest_result": "Latest result",
        "ready": "ready",
        "loading": "loading",
        "unavailable": "unavailable",
        "mode_changed": "Status: switched to {mode}",
        "offline_cancelled": "Status: {ring} went offline; capture cancelled",
        "subscribed_unavailable": "Status: subscribed to IMU + touch, inference unavailable",
        "subscribed_ready": "Status: subscribed to IMU + touch, press Space to start",
        "subscribed_loading": "Status: subscribed to IMU + touch, loading model",
        "select_one_ring": "Status: select one ring before pressing Space",
        "space_down": "Space",
        "space_up": "Space",
        "touch_down": "touch-down",
        "capture_started": "Status: {ring} | {mode} {trigger}, capturing IMU",
        "capture_started_log": "{ring} | {mode} {trigger} -> capture started",
        "capture_current": "Capture: {ring} | {samples} frames | {seconds:.2f} s",
        "capture_last": "Last capture: {ring} | {samples} frames | {seconds:.2f} s",
        "no_imu": "Status: {ring} {release}, no IMU samples",
        "ignored_zero": "{ring} {release} -> 0 frames, ignored",
        "inference_unavailable": "Status: {ring} {release}, inference unavailable",
        "inference_unavailable_log": "{ring} {release} -> inference unavailable",
        "submitted": "Status: {ring} | {mode} {release}, submitted {samples} frames to CPU",
        "submitted_log": "{ring} | {mode} {release} -> submitted, {samples} frames, {seconds:.2f} s",
        "device_ready": "Device: CPU | Model: {model} | ready",
        "device_failed": "Device: CPU | Model: load failed",
        "empty_result": "(empty)",
        "resampled": " | resampled",
        "result_meta": "{ring} | {mode} | {samples} frames | capture {capture_s:.2f} s | inference {infer_ms:.1f} ms | confidence {confidence:.3f}{resample_note}",
        "recognized": "Status: {ring} | {mode} recognized -> {prediction}",
        "result_log": "{ring} | {mode} -> {prediction} | frames={samples} infer={infer_ms:.1f}ms conf={confidence:.3f}{resample_note}",
        "failed": "Status: {ring} | {mode} inference failed ({error})",
        "failed_log": "{ring} | {mode} inference failed: {error}",
        "cleared": "Status: cleared, press Space to start",
        "unknown": "Unknown",
        "ring_online": "online",
        "ring_offline": "offline",
    },
}

if UI_LANGUAGE not in UI_TEXT:
    raise ValueError(f"Unsupported UI_LANGUAGE: {UI_LANGUAGE!r}")


def tr(key: str, **kwargs: Any) -> str:
    text = UI_TEXT[UI_LANGUAGE][key]
    return text.format(**kwargs) if kwargs else text


@dataclass(frozen=True)
class WhisperTouchModeSpec:
    key: str
    label: str
    model_label: str
    checkpoint_dir: Path
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class WhisperTouchAppSpec:
    window_title: str
    title: str
    description: str
    modes: tuple[WhisperTouchModeSpec, ...]
    default_mode_key: str


@dataclass
class CaptureState:
    ring_name: str
    recording: bool = False
    started_at: float = 0.0
    samples: list[list[float]] = field(default_factory=list)
    last_duration_ms: float = 0.0
    last_sample_count: int = 0
    last_result: str = "-"
    mode_key: str = ""
    mode_label: str = ""


@dataclass(frozen=True)
class InferenceJob:
    job_id: int
    address: str
    ring_name: str
    mode_key: str
    mode_label: str
    values: np.ndarray
    capture_ms: float
    sample_count: int


@dataclass(frozen=True)
class InferenceResult:
    job_id: int
    address: str
    ring_name: str
    mode_key: str
    mode_label: str
    capture_ms: float
    sample_count: int
    prediction: Any
    elapsed_ms: float


@dataclass(frozen=True)
class InferenceFailure:
    job_id: int
    address: str
    ring_name: str
    mode_key: str
    mode_label: str
    error: str


class WhisperInferenceWorker(QThread):
    """Background inference worker so capture UI stays simple and responsive."""

    status_changed = pyqtSignal(str)
    ready = pyqtSignal()
    result_ready = pyqtSignal(object)
    job_failed = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(
        self,
        *,
        mode_specs: tuple[WhisperTouchModeSpec, ...],
        recognizer_cls: type[Any] | None,
        runtime_error: str,
    ):
        super().__init__()
        self.mode_specs = mode_specs
        self.recognizer_cls = recognizer_cls
        self.runtime_error = runtime_error
        self._jobs: queue.Queue[InferenceJob | None] = queue.Queue()
        self._stopping = False

    def submit(self, job: InferenceJob) -> None:
        self._jobs.put(job)

    def stop(self) -> None:
        self._stopping = True
        self._jobs.put(None)
        self.wait(1500)

    def run(self) -> None:
        if self.recognizer_cls is None:
            self.fatal_error.emit(tr("runtime_unavailable", error=self.runtime_error))
            return

        recognizers: dict[str, Any] = {}
        try:
            for mode in self.mode_specs:
                self.status_changed.emit(tr("loading_mode_model", mode=mode.label, model=mode.model_label))
                if mode.checkpoint_path is None:
                    recognizers[mode.key] = self.recognizer_cls(mode.checkpoint_dir, device="cpu")
                else:
                    recognizers[mode.key] = self.recognizer_cls(
                        mode.checkpoint_dir,
                        device="cpu",
                        checkpoint_path=mode.checkpoint_path,
                    )
        except Exception as exc:
            self.fatal_error.emit(tr("model_load_failed", error=exc))
            return

        self.ready.emit()
        self.status_changed.emit(tr("ready_waiting"))

        while not self._stopping:
            job = self._jobs.get()
            if job is None:
                return

            self.status_changed.emit(tr("preprocessing", ring=job.ring_name, mode=job.mode_label, samples=job.sample_count))
            started = time.perf_counter()
            try:
                recognizer = recognizers[job.mode_key]
                prediction = recognizer.predict(job.values)
            except Exception as exc:
                self.job_failed.emit(
                    InferenceFailure(
                        job_id=job.job_id,
                        address=job.address,
                        ring_name=job.ring_name,
                        mode_key=job.mode_key,
                        mode_label=job.mode_label,
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
                    mode_key=job.mode_key,
                    mode_label=job.mode_label,
                    capture_ms=job.capture_ms,
                    sample_count=job.sample_count,
                    prediction=prediction,
                    elapsed_ms=elapsed_ms,
                )
            )
            self.status_changed.emit(tr("inference_done_status", ring=job.ring_name, mode=job.mode_label))


class WhisperTouchWindowBase(RingStreamWindowBase):
    """Base window that only manages capture flow and delegates model specifics to spec/runtime."""

    TOUCH_DOWN_CODES = {1}
    TOUCH_RELEASE_CODES = {5}

    def __init__(
        self,
        *,
        host: str,
        port: int,
        spec: WhisperTouchAppSpec,
        recognizer_cls: type[Any] | None,
        runtime_available: bool,
        runtime_error: str,
    ):
        super().__init__(host=host, port=port)
        self.spec = spec
        self.runtime_error = runtime_error
        self.mode_by_key = {mode.key: mode for mode in spec.modes}
        if spec.default_mode_key not in self.mode_by_key:
            raise ValueError(f"unknown default Whisper touch mode: {spec.default_mode_key}")
        self.setWindowTitle(spec.window_title)
        self.resize(980, 760)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.states: dict[str, CaptureState] = {}
        self.latest_job_id = 0
        self.active_address = ""
        self.runtime_ready = False
        self.runtime_failed = not runtime_available
        self.active_mode_key = spec.default_mode_key

        self.stream_status_label = QLabel(tr("stream_disconnected"))
        self.stream_status_label.setObjectName("statusTag")
        self.workflow_label = QLabel(tr("workflow_initial"))
        self.workflow_label.setWordWrap(True)
        self.device_label = QLabel(
            tr("device_loading", model=self.current_mode().model_label)
        )
        self.mode_combo = QComboBox()
        for mode in self.spec.modes:
            self.mode_combo.addItem(mode.label, mode.key)
        default_index = max(self.mode_combo.findData(spec.default_mode_key), 0)
        self.mode_combo.setCurrentIndex(default_index)
        self.capture_label = QLabel(tr("capture_none"))
        self.result_label = QLabel("--")
        self.result_label.setProperty("class", "title")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.result_meta_label = QLabel(tr("last_result_none"))
        self.result_meta_label.setWordWrap(True)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(tr("log_placeholder"))
        self.log_output.document().setMaximumBlockCount(200)

        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(200)
        self.capture_timer.timeout.connect(self.refresh_capture_status)
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_space_capture)

        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self._bind_signals()
        self.capture_timer.start()

        self.worker: WhisperInferenceWorker | None = None
        if runtime_available:
            self.worker = WhisperInferenceWorker(
                mode_specs=spec.modes,
                recognizer_cls=recognizer_cls,
                runtime_error=runtime_error,
            )
            self.worker.status_changed.connect(self.set_runtime_status)
            self.worker.ready.connect(self.on_worker_ready)
            self.worker.result_ready.connect(self.on_inference_result)
            self.worker.job_failed.connect(self.on_inference_failed)
            self.worker.fatal_error.connect(self.on_worker_fatal_error)
            self.worker.start()
        else:
            self.device_label.setText(tr("model_unavailable"))
            self.set_runtime_status(tr("runtime_unavailable_status", error=runtime_error))

        self.client.start()
        self.setFocus()

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

        title = QLabel(self.spec.title)
        title.setProperty("class", "title")
        desc = QLabel(self.spec.description)
        desc.setProperty("class", "sub-text")

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        panel_layout.addWidget(self.stream_status_label)
        panel_layout.addWidget(self.workflow_label)
        panel_layout.addWidget(self.device_label)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(tr("mode_label")))
        mode_row.addWidget(self.mode_combo, 1)
        panel_layout.addLayout(mode_row)

        self.add_ring_selection_panel(
            panel_layout,
            title_text=tr("ring_selection"),
            max_height=160,
        )

        actions = QHBoxLayout()
        self.apply_btn = QPushButton(tr("apply_subscription"))
        self.apply_btn.setProperty("class", "btn-primary")
        self.clear_btn = QPushButton(tr("clear_results"))
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

        result_title = QLabel(tr("latest_result"))
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
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.bind_stream_client_signals()

    def current_mode(self) -> WhisperTouchModeSpec:
        key = str(self.mode_combo.currentData()) if hasattr(self, "mode_combo") else self.active_mode_key
        return self.mode_by_key.get(key, self.mode_by_key[self.active_mode_key])

    def set_connection_status(self, connected: bool) -> None:
        if self.status_label is not None:
            self.status_label.setText(tr("stream_connected" if connected else "stream_disconnected"))

    def set_stream_error(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(tr("stream_error", message=message))

    def format_ring_text(self, ring: dict[str, Any]) -> str:
        ring_name = str(ring.get("name", tr("unknown")))
        address = str(ring.get("address", ""))
        connected = bool(ring.get("connected", False))
        return f"{ring_name} | {address} | {tr('ring_online' if connected else 'ring_offline')}"

    def on_mode_changed(self) -> None:
        self.active_mode_key = str(self.mode_combo.currentData())
        mode = self.current_mode()
        ready_text = tr("ready") if self.runtime_ready else tr("loading")
        if self.runtime_failed:
            ready_text = tr("unavailable")
        self.device_label.setText(f"{tr('device_loading', model=mode.model_label).rsplit('|', 1)[0]}| {ready_text}")
        self.result_label.setText("--")
        self.result_meta_label.setText(tr("last_result_none"))
        self.set_runtime_status(tr("mode_changed", mode=mode.label))

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def toggle_space_capture(self) -> None:
        if self.active_address:
            self.finish_capture(self.active_address, time.time(), tr("space_up"))
            return

        address = self.capture_address_for_space()
        if not address:
            self.set_runtime_status(tr("select_one_ring"))
            return
        self.start_capture(address, time.time(), tr("space_down"))

    def capture_address_for_space(self) -> str:
        selected = self.selected_ring_addresses()
        if selected:
            return selected[0]

        connected = [
            address
            for address, ring in self.ring_info.items()
            if bool(ring.get("connected", False))
        ]
        if len(connected) == 1:
            return connected[0]
        return ""

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
            connected = (
                bool(ring.get("connected", False)) if ring is not None else False
            )
            if connected:
                continue
            state = self.states.get(address)
            if state is None or not state.recording:
                continue
            state.recording = False
            state.samples.clear()
            if self.active_address == address:
                self.active_address = ""
                self.capture_label.setText(tr("capture_none"))
                self.mode_combo.setEnabled(True)
            self.set_runtime_status(tr("offline_cancelled", ring=state.ring_name))

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
            self.set_runtime_status(tr("subscribed_unavailable"))
        elif self.runtime_ready:
            self.set_runtime_status(tr("subscribed_ready"))
        else:
            self.set_runtime_status(tr("subscribed_loading"))

    def start_capture(self, address: str, host_timestamp: float, trigger_name: str | None = None) -> None:
        state = self.state_for(address)
        mode = self.current_mode()
        trigger_name = trigger_name or tr("touch_down")
        state.recording = True
        state.started_at = host_timestamp if host_timestamp > 0 else time.time()
        state.samples.clear()
        state.last_sample_count = 0
        state.mode_key = mode.key
        state.mode_label = mode.label
        self.active_address = address
        self.mode_combo.setEnabled(False)
        self.capture_label.setText(tr("capture_current", ring=state.ring_name, samples=0, seconds=0.0))
        self.set_runtime_status(tr("capture_started", ring=state.ring_name, mode=mode.label, trigger=trigger_name))
        self.append_log(tr("capture_started_log", ring=state.ring_name, mode=mode.label, trigger=trigger_name))

    def finish_capture(
        self,
        address: str,
        host_timestamp: float,
        release_name: str,
    ) -> None:
        state = self.state_for(address)
        if not state.recording:
            return

        state.recording = False
        end_timestamp = host_timestamp if host_timestamp > 0 else time.time()
        state.last_duration_ms = max((end_timestamp - state.started_at) * 1000.0, 0.0)
        state.last_sample_count = len(state.samples)
        self.active_address = ""
        self.mode_combo.setEnabled(True)

        capture_seconds = state.last_duration_ms / 1000.0
        self.capture_label.setText(
            tr("capture_last", ring=state.ring_name, samples=state.last_sample_count, seconds=capture_seconds)
        )

        if state.last_sample_count <= 0:
            self.set_runtime_status(
                tr("no_imu", ring=state.ring_name, release=release_name)
            )
            self.append_log(tr("ignored_zero", ring=state.ring_name, release=release_name))
            state.samples.clear()
            return

        if self.worker is None or self.runtime_failed:
            self.set_runtime_status(
                tr("inference_unavailable", ring=state.ring_name, release=release_name)
            )
            self.append_log(tr("inference_unavailable_log", ring=state.ring_name, release=release_name))
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
                mode_key=state.mode_key or self.current_mode().key,
                mode_label=state.mode_label or self.current_mode().label,
                values=values,
                capture_ms=state.last_duration_ms,
                sample_count=state.last_sample_count,
            )
        )
        self.set_runtime_status(
            tr(
                "submitted",
                ring=state.ring_name,
                mode=state.mode_label,
                release=release_name,
                samples=state.last_sample_count,
            )
        )
        self.append_log(
            tr(
                "submitted_log",
                ring=state.ring_name,
                mode=state.mode_label,
                release=release_name,
                samples=state.last_sample_count,
                seconds=capture_seconds,
            )
        )

    def on_worker_ready(self) -> None:
        self.runtime_ready = True
        self.device_label.setText(
            tr("device_ready", model=self.current_mode().model_label)
        )

    def on_worker_fatal_error(self, message: str) -> None:
        self.runtime_failed = True
        self.runtime_ready = False
        self.device_label.setText(tr("device_failed"))
        self.set_runtime_status(message)
        self.append_log(message)

    def on_inference_result(self, result: InferenceResult) -> None:
        prediction_text = result.prediction.text or tr("empty_result")
        resample_note = tr("resampled") if result.prediction.was_resampled else ""

        state = self.state_for(result.address)
        state.last_result = prediction_text

        self.result_label.setText(prediction_text)
        self.result_meta_label.setText(
            tr(
                "result_meta",
                ring=result.ring_name,
                mode=result.mode_label,
                samples=result.sample_count,
                capture_s=result.capture_ms / 1000.0,
                infer_ms=result.elapsed_ms,
                confidence=result.prediction.average_confidence,
                resample_note=resample_note,
            )
        )
        self.set_runtime_status(
            tr("recognized", ring=result.ring_name, mode=result.mode_label, prediction=prediction_text)
        )
        self.append_log(
            tr(
                "result_log",
                ring=result.ring_name,
                mode=result.mode_label,
                prediction=prediction_text,
                samples=result.sample_count,
                infer_ms=result.elapsed_ms,
                confidence=result.prediction.average_confidence,
                resample_note=resample_note,
            )
        )

    def on_inference_failed(self, failure: InferenceFailure) -> None:
        self.set_runtime_status(
            tr("failed", ring=failure.ring_name, mode=failure.mode_label, error=failure.error)
        )
        self.append_log(tr("failed_log", ring=failure.ring_name, mode=failure.mode_label, error=failure.error))

    def refresh_capture_status(self) -> None:
        if not self.active_address:
            return
        state = self.states.get(self.active_address)
        if state is None or not state.recording:
            return
        elapsed = max(time.time() - state.started_at, 0.0)
        self.capture_label.setText(
            tr("capture_current", ring=state.ring_name, samples=len(state.samples), seconds=elapsed)
        )

    def clear_results(self) -> None:
        self.result_label.setText("--")
        self.result_meta_label.setText(tr("last_result_none"))
        self.log_output.clear()
        self.capture_label.setText(tr("capture_none"))
        self.set_runtime_status(tr("cleared"))

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
        return tr("unknown")

    def append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.client.stop()
        self.client.wait(1500)
        event.accept()
