from __future__ import annotations

import argparse
import math
import queue
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    from common.stream_window import RingStreamWindowBase
    from common.ui_style import APP_STYLE_SHEET
except ModuleNotFoundError:
    from apps.common.stream_window import RingStreamWindowBase
    from apps.common.ui_style import APP_STYLE_SHEET

try:
    from models.pinch_cnn_recognizer import PinchCNNRecognizer, PinchPrediction

    HAS_PINCH_RUNTIME = True
    PINCH_RUNTIME_ERROR = ""
except Exception as exc:
    PinchCNNRecognizer = None
    PinchPrediction = object
    HAS_PINCH_RUNTIME = False
    PINCH_RUNTIME_ERROR = str(exc)

try:
    from pynput import keyboard

    HAS_KEYBOARD_CONTROL = True
    KEYBOARD_CONTROL_ERROR = ""
except Exception as exc:
    keyboard = None
    HAS_KEYBOARD_CONTROL = False
    KEYBOARD_CONTROL_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET
CHECKPOINT_DIR = APP_ROOT / "checkpoint" / "swipe" / "cls-pinch"
IMU_SAMPLE_RATE_HZ = 200.0
PINCH_CLASS_IDS = (0, 1, 2)
PINCH_CLASS_LABELS = {
    0: "空",
    1: "pinch-down",
    2: "pinch-up",
}
PINCH_CLASS_COLORS = {
    0: QColor("#B0B8C1"),
    1: QColor("#5B8FF9"),
    2: QColor("#61DDAA"),
}
PINCH_CLASS_TO_KEY = {}


@dataclass
class RingBufferState:
    ring_name: str
    samples: deque[list[float]] = field(default_factory=lambda: deque(maxlen=1024))
    frames_since_infer: int = 0
    prediction_history: deque[int] = field(default_factory=deque)
    committed_counts: Counter[int] = field(default_factory=Counter)
    reset_after_frames_remaining: int = 0


@dataclass(frozen=True)
class InferenceJob:
    address: str
    ring_name: str
    values: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    address: str
    ring_name: str
    prediction: PinchPrediction
    elapsed_ms: float
    sample_count: int


@dataclass(frozen=True)
class InferenceFailure:
    ring_name: str
    error: str


@dataclass(frozen=True)
class GestureDecision:
    queue_length: int
    threshold: int
    history_size: int
    class_count: int
    triggered_class_id: int | None


class PinchProbabilityChart(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._probabilities = [0.0 for _ in PINCH_CLASS_IDS]
        self.setMinimumHeight(280)

    def clear(self) -> None:
        self._probabilities = [0.0 for _ in PINCH_CLASS_IDS]
        self.update()

    def set_probabilities(self, probabilities: list[float] | tuple[float, ...] | np.ndarray) -> None:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if values.size != len(PINCH_CLASS_IDS):
            raise ValueError(
                f"概率长度必须为 {len(PINCH_CLASS_IDS)}，收到: {values.size}"
            )
        self._probabilities = [float(np.clip(value, 0.0, 1.0)) for value in values]
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(18, 18, -18, -18)
        painter.fillRect(rect, QColor("#FBFCFE"))

        plot_rect = rect.adjusted(12, 18, -12, -56)
        painter.setPen(QPen(QColor("#DCE5EE"), 1))
        for level in range(5):
            ratio = level / 4.0
            y = plot_rect.bottom() - ratio * plot_rect.height()
            painter.drawLine(int(plot_rect.left()), int(y), int(plot_rect.right()), int(y))

        painter.setPen(QColor("#7F90A4"))
        for level in range(5):
            ratio = level / 4.0
            y = plot_rect.bottom() - ratio * plot_rect.height()
            painter.drawText(rect.left(), int(y) - 2, f"{int(ratio * 100)}%")

        bar_count = len(PINCH_CLASS_IDS)
        gap = 14
        if bar_count > 1:
            max_gap = max(6, int(plot_rect.width() * 0.03))
            gap = min(gap, max_gap, max((plot_rect.width() - 18 * bar_count) // (bar_count - 1), 6))
        available_width = plot_rect.width() - gap * (bar_count - 1)
        bar_width = max(int(available_width / max(bar_count, 1)), 18)
        for index, class_id in enumerate(PINCH_CLASS_IDS):
            probability = self._probabilities[index]
            height = int(probability * plot_rect.height())
            x = int(plot_rect.left() + index * (bar_width + gap))
            y = int(plot_rect.bottom() - height)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#E8EEF5"))
            painter.drawRoundedRect(x, plot_rect.top(), bar_width, plot_rect.height(), 8, 8)

            painter.setBrush(PINCH_CLASS_COLORS[class_id])
            painter.drawRoundedRect(x, y, bar_width, height, 8, 8)

            painter.setPen(QColor("#44576A"))
            value_top = max(plot_rect.top(), y - 22)
            painter.drawText(
                x - 6,
                value_top,
                bar_width + 12,
                20,
                Qt.AlignmentFlag.AlignCenter,
                f"{probability * 100:.0f}%",
            )
            painter.drawText(
                x - 8,
                plot_rect.bottom() + 12,
                bar_width + 16,
                36,
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                f"{class_id}\n{PINCH_CLASS_LABELS[class_id]}",
            )

        painter.end()


class PinchInferenceWorker(QThread):
    ready = pyqtSignal()
    result_ready = pyqtSignal(object)
    job_failed = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self._jobs: queue.Queue[InferenceJob | None] = queue.Queue(maxsize=1)
        self._stopping = False

    def submit(self, job: InferenceJob) -> None:
        if self._jobs.full():
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
        self._jobs.put_nowait(job)

    def stop(self) -> None:
        self._stopping = True
        try:
            if self._jobs.full():
                self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._jobs.put_nowait(None)
        self.wait(1500)

    def run(self) -> None:
        try:
            recognizer = PinchCNNRecognizer(self.checkpoint_dir, device="cpu")
        except Exception as exc:
            self.fatal_error.emit(f"模型加载失败：{exc}")
            return

        self.ready.emit()

        while not self._stopping:
            job = self._jobs.get()
            if job is None:
                return

            started = time.perf_counter()
            try:
                prediction = recognizer.predict(job.values)
            except Exception as exc:
                self.job_failed.emit(InferenceFailure(ring_name=job.ring_name, error=str(exc)))
                continue

            self.result_ready.emit(
                InferenceResult(
                    address=job.address,
                    ring_name=job.ring_name,
                    prediction=prediction,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    sample_count=int(job.values.shape[0]),
                )
            )


class PinchRealtimeWindow(RingStreamWindowBase):
    DEFAULT_WINDOW_SIZE = 120
    DEFAULT_STEP_FRAMES = 5
    DEFAULT_STABLE_WINDOW_SECONDS = 0.15
    DEFAULT_P_RATIO = 0.80
    DEFAULT_RESET_DELAY_FRAMES = 20

    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("Pinch 实时检测")
        self.resize(1200, 800)

        self.states: dict[str, RingBufferState] = {}
        self.active_address = ""
        self.runtime_ready = False
        self.runtime_failed = not HAS_PINCH_RUNTIME
        self.worker_busy = False
        self.keyboard_controller = keyboard.Controller() if HAS_KEYBOARD_CONTROL else None

        self.stream_status_label = QLabel("数据流状态：未连接")
        self.stream_status_label.setObjectName("statusTag")
        self.workflow_label = QLabel("状态：准备启动")
        self.workflow_label.setWordWrap(True)
        self.device_label = QLabel("推理设备：CPU | 模型：checkpoint/swipe/cls-pinch | 加载中")
        self.active_ring_label = QLabel("当前显示：无")
        self.buffer_label = QLabel("缓冲区：0 / 100 帧")
        self.result_label = QLabel("--")
        self.result_label.setProperty("class", "title")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_meta_label = QLabel("最近结果：无")
        self.result_meta_label.setWordWrap(True)
        self.count_label = QLabel("累计计数：1=0 2=0")
        self.count_label.setWordWrap(True)
        self.chart = PinchProbabilityChart()

        self.window_size_input = QSpinBox()
        self.window_size_input.setRange(40, 400)
        self.window_size_input.setSingleStep(10)
        self.window_size_input.setValue(self.DEFAULT_WINDOW_SIZE)
        self.window_size_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.step_size_input = QSpinBox()
        self.step_size_input.setRange(1, 200)
        self.step_size_input.setSingleStep(1)
        self.step_size_input.setValue(self.DEFAULT_STEP_FRAMES)
        self.step_size_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.stable_window_input = QDoubleSpinBox()
        self.stable_window_input.setRange(0.02, 1.0)
        self.stable_window_input.setDecimals(2)
        self.stable_window_input.setSingleStep(0.02)
        self.stable_window_input.setValue(self.DEFAULT_STABLE_WINDOW_SECONDS)
        self.stable_window_input.setSuffix(" s")
        self.stable_window_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.p_ratio_input = QDoubleSpinBox()
        self.p_ratio_input.setRange(0.50, 1.00)
        self.p_ratio_input.setDecimals(2)
        self.p_ratio_input.setSingleStep(0.05)
        self.p_ratio_input.setValue(self.DEFAULT_P_RATIO)
        self.p_ratio_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.reset_delay_input = QSpinBox()
        self.reset_delay_input.setRange(1, 200)
        self.reset_delay_input.setSingleStep(1)
        self.reset_delay_input.setValue(self.DEFAULT_RESET_DELAY_FRAMES)
        self.reset_delay_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.enable_keyboard_checkbox = QCheckBox("控制键盘")
        self.enable_keyboard_checkbox.setChecked(False)
        self.enable_keyboard_checkbox.setEnabled(HAS_KEYBOARD_CONTROL)
        if HAS_KEYBOARD_CONTROL:
            self.enable_keyboard_checkbox.setToolTip("当前 pinch 类别未配置默认按键映射")
        else:
            self.enable_keyboard_checkbox.setToolTip(f"键盘控制不可用：{KEYBOARD_CONTROL_ERROR}")

        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self._bind_signals()

        self.worker: PinchInferenceWorker | None = None
        if HAS_PINCH_RUNTIME:
            self.worker = PinchInferenceWorker(CHECKPOINT_DIR)
            self.worker.ready.connect(self.on_worker_ready)
            self.worker.result_ready.connect(self.on_inference_result)
            self.worker.job_failed.connect(self.on_inference_failed)
            self.worker.fatal_error.connect(self.on_worker_fatal_error)
            self.worker.start()
        else:
            self.device_label.setText("推理设备：CPU | 模型：不可用")
            self.set_runtime_status(f"状态：推理依赖不可用（{PINCH_RUNTIME_ERROR}）")

        self.client.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        control_panel = QFrame()
        control_panel.setProperty("class", "panel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(14, 14, 14, 14)
        control_layout.setSpacing(10)

        title = QLabel("Pinch 实时检测")
        title.setProperty("class", "title")
        desc = QLabel(
            "每隔若干帧取最近一个固定窗口做一次分类。分类结果进入短队列，"
            "当某个非空类别在稳定窗口内占比超过阈值时，记为一次触发；"
            "触发后暂停若干帧，再把 IMU 缓冲收缩到最新一帧。"
        )
        desc.setProperty("class", "sub-text")
        control_layout.addWidget(title)
        control_layout.addWidget(desc)
        control_layout.addWidget(self.stream_status_label)
        control_layout.addWidget(self.workflow_label)
        control_layout.addWidget(self.device_label)
        control_layout.addWidget(self.active_ring_label)
        control_layout.addWidget(self.buffer_label)

        self.add_ring_selection_panel(
            control_layout,
            title_text="选择订阅戒指（不勾选表示接收全部）",
            max_height=160,
        )

        settings = QGridLayout()
        settings.setHorizontalSpacing(10)
        settings.setVerticalSpacing(10)
        settings.addWidget(QLabel("窗口帧数"), 0, 0)
        settings.addWidget(self.window_size_input, 0, 1)
        settings.addWidget(QLabel("步长帧数"), 0, 2)
        settings.addWidget(self.step_size_input, 0, 3)
        settings.addWidget(QLabel("稳定窗口"), 1, 0)
        settings.addWidget(self.stable_window_input, 1, 1)
        settings.addWidget(QLabel("触发比例"), 1, 2)
        settings.addWidget(self.p_ratio_input, 1, 3)
        settings.addWidget(QLabel("重置等待帧"), 2, 0)
        settings.addWidget(self.reset_delay_input, 2, 1)
        settings.addWidget(self.enable_keyboard_checkbox, 2, 2, 1, 2)
        control_layout.addLayout(settings)

        actions = QHBoxLayout()
        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        self.clear_btn = QPushButton("清空结果")
        self.clear_btn.setProperty("class", "btn-primary")
        actions.addWidget(self.apply_btn)
        actions.addWidget(self.clear_btn)
        actions.addStretch()
        control_layout.addLayout(actions)

        result_panel = QFrame()
        result_panel.setProperty("class", "panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(14, 14, 14, 14)
        result_layout.setSpacing(8)
        result_title = QLabel("实时概率柱状图")
        result_title.setProperty("class", "title")
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_meta_label)
        result_layout.addWidget(self.count_label)
        result_layout.addWidget(self.chart, 1)

        root.addWidget(control_panel)
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
        for address in removed_addresses:
            self.states.pop(address, None)

        for address in changed_connection_addresses:
            ring = self.ring_info.get(address)
            connected = bool(ring.get("connected", False)) if ring is not None else False
            if connected:
                continue
            self.states.pop(address, None)
            if self.active_address == address:
                self.active_address = ""
                self.active_ring_label.setText("当前显示：无")
                self.buffer_label.setText(f"缓冲区：0 / {self.window_size_input.value()} 帧")
                self.update_count_views(Counter())

    def handle_stream_message(self, message: dict) -> None:
        if str(message.get("type", "")) != "imu":
            return
        self.handle_imu_message(message)

    def handle_imu_message(self, message: dict) -> None:
        address = str(message.get("address", ""))
        if not address:
            return

        acc = message.get("acc")
        gyr = message.get("gyr")
        if not isinstance(acc, list) or not isinstance(gyr, list):
            return
        if len(acc) != 3 or len(gyr) != 3:
            return

        state = self.state_for(address)
        current_sample = [float(value) for value in acc] + [float(value) for value in gyr]
        state.samples.append(current_sample)
        state.frames_since_infer += 1
        reset_applied = self.advance_pending_reset(state, current_sample)
        self.active_address = address
        self.active_ring_label.setText(f"当前显示：{state.ring_name}")
        self.buffer_label.setText(f"缓冲区：{len(state.samples)} / {self.window_size_input.value()} 帧")
        self.update_count_views(state.committed_counts)
        if reset_applied:
            self.set_runtime_status(f"状态：{state.ring_name} 冷却结束，缓冲区已重置到最新一帧")
        self.maybe_submit_inference(address)

    def maybe_submit_inference(self, address: str) -> None:
        if self.worker is None or self.runtime_failed or not self.runtime_ready or self.worker_busy:
            return

        state = self.states.get(address)
        if state is None:
            return
        if state.reset_after_frames_remaining > 0:
            return

        window_size = int(self.window_size_input.value())
        step_size = int(self.step_size_input.value())
        if len(state.samples) < window_size or state.frames_since_infer < step_size:
            return

        values = np.asarray(list(state.samples)[-window_size:], dtype=np.float32)
        state.frames_since_infer = 0
        self.worker_busy = True
        self.worker.submit(
            InferenceJob(
                address=address,
                ring_name=state.ring_name,
                values=values,
            )
        )
        self.set_runtime_status(f"状态：{state.ring_name} 推理中（窗长 {window_size}，步长 {step_size}）")

    def apply_subscription(self) -> None:
        self.send_subscription(["imu"])
        if self.runtime_failed:
            self.set_runtime_status("状态：已订阅 IMU，但推理不可用")
        elif self.runtime_ready:
            self.set_runtime_status("状态：已订阅 IMU，等待窗口积累")
        else:
            self.set_runtime_status("状态：已订阅 IMU，等待模型加载")

    def on_worker_ready(self) -> None:
        self.runtime_ready = True
        self.device_label.setText("推理设备：CPU | 模型：checkpoint/swipe/cls-pinch | 已就绪")
        self.set_runtime_status("状态：模型已就绪，等待 IMU 数据")

    def on_worker_fatal_error(self, message: str) -> None:
        self.runtime_failed = True
        self.runtime_ready = False
        self.worker_busy = False
        self.device_label.setText("推理设备：CPU | 模型：加载失败")
        self.set_runtime_status(f"状态：{message}")

    def on_inference_result(self, result: InferenceResult) -> None:
        self.worker_busy = False
        state = self.state_for(result.address)
        self.chart.set_probabilities(result.prediction.probabilities)
        self.result_label.setText(f"{result.prediction.class_id} · {result.prediction.label}")
        decision = self.update_gesture_state(result.address, state, result.prediction.class_id)
        self.result_meta_label.setText(
            (
                f"{result.ring_name} | 窗长 {result.sample_count} 帧 | 推理 {result.elapsed_ms:.1f} ms | "
                f"最高概率 {result.prediction.confidence:.3f} | 队列 {decision.history_size}/{decision.queue_length} | "
                f"当前类计数 {decision.class_count}/{decision.threshold}"
            )
        )
        self.update_count_views(state.committed_counts)

        if decision.triggered_class_id is None:
            self.set_runtime_status(
                f"状态：{result.ring_name} -> {result.prediction.class_id} {result.prediction.label} "
                f"({result.prediction.confidence:.3f}) | 等待稳定触发"
            )
        else:
            key_name = self.emit_keyboard_action(decision.triggered_class_id)
            key_text = "" if key_name is None else f" | 按键 {key_name}"
            self.set_runtime_status(
                f"状态：{result.ring_name} 触发 {decision.triggered_class_id}:{PINCH_CLASS_LABELS[decision.triggered_class_id]} | "
                f"累计 {int(state.committed_counts.get(decision.triggered_class_id, 0))} | "
                f"等待 {state.reset_after_frames_remaining} 帧后重置缓冲{key_text}"
            )
        self.maybe_submit_inference(result.address)

    def on_inference_failed(self, failure: InferenceFailure) -> None:
        self.worker_busy = False
        self.set_runtime_status(f"状态：{failure.ring_name} 推理失败（{failure.error}）")

    def clear_results(self) -> None:
        self.states.clear()
        self.active_address = ""
        self.worker_busy = False
        self.chart.clear()
        self.result_label.setText("--")
        self.result_meta_label.setText("最近结果：无")
        self.count_label.setText("累计计数：1=0 2=0")
        self.active_ring_label.setText("当前显示：无")
        self.buffer_label.setText(f"缓冲区：0 / {self.window_size_input.value()} 帧")
        self.set_runtime_status("状态：结果已清空，等待 IMU 数据")

    def set_runtime_status(self, text: str) -> None:
        self.workflow_label.setText(text)

    def state_for(self, address: str) -> RingBufferState:
        state = self.states.get(address)
        ring_name = self.ring_name(address)
        if state is None:
            state = RingBufferState(ring_name=ring_name)
            self.states[address] = state
        elif state.ring_name != ring_name:
            state.ring_name = ring_name
        return state

    def result_queue_length(self) -> int:
        step_size = max(int(self.step_size_input.value()), 1)
        stable_window_seconds = float(self.stable_window_input.value())
        return max(1, int(round(stable_window_seconds * IMU_SAMPLE_RATE_HZ / step_size)))

    def min_trigger_count(self, queue_length: int) -> int:
        return max(1, int(math.ceil(float(self.p_ratio_input.value()) * queue_length)))

    def advance_pending_reset(self, state: RingBufferState, current_sample: list[float]) -> bool:
        if state.reset_after_frames_remaining <= 0:
            return False

        state.reset_after_frames_remaining -= 1
        if state.reset_after_frames_remaining > 0:
            return False

        state.samples.clear()
        for _ in range(self.window_size_input.value() // 2 - 1):
            state.samples.append(current_sample)
        state.frames_since_infer = 0
        state.prediction_history.clear()
        return True

    def update_gesture_state(self, address: str, state: RingBufferState, class_id: int) -> GestureDecision:
        _ = address
        queue_length = self.result_queue_length()
        history = state.prediction_history
        history.append(class_id)
        while len(history) > queue_length:
            history.popleft()

        threshold = self.min_trigger_count(queue_length)
        class_count = history.count(class_id)
        history_size = len(history)
        if len(history) < queue_length:
            return GestureDecision(
                queue_length=queue_length,
                threshold=threshold,
                history_size=history_size,
                class_count=class_count,
                triggered_class_id=None,
            )

        for candidate in PINCH_CLASS_IDS[1:]:
            if history.count(candidate) < threshold:
                continue
            state.committed_counts[candidate] += 1
            state.prediction_history.clear()
            state.frames_since_infer = 0
            state.reset_after_frames_remaining = int(self.reset_delay_input.value())
            return GestureDecision(
                queue_length=queue_length,
                threshold=threshold,
                history_size=history_size,
                class_count=class_count,
                triggered_class_id=candidate,
            )
        return GestureDecision(
            queue_length=queue_length,
            threshold=threshold,
            history_size=history_size,
            class_count=class_count,
            triggered_class_id=None,
        )

    def update_count_views(self, counts: Counter[int]) -> None:
        text = " ".join(f"{class_id}={int(counts.get(class_id, 0))}" for class_id in PINCH_CLASS_IDS[1:])
        self.count_label.setText(f"累计计数：{text}")

    def emit_keyboard_action(self, class_id: int) -> str | None:
        if not self.enable_keyboard_checkbox.isChecked() or self.keyboard_controller is None:
            return None

        key_name = PINCH_CLASS_TO_KEY.get(class_id)
        if key_name is None:
            return None

        key_mapping = {
            "up": keyboard.Key.up,
            "down": keyboard.Key.down,
            "left": keyboard.Key.left,
            "right": keyboard.Key.right,
            "space": keyboard.Key.space,
        }
        key = key_mapping.get(key_name)
        if key is None:
            return None

        try:
            self.keyboard_controller.tap(key)
        except Exception:
            return None
        return key_name

    def ring_name(self, address: str) -> str:
        ring = self.ring_info.get(address)
        if ring is not None:
            name = str(ring.get("name", "")).strip()
            if name:
                return name
        if address:
            return address[-8:]
        return "Unknown"

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
    win = PinchRealtimeWindow(host=args.host, port=args.port)
    win.show()
    raise SystemExit(app.exec())
