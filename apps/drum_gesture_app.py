from __future__ import annotations

import argparse
import math
import queue
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
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

from apps.features.window_classifier.runtime import ProbabilityChart, RateTracker
from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET
from models.drum_gesture_recognizer import (
    DRUM_CLASS_IDS,
    DRUM_CLASS_LABELS,
    WINDOW_SIZE,
    DrumGestureRecognizer,
    OnlineLinearAccelerationEstimator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "checkpoint" / "drum_gesture" / "linacc_early_stream_lead6_deep_gap_raw6.pth"
)
SOUND_DIR = PROJECT_ROOT / "assets" / "drum_gesture" / "sounds"
IMU_SAMPLE_RATE_HZ = 200.0

DRUM_CLASS_COLORS = {
    0: QColor("#B0B8C1"),
    1: QColor("#5B8FF9"),
    2: QColor("#61DDAA"),
    3: QColor("#F6BD16"),
    4: QColor("#E8684A"),
    5: QColor("#6DC8EC"),
    6: QColor("#9270CA"),
    7: QColor("#FF9D4D"),
}
DRUM_SOUND_FILES = {
    1: SOUND_DIR / "gesture_1_closed_hat.wav",
    2: SOUND_DIR / "gesture_2_tom2.wav",
    3: SOUND_DIR / "gesture_3_snare.wav",
    4: SOUND_DIR / "gesture_4_tom1.wav",
    5: SOUND_DIR / "gesture_5_tom3.wav",
    6: SOUND_DIR / "gesture_6_crash.wav",
    7: SOUND_DIR / "gesture_7_ride.wav",
}


@dataclass(frozen=True)
class DrumInferenceJob:
    address: str
    ring_name: str
    values: np.ndarray
    window_end_sample_index: int
    window_peak_norm: float


@dataclass(frozen=True)
class DrumInferenceResult:
    address: str
    ring_name: str
    prediction: object
    elapsed_ms: float
    sample_count: int
    window_end_sample_index: int
    window_peak_norm: float


@dataclass(frozen=True)
class DrumInferenceFailure:
    ring_name: str
    error: str


class DrumInferenceWorker(QThread):
    ready = pyqtSignal()
    result_ready = pyqtSignal(object)
    job_failed = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, model_path: Path):
        super().__init__()
        self.model_path = Path(model_path)
        self._jobs: queue.Queue[DrumInferenceJob | None] = queue.Queue(maxsize=1)
        self._stopping = False

    def submit(self, job: DrumInferenceJob) -> None:
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
            recognizer = DrumGestureRecognizer(self.model_path)
        except Exception as exc:
            self.fatal_error.emit(f"模型加载失败: {exc}")
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
                self.job_failed.emit(DrumInferenceFailure(job.ring_name, str(exc)))
                continue
            self.result_ready.emit(
                DrumInferenceResult(
                    address=job.address,
                    ring_name=job.ring_name,
                    prediction=prediction,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    sample_count=int(job.values.shape[0]),
                    window_end_sample_index=job.window_end_sample_index,
                    window_peak_norm=job.window_peak_norm,
                )
            )


@dataclass
class DrumRingState:
    ring_name: str
    estimator: OnlineLinearAccelerationEstimator = field(default_factory=OnlineLinearAccelerationEstimator)
    samples: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=512))
    norms: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    committed_counts: Counter[int] = field(default_factory=Counter)
    prediction_history: deque[int] = field(default_factory=deque)
    rate_tracker: RateTracker = field(default_factory=RateTracker)
    sample_count: int = 0
    frames_since_infer: int = 0
    last_timestamp: float | None = None
    last_trigger_sample: int = -100000


class DrumGestureWindow(RingStreamWindowBase):
    def __init__(self, host: str, port: int, model_path: Path):
        super().__init__(host=host, port=port)
        self.setWindowTitle("Drum Gesture 实时识别")
        self.resize(1180, 780)

        self.model_path = Path(model_path).expanduser().resolve()
        self.states: dict[str, DrumRingState] = {}
        self.active_address = ""
        self.active_result_address = ""
        self.runtime_ready = False
        self.runtime_failed = False
        self.worker_busy = False
        self.sound_effects: dict[int, QSoundEffect] = {}

        self.stream_status_label = QLabel("数据流状态：未连接")
        self.stream_status_label.setObjectName("statusTag")
        self.workflow_label = QLabel("状态：准备启动")
        self.workflow_label.setWordWrap(True)
        self.device_label = QLabel(f"模型：{self.display_path(self.model_path)} | 加载中")
        self.active_ring_label = QLabel("当前显示：无")
        self.buffer_label = QLabel("缓冲区：0 帧")
        self.fps_label = QLabel("结果帧率：0.0 Hz")
        self.peak_label = QLabel("最近峰值：-")

        self.chart = ProbabilityChart(
            class_ids=DRUM_CLASS_IDS,
            class_labels=DRUM_CLASS_LABELS,
            class_colors=DRUM_CLASS_COLORS,
        )

        self.confidence_threshold_input = QDoubleSpinBox()
        self.confidence_threshold_input.setRange(0.0, 1.0)
        self.confidence_threshold_input.setDecimals(2)
        self.confidence_threshold_input.setSingleStep(0.05)
        self.confidence_threshold_input.setValue(0.55)
        self.confidence_threshold_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.cooldown_input = QSpinBox()
        self.cooldown_input.setRange(1, 200)
        self.cooldown_input.setSingleStep(5)
        self.cooldown_input.setValue(24)
        self.cooldown_input.setSuffix(" 帧")
        self.cooldown_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.step_input = QSpinBox()
        self.step_input.setRange(1, 20)
        self.step_input.setValue(5)
        self.step_input.setSuffix(" 帧")
        self.step_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.stable_window_input = QDoubleSpinBox()
        self.stable_window_input.setRange(0.02, 0.50)
        self.stable_window_input.setDecimals(2)
        self.stable_window_input.setSingleStep(0.01)
        self.stable_window_input.setValue(0.04)
        self.stable_window_input.setSuffix(" s")
        self.stable_window_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.trigger_ratio_input = QDoubleSpinBox()
        self.trigger_ratio_input.setRange(0.50, 1.00)
        self.trigger_ratio_input.setDecimals(2)
        self.trigger_ratio_input.setSingleStep(0.05)
        self.trigger_ratio_input.setValue(0.67)
        self.trigger_ratio_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.volume_input = QDoubleSpinBox()
        self.volume_input.setRange(0.0, 1.0)
        self.volume_input.setDecimals(2)
        self.volume_input.setSingleStep(0.05)
        self.volume_input.setValue(0.80)
        self.volume_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.volume_input.valueChanged.connect(self.on_volume_changed)

        self.load_sound_effects()
        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self._bind_signals()

        self.worker = DrumInferenceWorker(self.model_path)
        self.worker.ready.connect(self.on_worker_ready)
        self.worker.result_ready.connect(self.on_inference_result)
        self.worker.job_failed.connect(self.on_inference_failed)
        self.worker.fatal_error.connect(self.on_worker_fatal_error)
        self.worker.start()
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

        title = QLabel("Drum Gesture 实时识别")
        title.setProperty("class", "title")
        desc = QLabel(
            "在线估计 linear acceleration，持续滑窗推理最近 15 帧，稳定判定后触发对应鼓件音色。"
        )
        desc.setProperty("class", "sub-text")
        desc.setWordWrap(True)
        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)
        for widget in (
            self.stream_status_label,
            self.workflow_label,
            self.device_label,
            self.active_ring_label,
            self.buffer_label,
            self.fps_label,
            self.peak_label,
        ):
            panel_layout.addWidget(widget)

        self.add_ring_selection_panel(
            panel_layout,
            title_text="选择订阅戒指（不勾选表示接收全部）",
            max_height=150,
        )

        settings_panel = QFrame()
        settings_panel.setProperty("class", "settings-panel")
        settings = QGridLayout(settings_panel)
        settings.setContentsMargins(14, 12, 14, 12)
        settings.setHorizontalSpacing(16)
        settings.setVerticalSpacing(12)
        settings.addWidget(QLabel("推理步长"), 0, 0)
        settings.addWidget(self.step_input, 0, 1)
        settings.addWidget(QLabel("置信阈值"), 0, 2)
        settings.addWidget(self.confidence_threshold_input, 0, 3)
        settings.addWidget(QLabel("冷却时间"), 1, 0)
        settings.addWidget(self.cooldown_input, 1, 1)
        settings.addWidget(QLabel("稳定窗口"), 1, 2)
        settings.addWidget(self.stable_window_input, 1, 3)
        settings.addWidget(QLabel("触发比例"), 2, 0)
        settings.addWidget(self.trigger_ratio_input, 2, 1)
        settings.addWidget(QLabel("音量"), 2, 2)
        settings.addWidget(self.volume_input, 2, 3)
        panel_layout.addWidget(settings_panel)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        self.clear_btn = QPushButton("清空结果")
        self.clear_btn.setProperty("class", "btn-gray")
        action_row.addWidget(self.apply_btn)
        action_row.addWidget(self.clear_btn)
        action_row.addStretch()
        panel_layout.addLayout(action_row)

        result_panel = QFrame()
        result_panel.setProperty("class", "panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(14, 14, 14, 14)
        result_layout.setSpacing(10)
        result_title = QLabel("实时概率")
        result_title.setProperty("class", "title")
        self.result_label = QLabel("--")
        self.result_label.setProperty("class", "title")
        self.result_meta_label = QLabel("最近结果：无")
        self.result_meta_label.setProperty("class", "sub-text")
        self.result_meta_label.setWordWrap(True)
        self.count_label = QLabel(self.empty_count_text())
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.chart, 1)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_meta_label)
        result_layout.addWidget(self.count_label)

        root.addWidget(panel)
        root.addWidget(result_panel, 1)

    def _bind_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_subscription)
        self.clear_btn.clicked.connect(self.clear_results)
        self.bind_stream_client_signals()

    def on_stream_connected(self) -> None:
        self.apply_subscription()

    def apply_subscription(self) -> None:
        self.send_subscription(["imu"])
        if self.runtime_failed:
            self.set_runtime_status("状态：已订阅 IMU，但推理不可用")
        elif self.runtime_ready:
            self.set_runtime_status("状态：已订阅 IMU，持续滑窗推理中")
        else:
            self.set_runtime_status("状态：已订阅 IMU，等待模型加载")

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
        if not isinstance(acc, list) or not isinstance(gyr, list) or len(acc) != 3 or len(gyr) != 3:
            return

        state = self.state_for(address)
        timestamp = self.message_timestamp(message)
        dt = self.sample_dt(state, timestamp)
        linear_acc, _gravity = state.estimator.update(
            np.asarray(acc, dtype=np.float32),
            np.asarray(gyr, dtype=np.float32),
            dt,
        )
        sample = np.concatenate((linear_acc, np.asarray(gyr, dtype=np.float32))).astype(np.float32)
        state.samples.append(sample)
        state.norms.append(float(np.linalg.norm(linear_acc)))
        state.sample_count += 1
        state.frames_since_infer += 1
        state.last_timestamp = timestamp

        self.active_address = address
        self.active_ring_label.setText(f"当前显示：{state.ring_name}")
        self.buffer_label.setText(f"缓冲区：{len(state.samples)} 帧 | 当前合加速度 {state.norms[-1]:.1f}")
        self.update_count_views(state.committed_counts)
        self.maybe_submit_window(address)

    def maybe_submit_window(self, address: str) -> None:
        if self.runtime_failed or not self.runtime_ready or self.worker_busy:
            return
        state = self.states.get(address)
        if state is None or len(state.samples) < WINDOW_SIZE:
            return

        step = int(self.step_input.value())
        if state.frames_since_infer < step:
            return

        values = np.stack(list(state.samples)[-WINDOW_SIZE:], axis=0).astype(np.float32, copy=False)
        if values.shape != (WINDOW_SIZE, 6):
            return

        state.frames_since_infer = 0
        window_end = state.sample_count - 1
        window_peak = float(max(list(state.norms)[-WINDOW_SIZE:]))
        self.worker_busy = True
        self.worker.submit(
            DrumInferenceJob(
                address=address,
                ring_name=state.ring_name,
                values=values,
                window_end_sample_index=window_end,
                window_peak_norm=window_peak,
            )
        )
        self.set_runtime_status(f"状态：{state.ring_name} 滑窗推理中（步长 {step} 帧）")

    def on_worker_ready(self) -> None:
        self.runtime_ready = True
        self.device_label.setText(f"模型：{self.display_path(self.model_path)} | 已就绪")
        self.set_runtime_status("状态：模型已就绪，等待 IMU 数据")

    def on_worker_fatal_error(self, message: str) -> None:
        self.runtime_failed = True
        self.runtime_ready = False
        self.worker_busy = False
        self.device_label.setText(f"模型：{self.display_path(self.model_path)} | 加载失败")
        self.set_runtime_status(f"状态：{message}")

    def on_inference_result(self, result: DrumInferenceResult) -> None:
        self.worker_busy = False
        state = self.state_for(result.address)
        self.active_result_address = result.address
        state.rate_tracker.mark()
        prediction = result.prediction
        self.chart.set_probabilities(prediction.probabilities)
        self.result_label.setText(f"{prediction.class_id} · {prediction.label}")
        self.peak_label.setText(
            f"窗口峰值：{result.window_peak_norm:.1f} | window end sample {result.window_end_sample_index}"
        )
        decision_class, history_size, queue_length, class_count, threshold = self.update_prediction_history(
            state,
            int(prediction.class_id),
            float(prediction.confidence),
        )
        self.result_meta_label.setText(
            f"{result.ring_name} | 窗长 {result.sample_count} 帧 | 推理 {result.elapsed_ms:.1f} ms | "
            f"最高概率 {prediction.confidence:.3f} | 队列 {history_size}/{queue_length} | "
            f"当前类计数 {class_count}/{threshold}"
        )

        is_trigger = (
            decision_class is not None
            and result.window_end_sample_index - state.last_trigger_sample >= int(self.cooldown_input.value())
        )
        if is_trigger:
            class_id = int(decision_class)
            state.committed_counts[class_id] += 1
            state.last_trigger_sample = int(result.window_end_sample_index)
            state.prediction_history.clear()
            self.play_gesture_sound(class_id)
            self.set_runtime_status(
                f"状态：{result.ring_name} 稳定触发 {class_id}:{DRUM_CLASS_LABELS[class_id]} "
                f"({prediction.confidence:.3f})"
            )
        else:
            self.set_runtime_status(
                f"状态：{result.ring_name} -> {prediction.class_id}:{prediction.label} "
                f"({prediction.confidence:.3f})，等待稳定触发"
            )
        self.update_count_views(state.committed_counts)
        self.maybe_submit_window(result.address)

    def on_inference_failed(self, failure: DrumInferenceFailure) -> None:
        self.worker_busy = False
        self.set_runtime_status(f"状态：{failure.ring_name} 推理失败（{failure.error}）")

    def result_queue_length(self) -> int:
        step = max(int(self.step_input.value()), 1)
        stable_seconds = float(self.stable_window_input.value())
        return max(1, int(round(stable_seconds * IMU_SAMPLE_RATE_HZ / step)))

    def min_trigger_count(self, queue_length: int) -> int:
        return max(1, int(math.ceil(float(self.trigger_ratio_input.value()) * queue_length)))

    def update_prediction_history(
        self,
        state: DrumRingState,
        class_id: int,
        confidence: float,
    ) -> tuple[int | None, int, int, int, int]:
        queue_length = self.result_queue_length()
        threshold = self.min_trigger_count(queue_length)
        stable_class_id = class_id
        if confidence < float(self.confidence_threshold_input.value()):
            stable_class_id = 0

        history = state.prediction_history
        history.append(stable_class_id)
        while len(history) > queue_length:
            history.popleft()

        history_size = len(history)
        class_count = history.count(stable_class_id)
        if history_size < queue_length:
            return None, history_size, queue_length, class_count, threshold

        for candidate in DRUM_CLASS_IDS[1:]:
            count = history.count(candidate)
            if count >= threshold:
                return int(candidate), history_size, queue_length, count, threshold
        return None, history_size, queue_length, class_count, threshold

    def clear_results(self) -> None:
        self.states.clear()
        self.active_address = ""
        self.active_result_address = ""
        self.worker_busy = False
        self.chart.clear()
        self.result_label.setText("--")
        self.result_meta_label.setText("最近结果：无")
        self.count_label.setText(self.empty_count_text())
        self.active_ring_label.setText("当前显示：无")
        self.buffer_label.setText("缓冲区：0 帧")
        self.fps_label.setText("结果帧率：0.0 Hz")
        self.peak_label.setText("最近峰值：-")
        self.set_runtime_status("状态：结果已清空，等待 IMU 数据")

    def load_sound_effects(self) -> None:
        self.sound_effects.clear()
        for class_id, path in DRUM_SOUND_FILES.items():
            effect = QSoundEffect(self)
            effect.setSource(QUrl.fromLocalFile(str(path.resolve())))
            effect.setVolume(float(self.volume_input.value()))
            self.sound_effects[class_id] = effect

    def on_volume_changed(self, value: float) -> None:
        for effect in self.sound_effects.values():
            effect.setVolume(float(value))

    def play_gesture_sound(self, class_id: int) -> None:
        effect = self.sound_effects.get(class_id)
        if effect is None:
            return
        if effect.isPlaying():
            effect.stop()
        effect.play()

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
            if not connected:
                self.states.pop(address, None)

    def state_for(self, address: str) -> DrumRingState:
        state = self.states.get(address)
        ring_name = self.ring_name(address)
        if state is None:
            state = DrumRingState(ring_name=ring_name)
            self.states[address] = state
        elif state.ring_name != ring_name:
            state.ring_name = ring_name
        return state

    def update_count_views(self, counts: Counter[int]) -> None:
        text = " ".join(f"{class_id}={int(counts.get(class_id, 0))}" for class_id in DRUM_CLASS_IDS[1:])
        self.count_label.setText(f"累计计数：{text}")
        if self.active_result_address:
            state = self.states.get(self.active_result_address)
            if state is not None:
                self.fps_label.setText(f"结果帧率：{state.rate_tracker.fps():.1f} Hz")

    def set_runtime_status(self, text: str) -> None:
        self.workflow_label.setText(text)

    def ring_name(self, address: str) -> str:
        ring = self.ring_info.get(address)
        if ring is not None:
            name = str(ring.get("name", "")).strip()
            if name:
                return name
        return address[-8:] if address else "Unknown"

    def display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path.resolve())

    def empty_count_text(self) -> str:
        text = " ".join(f"{class_id}=0" for class_id in DRUM_CLASS_IDS[1:])
        return f"累计计数：{text}"

    def message_timestamp(self, message: dict) -> float | None:
        for key in ("local_timestamp", "timestamp", "host_timestamp"):
            try:
                value = float(message.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return value
        return None

    def sample_dt(self, state: DrumRingState, timestamp: float | None) -> float:
        if timestamp is None or state.last_timestamp is None:
            return 1.0 / IMU_SAMPLE_RATE_HZ
        dt = float(timestamp - state.last_timestamp)
        if dt > 1.0:
            dt = dt / 1000.0
        if not np.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            return 1.0 / IMU_SAMPLE_RATE_HZ
        return dt

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def create_parser() -> argparse.ArgumentParser:
    parser = create_base_parser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    return run_window(
        lambda: DrumGestureWindow(host=args.host, port=args.port, model_path=args.model_path),
        style_sheet=APP_STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
