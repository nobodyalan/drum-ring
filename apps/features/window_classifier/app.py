from __future__ import annotations

"""Generic standalone window for window-based gesture classifiers."""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.stream_window import RingStreamWindowBase
from apps.framework.ui_style import APP_STYLE_SHEET

from .runtime import (
    GestureDecision,
    InferenceFailure,
    InferenceJob,
    InferenceResult,
    ProbabilityChart,
    RingBufferState,
    WindowClassifierPreset,
    WindowClassifierWorker,
)
from .ui import (
    create_window_classifier_control_panel,
    create_window_classifier_result_panel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMU_SAMPLE_RATE_HZ = 200.0

try:
    from pynput import keyboard

    HAS_KEYBOARD_CONTROL = True
    KEYBOARD_CONTROL_ERROR = ""
except Exception as exc:
    keyboard = None
    HAS_KEYBOARD_CONTROL = False
    KEYBOARD_CONTROL_ERROR = str(exc)


class WindowClassifierAppWindow(RingStreamWindowBase):
    """Shared window that turns a classifier preset into a runnable desktop app."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        preset: WindowClassifierPreset,
        backend: str = "pytorch",
        model_path: Path | None = None,
    ):
        super().__init__(host=host, port=port)
        self.preset = preset
        self.setWindowTitle(preset.window_title)
        self.resize(1200, 800)

        self.backend = str(backend).strip().lower()
        self.model_source = self.resolve_model_source(model_path)
        self.backend_display_name = "OpenVINO" if self.backend == "openvino" else "PyTorch"
        self.model_source_label = self.display_path(self.model_source)
        self.states: dict[str, RingBufferState] = {}
        self.active_address = ""
        self.active_result_address = ""
        self.runtime_ready = False
        self.runtime_failed = False
        self.worker_busy = False
        self.keyboard_controller = None
        if HAS_KEYBOARD_CONTROL and self.preset.keyboard_action_map:
            self.keyboard_controller = keyboard.Controller()

        self.stream_status_label = QLabel("数据流状态：未连接")
        self.stream_status_label.setObjectName("statusTag")
        self.workflow_label = QLabel("状态：准备启动")
        self.workflow_label.setWordWrap(True)
        self.device_label = QLabel(
            f"推理后端：{self.backend_display_name} | 模型：{self.model_source_label} | 加载中"
        )
        self.active_ring_label = QLabel("当前显示：无")
        self.buffer_label = QLabel("缓冲区：0 / 100 帧")
        self.fps_label: QLabel | None = None
        if self.preset.spec.enable_result_fps:
            self.fps_label = QLabel("结果帧率：0.0 Hz")
        self.chart = ProbabilityChart(
            class_ids=self.preset.spec.class_ids,
            class_labels=self.preset.spec.class_labels,
            class_colors=self.preset.spec.class_colors,
            allow_compact_probabilities=self.preset.spec.allow_compact_probabilities,
        )

        self.window_size_input = QSpinBox()
        self.window_size_input.setRange(40, 400)
        self.window_size_input.setSingleStep(10)
        self.window_size_input.setValue(self.preset.spec.default_window_size)
        self.window_size_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.step_size_input = QSpinBox()
        self.step_size_input.setRange(1, 200)
        self.step_size_input.setSingleStep(1)
        self.step_size_input.setValue(self.preset.spec.default_step_frames)
        self.step_size_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.stable_window_input = QDoubleSpinBox()
        self.stable_window_input.setRange(0.02, 1.0)
        self.stable_window_input.setDecimals(2)
        self.stable_window_input.setSingleStep(0.02)
        self.stable_window_input.setValue(self.preset.spec.default_stable_window_seconds)
        self.stable_window_input.setSuffix(" s")
        self.stable_window_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.p_ratio_input = QDoubleSpinBox()
        self.p_ratio_input.setRange(0.50, 1.00)
        self.p_ratio_input.setDecimals(2)
        self.p_ratio_input.setSingleStep(0.05)
        self.p_ratio_input.setValue(self.preset.spec.default_positive_ratio)
        self.p_ratio_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.reset_delay_input = QSpinBox()
        self.reset_delay_input.setRange(1, 200)
        self.reset_delay_input.setSingleStep(1)
        self.reset_delay_input.setValue(self.preset.spec.default_reset_delay_frames)
        self.reset_delay_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.enable_keyboard_checkbox = QCheckBox("控制键盘")
        self.enable_keyboard_checkbox.setChecked(False)
        self.enable_keyboard_checkbox.setEnabled(HAS_KEYBOARD_CONTROL and bool(self.preset.keyboard_action_map))
        self.enable_keyboard_checkbox.setToolTip(self.keyboard_tooltip())

        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self._bind_signals()

        self.fps_timer: QTimer | None = None
        if self.fps_label is not None:
            self.fps_timer = QTimer(self)
            self.fps_timer.setInterval(1000)
            self.fps_timer.timeout.connect(self.refresh_fps)
            self.fps_timer.start()

        self.worker = WindowClassifierWorker(
            recognizer_factory=lambda: self.preset.build_recognizer(
                backend=self.backend,
                model_path=self.model_source,
            ),
        )
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

        status_widgets = [
            self.stream_status_label,
            self.workflow_label,
            self.device_label,
            self.active_ring_label,
            self.buffer_label,
        ]
        if self.fps_label is not None:
            status_widgets.append(self.fps_label)

        control_view = create_window_classifier_control_panel(
            title_text=self.preset.panel_title,
            description_text=self.preset.description_text,
            status_widgets=status_widgets,
        )

        self.add_ring_selection_panel(
            control_view.layout,
            title_text="选择订阅戒指（不勾选表示接收全部）",
            max_height=160,
        )

        settings = control_view.settings_layout
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

        self.apply_btn = QPushButton("应用订阅")
        self.apply_btn.setProperty("class", "btn-primary")
        self.clear_btn = QPushButton("清空结果")
        self.clear_btn.setProperty("class", "btn-primary")
        control_view.action_row.addWidget(self.apply_btn)
        control_view.action_row.addWidget(self.clear_btn)
        control_view.action_row.addStretch()

        result_view = create_window_classifier_result_panel(
            chart=self.chart,
            title_text=self.preset.result_title_text,
        )
        self.result_label = result_view.result_label
        self.result_meta_label = result_view.meta_label
        self.count_label = result_view.count_label
        self.count_label.setText(self.empty_count_text())

        root.addWidget(control_view.panel)
        root.addWidget(result_view.panel, 1)

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
            self._remove_state(address)

        for address in changed_connection_addresses:
            ring = self.ring_info.get(address)
            connected = bool(ring.get("connected", False)) if ring is not None else False
            if not connected:
                self._remove_state(address)

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
        if self.runtime_failed or not self.runtime_ready or self.worker_busy:
            return

        state = self.states.get(address)
        if state is None or state.reset_after_frames_remaining > 0:
            return

        window_size = int(self.window_size_input.value())
        step_size = int(self.step_size_input.value())
        if len(state.samples) < window_size or state.frames_since_infer < step_size:
            return

        values = np.asarray(list(state.samples)[-window_size:], dtype=np.float32)
        state.frames_since_infer = 0
        self.worker_busy = True
        self.worker.submit(InferenceJob(address=address, ring_name=state.ring_name, values=values))
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
        self.device_label.setText(
            f"推理后端：{self.backend_display_name} | 模型：{self.model_source_label} | 已就绪"
        )
        self.set_runtime_status("状态：模型已就绪，等待 IMU 数据")

    def on_worker_fatal_error(self, message: str) -> None:
        self.runtime_failed = True
        self.runtime_ready = False
        self.worker_busy = False
        self.device_label.setText(
            f"推理后端：{self.backend_display_name} | 模型：{self.model_source_label} | 加载失败"
        )
        self.set_runtime_status(f"状态：{message}")

    def on_inference_result(self, result: InferenceResult) -> None:
        self.worker_busy = False
        state = self.state_for(result.address)
        self.active_result_address = result.address
        state.rate_tracker.mark()
        self.chart.set_probabilities(result.prediction.probabilities)
        self.result_label.setText(f"{result.prediction.class_id} · {result.prediction.label}")
        decision = self.update_gesture_state(state, result.prediction.class_id)
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
            class_id = int(decision.triggered_class_id)
            key_name = self.emit_keyboard_action(class_id)
            key_text = "" if key_name is None else f" | 按键 {key_name}"
            label = self.preset.spec.class_labels[class_id]
            total = int(state.committed_counts.get(class_id, 0))
            self.set_runtime_status(
                f"状态：{result.ring_name} 触发 {class_id}:{label} | "
                f"累计 {total} | 等待 {state.reset_after_frames_remaining} 帧后重置缓冲{key_text}"
            )
        self.maybe_submit_inference(result.address)

    def on_inference_failed(self, failure: InferenceFailure) -> None:
        self.worker_busy = False
        self.set_runtime_status(f"状态：{failure.ring_name} 推理失败（{failure.error}）")

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
        self.buffer_label.setText(f"缓冲区：0 / {self.window_size_input.value()} 帧")
        if self.fps_label is not None:
            self.fps_label.setText("结果帧率：0.0 Hz")
        self.set_runtime_status("状态：结果已清空，等待 IMU 数据")

    def refresh_fps(self) -> None:
        if self.fps_label is None:
            return
        if not self.active_result_address:
            self.fps_label.setText("结果帧率：0.0 Hz")
            return
        state = self.states.get(self.active_result_address)
        if state is None:
            self.fps_label.setText("结果帧率：0.0 Hz")
            return
        self.fps_label.setText(f"结果帧率：{state.rate_tracker.fps():.1f} Hz")

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

    def update_gesture_state(self, state: RingBufferState, class_id: int) -> GestureDecision:
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

        for candidate in self.preset.spec.class_ids[1:]:
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
        text = " ".join(
            f"{class_id}={int(counts.get(class_id, 0))}"
            for class_id in self.preset.spec.class_ids[1:]
        )
        self.count_label.setText(f"累计计数：{text}")

    def emit_keyboard_action(self, class_id: int) -> str | None:
        if not self.enable_keyboard_checkbox.isChecked() or self.keyboard_controller is None:
            return None

        key_name = self.preset.keyboard_action_map.get(class_id)
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

    def resolve_model_source(self, model_path: Path | None) -> Path:
        source = self.preset.spec.checkpoint_dir if model_path is None else Path(model_path).expanduser().resolve()
        if self.backend == "pytorch" and source.is_file():
            return source.parent
        return source

    def display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path.resolve())

    def keyboard_tooltip(self) -> str:
        if not self.preset.keyboard_action_map:
            return "当前分类器未配置默认按键映射"
        if not HAS_KEYBOARD_CONTROL:
            return f"键盘控制不可用：{KEYBOARD_CONTROL_ERROR}"
        return self.preset.keyboard_tooltip or "触发手势时发送按键"

    def empty_count_text(self) -> str:
        text = " ".join(f"{class_id}=0" for class_id in self.preset.spec.class_ids[1:])
        return f"累计计数：{text}"

    def _remove_state(self, address: str) -> None:
        self.states.pop(address, None)
        was_active = self.active_address == address
        was_active_result = self.active_result_address == address
        if was_active:
            self.active_address = ""
            self.active_ring_label.setText("当前显示：无")
            self.buffer_label.setText(f"缓冲区：0 / {self.window_size_input.value()} 帧")
        if was_active_result:
            self.active_result_address = ""
            if self.fps_label is not None:
                self.fps_label.setText("结果帧率：0.0 Hz")
        if was_active or was_active_result:
            self.update_count_views(Counter())

    def closeEvent(self, event) -> None:
        if self.fps_timer is not None:
            self.fps_timer.stop()
        self.worker.stop()
        self.client.stop()
        self.client.wait(1500)
        event.accept()


def create_window_classifier_parser() -> argparse.ArgumentParser:
    parser = create_base_parser()
    parser.add_argument("--backend", choices=("pytorch", "openvino"), default="pytorch")
    parser.add_argument("--model-path", type=Path)
    return parser


def run_window_classifier_app(preset: WindowClassifierPreset) -> int:
    args = create_window_classifier_parser().parse_args()
    return run_window(
        lambda: WindowClassifierAppWindow(
            host=args.host,
            port=args.port,
            preset=preset,
            backend=args.backend,
            model_path=args.model_path,
        ),
        style_sheet=APP_STYLE_SHEET,
    )
