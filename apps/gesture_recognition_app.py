import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import threading

import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from common.stream_window import RingStreamWindowBase
from common.ui_style import APP_STYLE_SHEET
from gesture_archive.constants import GESTURE_CODE_TO_NAME

try:
    import torch
    from tsai.models.InceptionTime import InceptionTime

    HAS_MODEL_RUNTIME = True
except Exception as exc:
    torch = None
    InceptionTime = None
    HAS_MODEL_RUNTIME = False
    MODEL_RUNTIME_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET


@dataclass
class GestureConfig:
    allowed_gestures: list[int]
    skipped_gestures: list[int]
    positive_ratio: float
    result_window_length: int
    window_size: int
    models: list[str]
    models_allowed_gestures: list[list[int]]
    device: str
    merge_window_seconds: float


@dataclass(frozen=True)
class GestureSnapshot:
    last_code: int = -1
    last_name: str = "-"
    last_ts: float = 0.0
    detected_count: int = 0


def load_gesture_config() -> GestureConfig:
    # Try reading gesture_archive/config.py runtime cfg first.
    return GestureConfig(
        allowed_gestures=[6, 8, 13, 14, 16, 36],
        skipped_gestures=[36],
        positive_ratio=0.8,
        result_window_length=8,
        window_size=200,
        models=[
            "checkpoint/it-pinch.pth",
            "checkpoint/it-snap.pth",
        ],
        models_allowed_gestures=[
            [8, 13, 14],
            [16, 30, 31, 34, 35, 36],
        ],
        device="cpu",
        merge_window_seconds=1.0,
    )


@dataclass
class RingRuntimeState:
    result_windows: list[deque[int]]
    ring_last_touch_time: float = 0.0
    last_code: int = -1
    last_name: str = "-"
    last_ts: float = 0.0
    detected_count: int = 0
    last_emitted_code: int = -1
    last_emitted_ts: float = 0.0


@dataclass
class PendingSample:
    event_ts: float


class InceptionGestureRecognizer:
    """手势识别主流程。

    逻辑只保留三步：
    1. 每次新窗口到来时，轮询所有模型各做一次预测。
    2. 每个模型都维护自己的结果滑窗，只有“同一结果占比足够高”才算稳定命中。
    3. 命中后再做业务规则过滤：跳过手势、touch/pinch 去抖、短时间重复合并。
    """

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.ready = False
        self.error_message = ""
        self.models = []
        self.model_pointer = 0
        self.ring_states: dict[str, RingRuntimeState] = {}
        self.allowed_output_indices = [i for i in range(1, 39) if i not in cfg.allowed_gestures]

        if not HAS_MODEL_RUNTIME:
            self.error_message = f"torch/tsai unavailable: {MODEL_RUNTIME_ERROR}"
            return

        try:
            device_str = cfg.device
            if device_str.startswith("cuda") and not torch.cuda.is_available():
                device_str = "cpu"
            self.device = torch.device(device_str)

            for model_path_str in cfg.models:
                model_path = APP_ROOT / model_path_str
                if not model_path.exists():
                    raise FileNotFoundError(f"model not found: {model_path}")

                model = InceptionTime(6, 39, nf=32).to(self.device)
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                model.eval()
                self.models.append(model)

            if len(self.models) != len(cfg.models_allowed_gestures):
                raise ValueError("models and models_allowed_gestures length mismatch")

            self.ready = True
        except Exception as exc:
            self.error_message = str(exc)

    def state_for(self, address: str) -> RingRuntimeState:
        if address not in self.ring_states:
            self.ring_states[address] = RingRuntimeState(
                result_windows=[deque(maxlen=self.cfg.result_window_length) for _ in self.models],
            )
        return self.ring_states[address]

    def snapshot_from_state(self, state: RingRuntimeState) -> GestureSnapshot:
        return GestureSnapshot(
            last_code=state.last_code,
            last_name=state.last_name,
            last_ts=state.last_ts,
            detected_count=state.detected_count,
        )

    def clear_all_result_windows(self, state: RingRuntimeState):
        for window in state.result_windows:
            window.clear()

    def _predict_one_model(self, model, data: np.ndarray) -> int:
        outputs = model(
            torch.from_numpy(data).float().view(1, 6, self.cfg.window_size).to(self.device)
        )
        outputs[:, self.allowed_output_indices] = -1e9
        return int(torch.max(outputs, 1).indices[0].item())

    def _collect_model_predictions(self, data: np.ndarray, state: RingRuntimeState) -> None:
        with torch.no_grad():
            for _ in self.models:
                model_index = self.model_pointer
                gesture = self._predict_one_model(self.models[model_index], data)
                state.result_windows[model_index].append(gesture)
                self.model_pointer = (self.model_pointer + 1) % len(self.models)

    def _stable_gesture_from_windows(self, state: RingRuntimeState) -> int:
        required_votes = self.cfg.result_window_length * self.cfg.positive_ratio
        for index, window in enumerate(state.result_windows):
            if not window:
                return -1
            gesture = window[-1]
            if window.count(gesture) < required_votes:
                continue
            if gesture not in self.cfg.models_allowed_gestures[index]:
                continue
            self.clear_all_result_windows(state)
            return gesture
        return -1

    def _detect_gesture_with_window(self, data: np.ndarray, state: RingRuntimeState) -> int:
        # 先更新每个模型自己的滑窗，再从滑窗里找“稳定结果”。
        self._collect_model_predictions(data, state)
        return self._stable_gesture_from_windows(state)

    def _check_if_skipped(self, gesture: int, state: RingRuntimeState) -> bool:
        if gesture in self.cfg.skipped_gestures:
            return True

        name = GESTURE_CODE_TO_NAME.get(gesture, "")
        now = time.time()
        if name.startswith("touch"):
            if now - state.ring_last_touch_time < 0.2:
                state.ring_last_touch_time = now
                return True
            state.ring_last_touch_time = now

        if name.startswith("pinch"):
            if now - state.ring_last_touch_time < 0.2:
                return True
        return False

    def _clear_runtime_state(self, state: RingRuntimeState, *, preserve_count: bool) -> None:
        state.ring_last_touch_time = 0.0
        state.last_emitted_code = -1
        state.last_emitted_ts = 0.0
        self.clear_all_result_windows(state)
        if not preserve_count:
            state.detected_count = 0
            state.last_code = -1
            state.last_name = "-"
            state.last_ts = 0.0

    def clear_stream_state(self, addresses: list[str] | None = None) -> None:
        if addresses is None:
            for state in self.ring_states.values():
                self._clear_runtime_state(state, preserve_count=True)
            return
        for address in addresses:
            state = self.ring_states.get(address)
            if state is not None:
                self._clear_runtime_state(state, preserve_count=True)

    def detect_window(self, address: str, window_data: np.ndarray, event_ts: float) -> int:
        if not self.ready:
            return -1

        state = self.state_for(address)
        gesture = self._detect_gesture_with_window(window_data, state)

        # 兼容旧逻辑：gesture 16 幅值太小就视为无效。
        if gesture == 16:
            if float(np.max(window_data) - np.min(window_data)) < 25.0:
                gesture = -2

        # 命中了也不一定上报，还要过一层业务过滤。
        if gesture >= 0 and self._check_if_skipped(gesture, state):
            self.clear_all_result_windows(state)
            return -2

        if (
            gesture >= 0
            and gesture == state.last_emitted_code
            and event_ts - state.last_emitted_ts < self.cfg.merge_window_seconds
        ):
            return -2

        if gesture >= 0:
            state.last_code = gesture
            state.last_name = GESTURE_CODE_TO_NAME.get(gesture, f"gesture-{gesture}")
            state.last_ts = event_ts
            state.detected_count += 1
            state.last_emitted_code = gesture
            state.last_emitted_ts = event_ts

        return gesture

    def reset_counts(self):
        for state in self.ring_states.values():
            self._clear_runtime_state(state, preserve_count=False)


class GestureInferenceWorker(QThread):
    sig_runtime = pyqtSignal(bool, str)
    sig_result = pyqtSignal(str, dict, int)
    sig_stats = pyqtSignal(dict)

    MODE_MAX = "max"
    MODE_TARGET = "target"
    STATS_INTERVAL = 0.5
    RESET_MARKER = object()
    CLEAR_MARKER = object()

    def __init__(self, cfg: GestureConfig):
        super().__init__()
        self.cfg = cfg
        self._running = True
        self._mode = self.MODE_MAX
        self._target_fps = 30.0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest_samples: dict[str, PendingSample] = {}
        self._sample_windows: dict[str, deque[np.ndarray]] = {}
        self._latest_event_ts: dict[str, float] = {}
        self._last_processed_at: dict[str, float] = {}
        self._ring_order: deque[str] = deque()
        self._dropped_samples = 0
        self._received_since_stats = 0
        self._processed_since_stats = 0
        self._last_stats_time = time.perf_counter()
        self._reset_requested = False
        self._clear_all_requested = False
        self._clear_addresses: set[str] = set()

    def _state_payload(self, state: RingRuntimeState) -> dict:
        snapshot = self.recognizer.snapshot_from_state(state)
        return {
            "last_code": snapshot.last_code,
            "last_name": snapshot.last_name,
            "last_ts": snapshot.last_ts,
            "detected_count": snapshot.detected_count,
        }

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()

    def set_processing_mode(self, mode: str, target_fps: float) -> None:
        normalized_mode = self.MODE_TARGET if mode == self.MODE_TARGET else self.MODE_MAX
        normalized_fps = max(0.1, float(target_fps))
        with self._condition:
            if self._mode != normalized_mode:
                self._latest_samples.clear()
                self._last_processed_at.clear()
            self._mode = normalized_mode
            self._target_fps = normalized_fps
            self._condition.notify_all()

    def enqueue_sample(self, address: str, data: np.ndarray, event_ts: float) -> None:
        sample = PendingSample(event_ts=event_ts)
        with self._condition:
            self._received_since_stats += 1
            window = self._sample_windows.get(address)
            if window is None:
                window = deque(maxlen=self.cfg.window_size)
                self._sample_windows[address] = window
            window.append(data)
            self._latest_event_ts[address] = event_ts
            if address in self._latest_samples:
                self._dropped_samples += 1
            self._latest_samples[address] = sample
            if address not in self._ring_order:
                self._ring_order.append(address)
            self._condition.notify_all()

    def reset_counts(self) -> None:
        with self._condition:
            self._latest_samples.clear()
            self._sample_windows.clear()
            self._latest_event_ts.clear()
            self._last_processed_at.clear()
            self._dropped_samples = 0
            self._received_since_stats = 0
            self._processed_since_stats = 0
            self._last_stats_time = time.perf_counter()
            self._reset_requested = True
            self._condition.notify_all()

    def clear_stream_state(self, addresses: list[str] | None = None) -> None:
        with self._condition:
            if addresses is None:
                self._clear_all_requested = True
                self._clear_addresses.clear()
                self._latest_samples.clear()
                self._sample_windows.clear()
                self._latest_event_ts.clear()
                self._last_processed_at.clear()
            else:
                normalized = {addr for addr in addresses if addr}
                if not normalized:
                    return
                if not self._clear_all_requested:
                    self._clear_addresses.update(normalized)
                for address in normalized:
                    self._latest_samples.pop(address, None)
                    self._sample_windows.pop(address, None)
                    self._latest_event_ts.pop(address, None)
                    self._last_processed_at.pop(address, None)
            self._condition.notify_all()

    def run(self) -> None:
        self.recognizer = InceptionGestureRecognizer(self.cfg)
        self.sig_runtime.emit(self.recognizer.ready, self.recognizer.error_message)
        if not self.recognizer.ready:
            return

        while True:
            sample = self._wait_for_sample()
            if sample is None:
                break
            if sample is self.RESET_MARKER:
                self._apply_reset()
                continue
            if sample is self.CLEAR_MARKER:
                self._apply_stream_clear()
                continue
            address, window_data, event_ts = sample
            code = self.recognizer.detect_window(address, window_data, event_ts)
            state = self.recognizer.state_for(address)
            self._processed_since_stats += 1
            self.sig_result.emit(address, self._state_payload(state), code)
            self._emit_stats_if_needed()

        self._emit_stats(force=True)

    def _apply_reset(self) -> None:
        self.recognizer.reset_counts()
        for address, state in self.recognizer.ring_states.items():
            self.sig_result.emit(address, self._state_payload(state), -1)
        self.sig_stats.emit(
            {
                "recv_fps": 0.0,
                "infer_fps": 0.0,
                "dropped": 0,
                "mode": self._mode,
                "target_fps": self._target_fps,
            }
        )

    def _apply_stream_clear(self) -> None:
        with self._condition:
            clear_all = self._clear_all_requested
            clear_addresses = list(self._clear_addresses)
            self._clear_all_requested = False
            self._clear_addresses.clear()
        if clear_all:
            self.recognizer.clear_stream_state()
            return
        if clear_addresses:
            self.recognizer.clear_stream_state(clear_addresses)

    def _wait_for_sample(self):
        with self._condition:
            while self._running:
                if self._reset_requested:
                    self._reset_requested = False
                    return self.RESET_MARKER
                if self._clear_all_requested or self._clear_addresses:
                    return self.CLEAR_MARKER
                now = time.perf_counter()
                item = self._pop_next_sample_locked(now)
                if item is not None:
                    return item
                wait_timeout = self._next_wait_timeout_locked(now)
                self._condition.wait(timeout=wait_timeout)
            return None

    def _pop_next_sample_locked(
        self, now: float
    ) -> tuple[str, np.ndarray, float] | None:
        if not self._ring_order:
            return None

        # 每个戒指只保留最新待处理样本，优先降低延迟而不是补算旧帧。
        rounds = len(self._ring_order)
        for _ in range(rounds):
            address = self._ring_order[0]
            self._ring_order.rotate(-1)
            sample = self._latest_samples.get(address)
            if sample is None:
                continue
            window = self._sample_windows.get(address)
            if window is None or len(window) < self.cfg.window_size:
                continue
            if self._mode == self.MODE_TARGET:
                last_time = self._last_processed_at.get(address, 0.0)
                interval = 1.0 / self._target_fps
                if now - last_time < interval:
                    continue
                self._last_processed_at[address] = now
            self._latest_samples.pop(address, None)
            window_data = np.asarray(window, dtype=np.float32).T
            return address, window_data, self._latest_event_ts.get(address, sample.event_ts)
        return None

    def _next_wait_timeout_locked(self, now: float) -> float | None:
        if self._mode != self.MODE_TARGET or not self._latest_samples:
            return 0.05

        interval = 1.0 / self._target_fps
        remaining = []
        for address in self._latest_samples:
            last_time = self._last_processed_at.get(address, 0.0)
            remaining.append(max(0.0, interval - (now - last_time)))
        if not remaining:
            return 0.05
        return min(min(remaining), 0.05)

    def _emit_stats_if_needed(self) -> None:
        now = time.perf_counter()
        if now - self._last_stats_time >= self.STATS_INTERVAL:
            self._emit_stats(force=True)

    def _emit_stats(self, force: bool = False) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_stats_time
        if not force and elapsed < self.STATS_INTERVAL:
            return
        elapsed = max(elapsed, 1e-6)
        self.sig_stats.emit(
            {
                "recv_fps": self._received_since_stats / elapsed,
                "infer_fps": self._processed_since_stats / elapsed,
                "dropped": self._dropped_samples,
                "mode": self._mode,
                "target_fps": self._target_fps,
            }
        )
        self._received_since_stats = 0
        self._processed_since_stats = 0
        self._last_stats_time = now


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

        self.status_label = QLabel("数据流状态：未连接")
        self.status_label.setObjectName("statusTag")
        self.total_count_label = QLabel("总识别次数: 0")
        self.rate_label = QLabel("接收: --Hz | 识别: --Hz | 覆盖: 0")
        self.rate_label.setProperty("class", "sub-text")
        self.total_count = 0

        self._build_ui()
        self.set_stream_status_label(self.status_label)
        self._bind_signals()
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

        title = QLabel("手势识别可视化（InceptionTime）")
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
        action_row.addWidget(self.apply_btn)
        action_row.addWidget(self.reset_btn)
        action_row.addStretch()
        panel_layout.addLayout(action_row)

        settings_panel = QFrame()
        settings_panel.setProperty("class", "settings-panel")
        settings_layout = QGridLayout(settings_panel)
        settings_layout.setContentsMargins(14, 12, 14, 12)
        settings_layout.setHorizontalSpacing(16)
        settings_layout.setVerticalSpacing(12)

        mode_title = QLabel("处理模式")
        mode_title.setProperty("class", "settings-title")
        mode_desc = QLabel("控制识别线程是持续吃满最新样本，还是按指定速率运行。")
        mode_desc.setProperty("class", "settings-desc")
        mode_desc.setWordWrap(True)
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
        mode_text.addWidget(mode_desc)

        fps_text = QVBoxLayout()
        fps_text.setSpacing(4)
        fps_text.addWidget(fps_title)
        fps_text.addWidget(fps_desc)

        settings_layout.addLayout(mode_text, 0, 0)
        settings_layout.addLayout(mode_switch, 0, 1)
        settings_layout.addLayout(fps_text, 1, 0)
        settings_layout.addWidget(self.fps_spin, 1, 1)
        settings_layout.setColumnStretch(0, 1)

        self.mode_hint = QLabel("识别始终优先使用最新数据；旧样本会被覆盖以降低延迟。")
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

    def on_worker_stats(self, payload: dict):
        recv_fps = float(payload.get("recv_fps", 0.0) or 0.0)
        infer_fps = float(payload.get("infer_fps", 0.0) or 0.0)
        dropped = int(payload.get("dropped", 0) or 0)
        self.rate_label.setText(
            f"接收: {recv_fps:.1f}Hz | 识别: {infer_fps:.1f}Hz | 覆盖: {dropped}"
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
        if message.get("type") != "imu":
            return
        if not self._worker_ready:
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
        self.worker.enqueue_sample(address, data, event_ts)

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
        self.send_subscription(["imu"])
        self.refresh_cards()

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
    win = GestureRecognitionWindow(host=args.host, port=args.port)
    win.show()
    sys.exit(app.exec())
