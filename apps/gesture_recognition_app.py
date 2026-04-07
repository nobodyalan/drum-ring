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
from common.constants import GESTURE_CODE_TO_NAME

try:
    from pynput import keyboard, mouse
    from pyquaternion import Quaternion as PyQuaternion
    from screeninfo import get_monitors

    from utils.cd_ratio import OneEuroFilter
    from utils.imu_processor import IMUProcessor

    HAS_PC_CONTROL = True
    PC_CONTROL_ERROR = ""
except Exception as exc:
    keyboard = None
    mouse = None
    PyQuaternion = None
    get_monitors = None
    OneEuroFilter = None
    IMUProcessor = None
    HAS_PC_CONTROL = False
    PC_CONTROL_ERROR = str(exc)

try:
    from openvino import Core

    HAS_MODEL_RUNTIME = True
except Exception as exc:
    Core = None
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
        result_window_length=20,
        window_size=200,
        models=[
            "checkpoint/it-pinch/it-pinch.xml",
            "checkpoint/it-snap/it-snap.xml",
        ],
        models_allowed_gestures=[
            [8, 13, 14],
            [16, 30, 31, 34, 35, 36],
        ],
        device="CPU",
        merge_window_seconds=0.5,
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
class PendingReading:
    data: np.ndarray
    event_ts: float


@dataclass
class PCControlState:
    imu_worker: object
    one_euro_filters: list[object]
    orientation_queue: deque
    position: list[float]
    is_moving: bool = False
    task_called: bool = False


@dataclass
class PendingPCReading:
    address: str
    data: np.ndarray
    event_ts: float


class RingPCController:
    IMU_FREQ = 200
    POINTER_SCALE = 2000
    QUEUE_SIZE = 2
    LOOP_INTERVAL = 1 / 300

    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = True
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        self.states: dict[str, PCControlState] = {}
        self.pending_readings: deque[PendingPCReading] = deque(maxlen=2048)
        mons = get_monitors()
        self.left = min(m.x for m in mons)
        self.top = min(m.y for m in mons)
        self.right = max(m.x + m.width for m in mons)
        self.bottom = max(m.y + m.height for m in mons)
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def clear_states(self, addresses: list[str] | None = None) -> None:
        with self._lock:
            if addresses is None:
                self.states.clear()
                self.pending_readings.clear()
                return
            address_set = set(addresses)
            for address in addresses:
                self.states.pop(address, None)
            if self.pending_readings:
                self.pending_readings = deque(
                    [item for item in self.pending_readings if item.address not in address_set],
                    maxlen=2048,
                )

    def disable(self) -> None:
        with self._lock:
            for state in self.states.values():
                state.is_moving = False
                state.task_called = False
        self._release_task_switch_if_needed()

    def handle_touch(self, address: str, action_code: int) -> None:
        with self._lock:
            state = self._state_for(address)
            if action_code == 2:
                state.position = list(self.mouse_controller.position)
                state.is_moving = not state.is_moving
            elif action_code == 5:
                state.is_moving = False

    def handle_gesture(self, address: str, gesture_code: int) -> None:
        with self._lock:
            state = self._state_for(address)
            if gesture_code == 8:
                self.mouse_controller.click(mouse.Button.left)
                if state.task_called:
                    state.task_called = False
                    self._release_task_switch_if_needed()
            elif gesture_code == 14:
                self.keyboard_controller.press(keyboard.Key.cmd)
                self.keyboard_controller.tap(keyboard.Key.left)
            elif gesture_code == 16:
                self.keyboard_controller.tap(keyboard.Key.enter)
                if state.task_called:
                    state.task_called = False
                    self._release_task_switch_if_needed()
            elif gesture_code == 13:
                self.keyboard_controller.press(keyboard.Key.cmd)
                self.keyboard_controller.tap(keyboard.Key.tab)
                state.task_called = True

    def handle_imu(self, address: str, data: np.ndarray, event_ts: float) -> None:
        with self._lock:
            self.pending_readings.append(
                PendingPCReading(address=address, data=np.asarray(data, dtype=np.float32), event_ts=event_ts)
            )
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self.worker_thread.join(timeout=1.5)
        self.disable()

    def _state_for(self, address: str) -> PCControlState:
        state = self.states.get(address)
        if state is None:
            state = PCControlState(
                imu_worker=IMUProcessor(frequency=self.IMU_FREQ, core="VQF"),
                one_euro_filters=[
                    OneEuroFilter(freq=self.IMU_FREQ, min_cutoff=1.0, beta=0.2, d_cutoff=1.0),
                    OneEuroFilter(freq=self.IMU_FREQ, min_cutoff=1.0, beta=0.2, d_cutoff=1.0),
                ],
                orientation_queue=deque(maxlen=self.QUEUE_SIZE),
                position=list(self.mouse_controller.position),
            )
            self.states[address] = state
        return state

    def _release_task_switch_if_needed(self) -> None:
        try:
            self.keyboard_controller.release(keyboard.Key.cmd)
        except Exception:
            pass

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and not self.pending_readings:
                    self._condition.wait(timeout=self.LOOP_INTERVAL)
                if not self._running:
                    return
                reading = self.pending_readings.popleft()
            self._process_reading(reading)
            time.sleep(self.LOOP_INTERVAL)

    def _process_reading(self, reading: PendingPCReading) -> None:
        with self._lock:
            state = self._state_for(reading.address)
            timestamp_ms = reading.event_ts * 1000.0
            if state.imu_worker.pre_timestamp == 0:
                state.imu_worker.init_quaternion_and_timestamp(reading.data[:3], timestamp_ms)
                return

            state.imu_worker.update_attitude_by_data(
                reading.data[3:],
                reading.data[:3],
                timestamp=timestamp_ms,
            )
            quaternion = np.asarray(state.imu_worker.quaternion, dtype=np.float64)
            current_orientation = PyQuaternion(quaternion)
            state.orientation_queue.append(current_orientation)
            if len(state.orientation_queue) < self.QUEUE_SIZE:
                return

            delta_orientation = state.orientation_queue[0].inverse * current_orientation
            delta_orientation = np.array(delta_orientation.yaw_pitch_roll, dtype=np.float64)
            x_move, y_move = self._ring_to_world_motion(delta_orientation, current_orientation)
            x_move *= self.POINTER_SCALE
            y_move *= self.POINTER_SCALE
            vx = float(state.one_euro_filters[0].filter(x_move))
            vy = float(state.one_euro_filters[1].filter(y_move))

            if state.is_moving:
                cx, cy = self.mouse_controller.position
                if cx + vx < self.left:
                    vx = self.left - cx
                elif cx + vx > self.right:
                    vx = self.right - cx
                if cy + vy < self.top:
                    vy = self.top - cy
                elif cy + vy > self.bottom:
                    vy = self.bottom - cy
                state.position[0] += vx
                state.position[1] += vy
                self.mouse_controller.position = (state.position[0], state.position[1])

            state.orientation_queue.clear()
            state.orientation_queue.append(current_orientation)

    def _ring_to_world_motion(
        self,
        delta_orientation: np.ndarray,
        current_orientation: object,
        mode: str = "pointing",
    ) -> tuple[float, float]:
        ring_yaw_change, ring_pitch_change, ring_roll_change = delta_orientation
        rotated_y = current_orientation.rotate(np.array([0.0, 1.0, 0.0]))
        rotated_z = current_orientation.rotate(np.array([0.0, 0.0, 1.0]))

        if mode == "pointing":
            x_move = ring_yaw_change * rotated_z[2] + ring_pitch_change * rotated_y[2]
            y_move = ring_yaw_change * rotated_y[2] - ring_pitch_change * rotated_z[2]
        elif mode == "fist":
            x_move = ring_yaw_change * rotated_z[2] + ring_pitch_change * rotated_y[2]
            y_move = -ring_roll_change
        else:
            raise ValueError("Invalid mode")

        return -float(x_move), -float(y_move)


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
            self.error_message = f"openvino unavailable: {MODEL_RUNTIME_ERROR}"
            return

        try:
            self.device = cfg.device.upper()
            core = Core()

            for model_path_str in cfg.models:
                model_path = APP_ROOT / model_path_str
                if not model_path.exists():
                    raise FileNotFoundError(f"model not found: {model_path}")
                compiled = core.compile_model(model=str(model_path), device_name=self.device)
                self.models.append(
                    {
                        "compiled": compiled,
                        "input": compiled.input(0),
                        "output": compiled.output(0),
                    }
                )

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

    def _predict_one_model(self, model: dict, data: np.ndarray) -> int:
        input_data = data.reshape(1, 6, self.cfg.window_size).astype(np.float32, copy=False)
        outputs = model["compiled"]({model["input"]: input_data})[model["output"]]
        outputs = np.asarray(outputs, dtype=np.float32)
        outputs[:, self.allowed_output_indices] = -1e9
        return int(np.argmax(outputs, axis=1)[0])

    def _collect_model_predictions(self, data: np.ndarray, state: RingRuntimeState) -> None:
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
    MAX_ACCEPTABLE_BACKLOG = 10
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
        self._pending_readings: dict[str, deque[PendingReading]] = {}
        self._reading_windows: dict[str, deque[np.ndarray]] = {}
        self._last_processed_at: dict[str, float] = {}
        self._ring_order: deque[str] = deque()
        self._dropped_readings = 0
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
                self._last_processed_at.clear()
            self._mode = normalized_mode
            self._target_fps = normalized_fps
            self._condition.notify_all()

    def enqueue_reading(self, address: str, data: np.ndarray, event_ts: float) -> None:
        reading = PendingReading(data=data, event_ts=event_ts)
        with self._condition:
            self._received_since_stats += 1
            pending = self._pending_readings.get(address)
            if pending is None:
                pending = deque()
                self._pending_readings[address] = pending
            pending.append(reading)
            if address not in self._ring_order:
                self._ring_order.append(address)
            self._condition.notify_all()

    def reset_counts(self) -> None:
        with self._condition:
            self._pending_readings.clear()
            self._reading_windows.clear()
            self._last_processed_at.clear()
            self._dropped_readings = 0
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
                self._pending_readings.clear()
                self._reading_windows.clear()
                self._last_processed_at.clear()
            else:
                normalized = {addr for addr in addresses if addr}
                if not normalized:
                    return
                if not self._clear_all_requested:
                    self._clear_addresses.update(normalized)
                for address in normalized:
                    self._pending_readings.pop(address, None)
                    self._reading_windows.pop(address, None)
                    self._last_processed_at.pop(address, None)
            self._condition.notify_all()

    def run(self) -> None:
        self.recognizer = InceptionGestureRecognizer(self.cfg)
        self.sig_runtime.emit(self.recognizer.ready, self.recognizer.error_message)
        if not self.recognizer.ready:
            return

        while True:
            reading = self._wait_for_reading()
            if reading is None:
                break
            if reading is self.RESET_MARKER:
                self._apply_reset()
                continue
            if reading is self.CLEAR_MARKER:
                self._apply_stream_clear()
                continue
            address, batch = reading
            for window_data, event_ts in batch:
                code = self.recognizer.detect_window(address, window_data, event_ts)
                state = self.recognizer.state_for(address)
                self._processed_since_stats += 1
                self.sig_result.emit(address, self._state_payload(state), code)
                if code >= 0:
                    self._flatten_stream_after_detection(address, window_data[:, -1])
                    break
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

    def _wait_for_reading(self):
        with self._condition:
            while self._running:
                if self._reset_requested:
                    self._reset_requested = False
                    return self.RESET_MARKER
                if self._clear_all_requested or self._clear_addresses:
                    return self.CLEAR_MARKER
                now = time.perf_counter()
                item = self._pop_next_reading_locked(now)
                if item is not None:
                    return item
                wait_timeout = self._next_wait_timeout_locked(now)
                self._condition.wait(timeout=wait_timeout)
            return None

    def _pop_next_reading_locked(
        self, now: float
    ) -> tuple[str, list[tuple[np.ndarray, float]]] | None:
        if not self._ring_order:
            return None

        rounds = len(self._ring_order)
        for _ in range(rounds):
            address = self._ring_order[0]
            self._ring_order.rotate(-1)
            pending = self._pending_readings.get(address)
            if not pending:
                continue
            take_count = 1 if len(pending) <= self.MAX_ACCEPTABLE_BACKLOG else len(pending)
            window = self._reading_windows.get(address)
            window_len = 0 if window is None else len(window)
            if self._mode == self.MODE_TARGET:
                # Warm-up stage should keep filling the sliding window without FPS throttling.
                if window_len + take_count < self.cfg.window_size:
                    pass
                else:
                    last_time = self._last_processed_at.get(address, 0.0)
                    interval = 1.0 / self._target_fps
                    if now - last_time < interval:
                        continue
            if window is None:
                window = deque(maxlen=self.cfg.window_size)
                self._reading_windows[address] = window
            batch: list[tuple[np.ndarray, float]] = []
            latest_window: tuple[np.ndarray, float] | None = None
            for _ in range(take_count):
                reading = pending.popleft()
                window.append(reading.data)
                if len(window) < self.cfg.window_size:
                    continue
                snapshot = (np.asarray(window, dtype=np.float32).T, reading.event_ts)
                if self._mode == self.MODE_TARGET:
                    latest_window = snapshot
                else:
                    batch.append(snapshot)
            if self._mode == self.MODE_TARGET and latest_window is not None:
                batch.append(latest_window)
            if not pending:
                self._pending_readings.pop(address, None)
            if batch:
                if self._mode == self.MODE_TARGET:
                    self._last_processed_at[address] = now
                return address, batch
        return None

    def _flatten_stream_after_detection(self, address: str, last_frame: np.ndarray) -> None:
        flattened_frame = np.asarray(last_frame, dtype=np.float32).copy()
        with self._condition:
            window = self._reading_windows.get(address)
            if window is not None and window:
                maxlen = window.maxlen
                flattened_window = deque(maxlen=maxlen)
                for _ in range(len(window)):
                    flattened_window.append(flattened_frame.copy())
                self._reading_windows[address] = flattened_window

            pending = self._pending_readings.get(address)
            if pending:
                flattened_pending = deque()
                for reading in pending:
                    flattened_pending.append(
                        PendingReading(data=flattened_frame.copy(), event_ts=reading.event_ts)
                    )
                self._pending_readings[address] = flattened_pending

    def _next_wait_timeout_locked(self, now: float) -> float | None:
        if self._mode != self.MODE_TARGET or not self._pending_readings:
            return 0.001

        interval = 1.0 / self._target_fps
        remaining = []
        for address in self._pending_readings:
            last_time = self._last_processed_at.get(address, 0.0)
            remaining.append(max(0.0, interval - (now - last_time)))
        if not remaining:
            return 0.001
        return min(min(remaining), 0.001)

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
                "dropped": self._dropped_readings,
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
