from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from apps.shared.constants import GESTURE_CODE_TO_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    from openvino import Core

    HAS_MODEL_RUNTIME = True
    MODEL_RUNTIME_ERROR = ""
except Exception as exc:
    Core = None
    HAS_MODEL_RUNTIME = False
    MODEL_RUNTIME_ERROR = str(exc)


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


class InceptionGestureRecognizer:
    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.ready = False
        self.error_message = ""
        self.models = []
        self.model_pointer = 0
        self.ring_states: dict[str, RingRuntimeState] = {}
        self.allowed_output_indices = [
            i for i in range(1, 39) if i not in cfg.allowed_gestures
        ]

        if not HAS_MODEL_RUNTIME:
            self.error_message = f"openvino unavailable: {MODEL_RUNTIME_ERROR}"
            return

        try:
            self.device = cfg.device.upper()
            core = Core()

            for model_path_str in cfg.models:
                model_path = PROJECT_ROOT / model_path_str
                if not model_path.exists():
                    raise FileNotFoundError(f"model not found: {model_path}")
                compiled = core.compile_model(
                    model=str(model_path),
                    device_name=self.device,
                )
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
                result_windows=[
                    deque(maxlen=self.cfg.result_window_length)
                    for _ in self.models
                ],
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
        input_data = data.reshape(1, 6, self.cfg.window_size).astype(
            np.float32,
            copy=False,
        )
        outputs = model["compiled"]({model["input"]: input_data})[model["output"]]
        outputs = np.asarray(outputs, dtype=np.float32)
        outputs[:, self.allowed_output_indices] = -1e9
        return int(np.argmax(outputs, axis=1)[0])

    def _collect_model_predictions(
        self,
        data: np.ndarray,
        state: RingRuntimeState,
    ) -> None:
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

    def _detect_gesture_with_window(
        self,
        data: np.ndarray,
        state: RingRuntimeState,
    ) -> int:
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

    def _clear_runtime_state(
        self,
        state: RingRuntimeState,
        *,
        preserve_count: bool,
    ) -> None:
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

    def detect_window(
        self,
        address: str,
        window_data: np.ndarray,
        event_ts: float,
    ) -> int:
        if not self.ready:
            return -1

        state = self.state_for(address)
        gesture = self._detect_gesture_with_window(window_data, state)

        if gesture == 16 and float(np.max(window_data) - np.min(window_data)) < 25.0:
            gesture = -2

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
            state.last_name = GESTURE_CODE_TO_NAME.get(
                gesture,
                f"gesture-{gesture}",
            )
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
        self,
        now: float,
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

    def _flatten_stream_after_detection(
        self,
        address: str,
        last_frame: np.ndarray,
    ) -> None:
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
                        PendingReading(
                            data=flattened_frame.copy(),
                            event_ts=reading.event_ts,
                        )
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
