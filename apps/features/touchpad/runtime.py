from __future__ import annotations

"""Reusable touchpad gesture + motion runtime shared by standalone and unified apps."""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from screeninfo import get_monitors
from tsai.models.InceptionTime import InceptionTime

from models.tcn_lstm import TCNLSTMStreaming
from utils.cd_ratio import CDRatioEstimator, OneEuroFilter
from utils.imu_processor import IMUProcessor
from utils.window import Window

try:
    from pynput import mouse

    HAS_TOUCHPAD_CURSOR_CONTROL = True
    TOUCHPAD_CURSOR_CONTROL_ERROR = ""
except Exception as exc:
    mouse = None
    HAS_TOUCHPAD_CURSOR_CONTROL = False
    TOUCHPAD_CURSOR_CONTROL_ERROR = str(exc)


@dataclass(frozen=True, slots=True)
class TouchpadConfig:
    gesture_ckpt: Path
    motion_ckpt: Path
    gesture_window_size: int = 20
    gesture_calculate_interval: int = 2
    gesture_history_size: int = 5
    threshold_down: int = 1
    threshold_up: int = 2
    click_interval_seconds: float = 0.1
    idle_gyro_threshold: float = 0.03
    min_move_abs: float = 0.05
    move_emit_interval_seconds: float = 1 / 30.0


@dataclass(frozen=True, slots=True)
class TouchpadEvent:
    name: str
    data: dict[str, float]
    event_ts: float


@dataclass(slots=True)
class PendingImuReading:
    address: str
    data: np.ndarray
    event_ts: float = 0.0


@dataclass(slots=True)
class TouchpadState:
    gesture_detector: "GestureDetector"
    motion_model: TCNLSTMStreaming
    imu_processor: IMUProcessor
    cd_ratio_estimator: CDRatioEstimator
    filter_x: OneEuroFilter
    filter_y: OneEuroFilter
    imu_data: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=1))
    last_gesture: int = 3
    last_down_time: float = 0.0
    last_move_emit_time: float = 0.0
    packet_count: int = 0


class GestureDetector:
    def __init__(self, model: InceptionTime, config: TouchpadConfig, device: torch.device):
        self.device = device
        self.model = model
        self.config = config
        self.data_buffer: deque[np.ndarray] = deque()
        self.imu_window: Window[np.ndarray] = Window(config.gesture_window_size)
        self.cur_time_state = 0
        self.history_queue = deque(maxlen=config.gesture_history_size)

    def calculate(self, data: np.ndarray) -> int:
        self.data_buffer.append(data)
        while self.data_buffer:
            self.imu_window.push(self.data_buffer.popleft())
        data_window = self.imu_window.map(lambda item: item).to_numpy().reshape(-1, 6)
        if data_window.shape[0] != self.config.gesture_window_size:
            return 0
        if self.cur_time_state < self.config.gesture_calculate_interval:
            self.cur_time_state += 1
            return 0
        self.cur_time_state = 0
        return self._detect(data_window)

    def _detect(self, data: np.ndarray) -> int:
        outputs = self.model(
            torch.from_numpy(data.T).float().view(1, 6, self.config.gesture_window_size).to(self.device)
        )
        pred = int(torch.argmax(outputs, dim=1).detach().cpu().item())
        self.history_queue.append(pred)

        if self.history_queue.count(0) >= self.config.threshold_down:
            self.history_queue.clear()
            return 4
        if self.history_queue.count(1) >= self.config.threshold_up:
            self.history_queue.clear()
            return 3
        return 0


class TouchpadEngine:
    """Per-ring touchpad engine that can emit events or drive the local cursor."""

    def __init__(self, config: TouchpadConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gesture_state_dict = self._load_gesture_state_dict(config.gesture_ckpt)
        self.motion_state_dict = self._load_motion_state_dict(config.motion_ckpt)
        self.states: dict[str, TouchpadState] = {}
        self.cursor_control_enabled = False
        self.mouse_controller = mouse.Controller() if HAS_TOUCHPAD_CURSOR_CONTROL else None
        self.last_status = "等待 IMU 数据"

        monitors = []
        if HAS_TOUCHPAD_CURSOR_CONTROL:
            try:
                monitors = get_monitors()
            except Exception:
                monitors = []
        if monitors:
            self.left = min(m.x for m in monitors)
            self.top = min(m.y for m in monitors)
            self.right = max(m.x + m.width for m in monitors)
            self.bottom = max(m.y + m.height for m in monitors)
        else:
            self.left = -10**9
            self.top = -10**9
            self.right = 10**9
            self.bottom = 10**9

    def clear(self) -> None:
        self.clear_states(None)

    def clear_states(self, addresses: list[str] | None = None) -> None:
        if addresses is None:
            self.states.clear()
            self.last_status = "状态已清空"
            return
        for address in addresses:
            self.states.pop(address, None)

    def set_cursor_control_enabled(self, enabled: bool) -> None:
        self.cursor_control_enabled = bool(enabled) and self.mouse_controller is not None

    def snapshot(self) -> dict[str, object]:
        return {
            "ring_count": len(self.states),
            "last_status": self.last_status,
            "active_rings": [addr for addr, state in self.states.items() if state.last_gesture == 4],
            "packet_count": sum(state.packet_count for state in self.states.values()),
        }

    def process_imu(
        self,
        address: str,
        data: np.ndarray,
        event_ts: float,
    ) -> list[TouchpadEvent]:
        state = self._state_for(address)
        state.packet_count += 1

        linear_data = np.asarray(data, dtype=np.float32).copy()
        if state.imu_processor.pre_timestamp == 0:
            state.imu_processor.init_quaternion_and_timestamp(linear_data[:3], timestamp=1)
        else:
            state.imu_processor.update_attitude_by_data(linear_data[3:6], linear_data[:3])
        linear_data[:3] = state.imu_processor.calculate_linear_acceleration(linear_data[:3])
        state.imu_data.append(linear_data)

        events: list[TouchpadEvent] = []
        current_gesture = state.gesture_detector.calculate(linear_data)
        if current_gesture != 0:
            events.extend(self._handle_gesture_transition(address, state, current_gesture, event_ts))

        if len(state.imu_data) < state.imu_data.maxlen:
            return events

        merged_data = torch.tensor(np.array(state.imu_data), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = state.motion_model(merged_data.unsqueeze(0))
        outputs = output.detach().cpu().numpy()
        if outputs.shape != (1, 1, 2):
            raise RuntimeError(f"unexpected touchpad output shape: {outputs.shape}")

        result = outputs[0]
        if np.mean(np.abs(linear_data[3:6])) < self.config.idle_gyro_threshold:
            result = np.array([[0.0, 0.0]], dtype=np.float32)

        if state.last_gesture == 4:
            if self.cursor_control_enabled:
                self._move_mouse(state, result)
            events.extend(self._build_move_events(state, result, event_ts))
        return events

    def _handle_gesture_transition(
        self,
        address: str,
        state: TouchpadState,
        current_gesture: int,
        event_ts: float,
    ) -> list[TouchpadEvent]:
        events: list[TouchpadEvent] = []
        now = time.time()
        if current_gesture == 4:
            state.last_down_time = now
            if state.last_gesture != 4:
                self.last_status = f"{address} Down"
                events.append(TouchpadEvent("touch-down", {}, event_ts))
        elif current_gesture == 3:
            if state.last_gesture != 3:
                self.last_status = f"{address} Up"
                events.append(TouchpadEvent("touch-up", {}, event_ts))
            if (
                self.cursor_control_enabled
                and self.mouse_controller is not None
                and state.last_gesture == 4
                and now - state.last_down_time < self.config.click_interval_seconds
            ):
                try:
                    self.mouse_controller.click(mouse.Button.left)
                except Exception:
                    pass
                self.last_status = f"{address} Click"
                events.append(TouchpadEvent("click", {}, event_ts))
        state.last_gesture = current_gesture
        return events

    def _build_move_events(
        self,
        state: TouchpadState,
        outputs: np.ndarray,
        event_ts: float,
    ) -> list[TouchpadEvent]:
        events: list[TouchpadEvent] = []
        for output in outputs:
            vx = float(output[0]) / 10.0 * 9.0
            vy = float(output[1]) / 10.0 * 9.0
            velocity_norm = float(np.linalg.norm([vx, vy]))
            cd_ratio = state.cd_ratio_estimator.update(velocity_norm)
            vx = float(state.filter_x.filter(vx))
            vy = float(state.filter_y.filter(vy))
            vx *= cd_ratio
            vy *= cd_ratio

            now = time.perf_counter()
            if now - state.last_move_emit_time < self.config.move_emit_interval_seconds:
                continue
            if abs(vx) < self.config.min_move_abs and abs(vy) < self.config.min_move_abs:
                continue

            state.last_move_emit_time = now
            self.last_status = f"移动鼠标 ({vx:.2f}, {vy:.2f})"
            events.append(TouchpadEvent("move", {"move_x": vx, "move_y": vy}, event_ts))
        return events

    def _move_mouse(self, state: TouchpadState, outputs: np.ndarray) -> None:
        if self.mouse_controller is None:
            return
        for output in outputs:
            vx = float(output[0]) / 10.0 * 9.0
            vy = float(output[1]) / 10.0 * 9.0
            velocity_norm = float(np.linalg.norm([vx, vy]))
            cd_ratio = state.cd_ratio_estimator.update(velocity_norm)
            vx = float(state.filter_x.filter(vx))
            vy = float(state.filter_y.filter(vy))
            vx *= cd_ratio
            vy *= cd_ratio

            cx, cy = self.mouse_controller.position
            if cx + vx < self.left:
                vx = self.left - cx
            elif cx + vx > self.right:
                vx = self.right - cx
            if cy + vy < self.top:
                vy = self.top - cy
            elif cy + vy > self.bottom:
                vy = self.bottom - cy
            try:
                self.mouse_controller.move(vx, vy)
            except Exception:
                return

    def _state_for(self, address: str) -> TouchpadState:
        state = self.states.get(address)
        if state is not None:
            return state

        gesture_model = InceptionTime(6, 4, nf=16).to(self.device)
        gesture_model.load_state_dict(self.gesture_state_dict)
        gesture_model.eval()

        motion_model = TCNLSTMStreaming(6, 128, 2, [64, 128], 3, 0.2).to(self.device)
        motion_model.load_state_dict(self.motion_state_dict)
        motion_model.eval()

        state = TouchpadState(
            gesture_detector=GestureDetector(gesture_model, self.config, self.device),
            motion_model=motion_model,
            imu_processor=IMUProcessor(frequency=200, core="VQF"),
            cd_ratio_estimator=CDRatioEstimator(min_cd=0.3, max_cd=2.0, c=0.2),
            filter_x=OneEuroFilter(freq=200, min_cutoff=0.0004, beta=0.08),
            filter_y=OneEuroFilter(freq=200, min_cutoff=0.0004, beta=0.08),
        )
        self.states[address] = state
        return state

    def _load_gesture_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"gesture checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(f"invalid gesture checkpoint: {checkpoint_path}")
        return checkpoint["model"]

    def _load_motion_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"motion checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"invalid motion checkpoint: {checkpoint_path}")
        return checkpoint


class TouchpadWorker:
    """Background worker for the standalone touchpad window."""

    def __init__(self, config: TouchpadConfig):
        self.engine = TouchpadEngine(config)
        self.pending_readings: deque[PendingImuReading] = deque(maxlen=4096)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def enqueue(self, address: str, data: np.ndarray, *, event_ts: float = 0.0) -> None:
        with self._condition:
            self.pending_readings.append(
                PendingImuReading(
                    address=address,
                    data=np.asarray(data, dtype=np.float32),
                    event_ts=float(event_ts),
                )
            )
            self._condition.notify()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            pending_count = len(self.pending_readings)
        snapshot = self.engine.snapshot()
        snapshot["pending_count"] = pending_count
        return snapshot

    def clear_states(self, addresses: list[str] | None = None) -> None:
        with self._condition:
            self.engine.clear_states(addresses)
            if addresses is None:
                self.pending_readings.clear()
            else:
                blocked = set(addresses)
                self.pending_readings = deque(
                    [item for item in self.pending_readings if item.address not in blocked],
                    maxlen=4096,
                )

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and not self.pending_readings:
                    self._condition.wait(timeout=0.2)
                if not self._running:
                    return
                reading = self.pending_readings.popleft()
            self.engine.process_imu(reading.address, reading.data, reading.event_ts)
