from __future__ import annotations

"""Desktop-control helper used by gesture-recognition style apps."""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from apps.shared.constants import TOUCH_CODE_TO_NAME

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


@dataclass(slots=True)
class PCControlState:
    imu_worker: object
    one_euro_filters: list[object]
    orientation_queue: deque
    position: list[float]
    is_moving: bool = False
    task_called: bool = False


@dataclass(slots=True)
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
        monitors = get_monitors()
        self.left = min(m.x for m in monitors)
        self.top = min(m.y for m in monitors)
        self.right = max(m.x + m.width for m in monitors)
        self.bottom = max(m.y + m.height for m in monitors)
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
        _ = TOUCH_CODE_TO_NAME.get(action_code, "")
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
                PendingPCReading(
                    address=address,
                    data=np.asarray(data, dtype=np.float32),
                    event_ts=event_ts,
                )
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
        if state is not None:
            return state
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
