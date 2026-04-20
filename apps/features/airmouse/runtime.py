from __future__ import annotations

"""Reusable airmouse runtime shared by unified gesture modes."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.cd_ratio import OneEuroFilter
from utils.imu_processor import IMUProcessor

try:
    from pyquaternion import Quaternion as PyQuaternion

    HAS_AIRMOUSE_RUNTIME = True
    AIRMOUSE_RUNTIME_ERROR = ""
except Exception as exc:
    PyQuaternion = None
    HAS_AIRMOUSE_RUNTIME = False
    AIRMOUSE_RUNTIME_ERROR = str(exc)


@dataclass(frozen=True, slots=True)
class AirmouseConfig:
    pointer_scale: float = 4000.0
    min_move_abs: float = 0.05
    move_emit_interval_seconds: float = 1 / 60.0
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.2
    filter_d_cutoff: float = 1.0
    cursor_home_x_ratio: float = 0.5
    cursor_home_y_ratio: float = 0.5


@dataclass(frozen=True, slots=True)
class AirmouseEvent:
    name: str
    data: dict[str, float]
    event_ts: float


@dataclass(slots=True)
class AirmouseState:
    imu_processor: IMUProcessor
    filter_x: OneEuroFilter
    filter_y: OneEuroFilter
    orientation_queue: deque = field(default_factory=lambda: deque(maxlen=2))
    last_move_emit_time: float = 0.0


class AirmouseEngine:
    def __init__(self, config: AirmouseConfig):
        if not HAS_AIRMOUSE_RUNTIME:
            raise RuntimeError(AIRMOUSE_RUNTIME_ERROR)
        self.config = config
        self.state = self._build_state()

    def clear(self) -> None:
        self.state = self._build_state()

    def process_imu(
        self,
        _address: str,
        data: np.ndarray,
        event_ts: float,
    ) -> list[AirmouseEvent]:
        state = self.state
        timestamp_ms = event_ts * 1000.0
        if state.imu_processor.pre_timestamp == 0:
            state.imu_processor.init_quaternion_and_timestamp(data[:3], timestamp_ms)
            return []

        state.imu_processor.update_attitude_by_data(data[3:], data[:3], timestamp=timestamp_ms)
        quaternion = np.asarray(state.imu_processor.quaternion, dtype=np.float64)
        current_orientation = PyQuaternion(quaternion)
        state.orientation_queue.append(current_orientation)
        if len(state.orientation_queue) < state.orientation_queue.maxlen:
            return []

        delta_orientation = state.orientation_queue[0].inverse * current_orientation
        delta_orientation = np.array(delta_orientation.yaw_pitch_roll, dtype=np.float64)
        x_move, y_move = self._ring_to_world_motion(delta_orientation, current_orientation)
        x_move *= self.config.pointer_scale
        y_move *= self.config.pointer_scale
        vx = float(state.filter_x.filter(x_move))
        vy = float(state.filter_y.filter(y_move))
        state.orientation_queue.clear()
        state.orientation_queue.append(current_orientation)

        now = time.perf_counter()
        if now - state.last_move_emit_time < self.config.move_emit_interval_seconds:
            return []
        if abs(vx) < self.config.min_move_abs and abs(vy) < self.config.min_move_abs:
            return []

        state.last_move_emit_time = now
        return [AirmouseEvent("move", {"move_x": vx, "move_y": vy}, event_ts)]

    def _build_state(self) -> AirmouseState:
        return AirmouseState(
            imu_processor=IMUProcessor(frequency=200, core="VQF"),
            filter_x=OneEuroFilter(
                freq=200,
                min_cutoff=self.config.filter_min_cutoff,
                beta=self.config.filter_beta,
                d_cutoff=self.config.filter_d_cutoff,
            ),
            filter_y=OneEuroFilter(
                freq=200,
                min_cutoff=self.config.filter_min_cutoff,
                beta=self.config.filter_beta,
                d_cutoff=self.config.filter_d_cutoff,
            ),
        )

    def _ring_to_world_motion(
        self,
        delta_orientation: np.ndarray,
        current_orientation: Any,
    ) -> tuple[float, float]:
        ring_yaw_change, ring_pitch_change, _ring_roll_change = delta_orientation
        rotated_y = current_orientation.rotate(np.array([0.0, 1.0, 0.0]))
        rotated_z = current_orientation.rotate(np.array([0.0, 0.0, 1.0]))
        x_move = ring_yaw_change * rotated_z[2] + ring_pitch_change * rotated_y[2]
        y_move = ring_yaw_change * rotated_y[2] - ring_pitch_change * rotated_z[2]
        return -float(x_move), -float(y_move)
