from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from apps.features.window_classifier.swipe import SWIPE_SPEC, build_swipe_recognizer

IMU_SAMPLE_RATE_HZ = 200.0
SWIPE_BACKEND = "openvino"


@dataclass(frozen=True, slots=True)
class SwipeRuntimeConfig:
    window_size: int = SWIPE_SPEC.default_window_size
    step_frames: int = SWIPE_SPEC.default_step_frames
    stable_window_seconds: float = SWIPE_SPEC.default_stable_window_seconds
    positive_ratio: float = SWIPE_SPEC.default_positive_ratio
    reset_delay_frames: int = SWIPE_SPEC.default_reset_delay_frames
    sample_rate_hz: float = IMU_SAMPLE_RATE_HZ
    event_rate_window_seconds: float = 5.0


@dataclass(slots=True)
class SwipeRingState:
    ring_name: str
    enabled: bool = False
    samples: deque[list[float]] = field(default_factory=lambda: deque(maxlen=1024))
    frames_since_infer: int = 0
    prediction_history: deque[int] = field(default_factory=deque)
    committed_counts: Counter[int] = field(default_factory=Counter)
    inference_timestamps: deque[float] = field(default_factory=deque)
    event_timestamps: deque[float] = field(default_factory=deque)
    reset_after_frames_remaining: int = 0
    inference_in_flight: bool = False


@dataclass(frozen=True, slots=True)
class SwipeInferenceJob:
    address: str
    ring_name: str
    values: np.ndarray
    event_timestamp: float


@dataclass(frozen=True, slots=True)
class SwipeInferenceResult:
    address: str
    ring_name: str
    prediction: object
    elapsed_ms: float
    event_timestamp: float


@dataclass(frozen=True, slots=True)
class SwipeGestureEvent:
    address: str
    ring_name: str
    class_id: int
    event: str
    confidence: float
    probabilities: tuple[float, ...]
    timestamp: float
    total_count: int


class SwipeInferenceWorker:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        on_ready,
        on_result,
        on_error,
        backend: str = SWIPE_BACKEND,
    ):
        self.loop = loop
        self.on_ready = on_ready
        self.on_result = on_result
        self.on_error = on_error
        self.backend = backend
        self._jobs: queue.Queue[SwipeInferenceJob | None] = queue.Queue(maxsize=32)
        self._thread = threading.Thread(
            target=self._run,
            name="swipe-inference-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            if self._jobs.full():
                self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._jobs.put_nowait(None)
        self._thread.join(timeout=1.5)

    def submit(self, job: SwipeInferenceJob) -> None:
        if self._jobs.full():
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
        self._jobs.put_nowait(job)

    def _run(self) -> None:
        try:
            recognizer = build_swipe_recognizer(backend=self.backend)
        except Exception as exc:
            self.loop.call_soon_threadsafe(self.on_error, f"Swipe 模型加载失败: {exc}")
            return

        self.loop.call_soon_threadsafe(self.on_ready)

        while True:
            job = self._jobs.get()
            if job is None:
                return
            started = time.perf_counter()
            try:
                prediction = recognizer.predict(job.values)
            except Exception as exc:
                self.loop.call_soon_threadsafe(
                    self.on_error,
                    f"{job.ring_name} Swipe 推理失败: {exc}",
                )
                continue
            result = SwipeInferenceResult(
                address=job.address,
                ring_name=job.ring_name,
                prediction=prediction,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                event_timestamp=job.event_timestamp,
            )
            self.loop.call_soon_threadsafe(self.on_result, result)


class SwipeRuntime:
    def __init__(self, *, on_event, on_status=None, config: SwipeRuntimeConfig | None = None):
        self._loop = asyncio.get_event_loop()
        self._on_event = on_event
        self._on_status = on_status
        self.config = config or SwipeRuntimeConfig()
        self.states: dict[str, SwipeRingState] = {}
        self._worker: SwipeInferenceWorker | None = None
        self._ready = False
        self._failed = False

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._ready = False

    def set_enabled(self, address: str, ring_name: str, enabled: bool) -> None:
        state = self.state_for(address, ring_name)
        state.enabled = bool(enabled)
        if state.enabled:
            self._ensure_worker_started()
            self._notify_status(f"{ring_name} Swipe 识别已开启")
            return
        self.reset_device(address, keep_enabled=False)
        self._notify_status(f"{ring_name} Swipe 识别已关闭")

    def remove_device(self, address: str) -> None:
        self.states.pop(address, None)

    def reset_device(self, address: str, *, keep_enabled: bool = True) -> None:
        state = self.states.get(address)
        if state is None:
            return
        enabled = state.enabled if keep_enabled else False
        ring_name = state.ring_name
        self.states[address] = SwipeRingState(ring_name=ring_name, enabled=enabled)

    def recognizer_fps(self, address: str, now: float | None = None) -> float:
        state = self.states.get(address)
        if state is None:
            return 0.0
        current = time.perf_counter() if now is None else float(now)
        self._trim_rate_timestamps(state.inference_timestamps, current)
        return len(state.inference_timestamps) / self.config.event_rate_window_seconds

    def handle_imu(
        self,
        *,
        address: str,
        ring_name: str,
        sample: list[float],
        event_timestamp: float,
    ) -> None:
        state = self.states.get(address)
        if state is None or not state.enabled:
            return
        if state.ring_name != ring_name:
            state.ring_name = ring_name

        state.samples.append(sample)
        state.frames_since_infer += 1
        self._advance_pending_reset(state, sample)
        if not self._ready or self._failed or state.reset_after_frames_remaining > 0:
            return
        if state.inference_in_flight:
            return
        if len(state.samples) < self.config.window_size:
            return
        if state.frames_since_infer < self.config.step_frames:
            return

        state.frames_since_infer = 0
        state.inference_in_flight = True
        values = np.asarray(list(state.samples)[-self.config.window_size :], dtype=np.float32)
        self._worker.submit(
            SwipeInferenceJob(
                address=address,
                ring_name=state.ring_name,
                values=values,
                event_timestamp=float(event_timestamp),
            )
        )

    def state_for(self, address: str, ring_name: str) -> SwipeRingState:
        state = self.states.get(address)
        if state is None:
            state = SwipeRingState(ring_name=ring_name)
            self.states[address] = state
            return state
        if state.ring_name != ring_name:
            state.ring_name = ring_name
        return state

    def _ensure_worker_started(self) -> None:
        if self._worker is not None or self._failed:
            return
        self._worker = SwipeInferenceWorker(
            loop=self._loop,
            on_ready=self._handle_worker_ready,
            on_result=self._handle_inference_result,
            on_error=self._handle_worker_error,
        )
        self._worker.start()

    def _handle_worker_ready(self) -> None:
        self._ready = True
        self._failed = False
        self._notify_status("Swipe 模型已就绪")

    def _handle_worker_error(self, message: str) -> None:
        if message.startswith("Swipe 模型加载失败"):
            self._failed = True
            self._ready = False
        self._notify_status(message)
        for state in self.states.values():
            state.inference_in_flight = False

    def _handle_inference_result(self, result: SwipeInferenceResult) -> None:
        state = self.states.get(result.address)
        if state is None:
            return
        state.inference_in_flight = False
        if not state.enabled:
            return
        current = time.perf_counter()
        state.inference_timestamps.append(current)
        self._trim_rate_timestamps(state.inference_timestamps, current)

        class_id = int(getattr(result.prediction, "class_id", 0))
        queue_length = self._result_queue_length()
        threshold = self._min_trigger_count(queue_length)
        history = state.prediction_history
        history.append(class_id)
        while len(history) > queue_length:
            history.popleft()

        if len(history) < queue_length:
            return

        for candidate in SWIPE_SPEC.class_ids[1:]:
            if history.count(candidate) < threshold:
                continue
            state.committed_counts[candidate] += 1
            state.event_timestamps.append(current)
            self._trim_rate_timestamps(state.event_timestamps, current)
            state.prediction_history.clear()
            state.frames_since_infer = 0
            state.reset_after_frames_remaining = self.config.reset_delay_frames
            label = SWIPE_SPEC.class_labels.get(candidate, str(candidate))
            event = SwipeGestureEvent(
                address=result.address,
                ring_name=result.ring_name,
                class_id=candidate,
                event=label,
                confidence=float(getattr(result.prediction, "confidence", 0.0)),
                probabilities=tuple(
                    float(value) for value in getattr(result.prediction, "probabilities", ())
                ),
                timestamp=float(result.event_timestamp),
                total_count=int(state.committed_counts[candidate]),
            )
            self._on_event(event)
            self._notify_status(
                f"{result.ring_name} 检测到 {label}（{result.elapsed_ms:.1f} ms）"
            )
            return

    def _advance_pending_reset(self, state: SwipeRingState, current_sample: list[float]) -> None:
        if state.reset_after_frames_remaining <= 0:
            return
        state.reset_after_frames_remaining -= 1
        if state.reset_after_frames_remaining > 0:
            return
        state.samples.clear()
        for _ in range(self.config.window_size // 2 - 1):
            state.samples.append(current_sample)
        state.frames_since_infer = 0
        state.prediction_history.clear()

    def _result_queue_length(self) -> int:
        return max(
            1,
            int(
                round(
                    self.config.stable_window_seconds
                    * self.config.sample_rate_hz
                    / max(self.config.step_frames, 1)
                )
            ),
        )

    def _min_trigger_count(self, queue_length: int) -> int:
        return max(1, int(math.ceil(self.config.positive_ratio * queue_length)))

    def _trim_rate_timestamps(self, timestamps: deque[float], now: float) -> None:
        threshold = now - self.config.event_rate_window_seconds
        while timestamps and timestamps[0] < threshold:
            timestamps.popleft()

    def _notify_status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)
