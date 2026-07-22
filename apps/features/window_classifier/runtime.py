from __future__ import annotations
"""Shared runtime primitives for realtime window-based gesture classifiers."""

import queue
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


@dataclass(frozen=True)
class ClassifierSpec:
    checkpoint_dir: Path
    class_ids: tuple[int, ...]
    class_labels: dict[int, str]
    class_colors: dict[int, QColor]
    model_label: str
    default_window_size: int
    default_step_frames: int
    default_stable_window_seconds: float
    default_positive_ratio: float
    default_reset_delay_frames: int
    allow_compact_probabilities: bool = False
    enable_result_fps: bool = False


@dataclass(frozen=True)
class WindowClassifierPreset:
    """Complete descriptor for one classifier family.

    Standalone apps and the unified app both consume the same preset so model
    loading, labels, event names, and keyboard mappings stay in one place.
    """

    name: str
    window_title: str
    panel_title: str
    description_text: str
    spec: ClassifierSpec
    event_map: dict[int, str]
    recognizer_factory: Callable[..., object]
    keyboard_action_map: dict[int, str] = field(default_factory=dict)
    keyboard_tooltip: str = ""
    result_title_text: str = "实时概率柱状图"

    def build_recognizer(self, *, backend: str, model_path: Path | None = None) -> object:
        return self.recognizer_factory(backend=backend, model_path=model_path)


class RateTracker:
    def __init__(self, window_seconds: float = 1.0):
        self.window_seconds = max(0.1, float(window_seconds))
        self._timestamps: deque[float] = deque()

    def reset(self) -> None:
        self._timestamps.clear()

    def mark(self, timestamp: float | None = None) -> None:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._timestamps.append(now)
        self._trim(now)

    def fps(self) -> float:
        now = time.perf_counter()
        self._trim(now)
        return len(self._timestamps) / self.window_seconds

    def _trim(self, now: float) -> None:
        threshold = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < threshold:
            self._timestamps.popleft()


@dataclass
class RingBufferState:
    ring_name: str
    samples: deque[list[float]] = field(default_factory=lambda: deque(maxlen=1024))
    frames_since_infer: int = 0
    prediction_history: deque[int] = field(default_factory=deque)
    committed_counts: Counter[int] = field(default_factory=Counter)
    reset_after_frames_remaining: int = 0
    rate_tracker: RateTracker = field(default_factory=RateTracker)


@dataclass(frozen=True)
class InferenceJob:
    address: str
    ring_name: str
    values: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    address: str
    ring_name: str
    prediction: Any
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


class ProbabilityChart(QWidget):
    """Reusable probability view shared by swipe/pinch style apps."""

    def __init__(
        self,
        *,
        class_ids: tuple[int, ...],
        class_labels: dict[int, str],
        class_colors: dict[int, QColor],
        allow_compact_probabilities: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.class_ids = tuple(class_ids)
        self.class_labels = dict(class_labels)
        self.class_colors = dict(class_colors)
        self.allow_compact_probabilities = bool(allow_compact_probabilities)
        self._probabilities = [0.0 for _ in self.class_ids]
        self.setMinimumHeight(280)

    def clear(self) -> None:
        self._probabilities = [0.0 for _ in self.class_ids]
        self.update()

    def set_probabilities(
        self,
        probabilities: list[float] | tuple[float, ...] | np.ndarray,
    ) -> None:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if self.allow_compact_probabilities and values.size == len(self.class_ids) - 1:
            values = np.concatenate((np.zeros(1, dtype=np.float32), values))
        elif values.size != len(self.class_ids):
            raise ValueError(
                f"概率长度必须为 {len(self.class_ids)}"
                + (
                    f" 或 {len(self.class_ids) - 1}"
                    if self.allow_compact_probabilities
                    else ""
                )
                + f"，收到: {values.size}"
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

        bar_count = len(self.class_ids)
        gap = 14
        if bar_count > 1:
            max_gap = max(6, int(plot_rect.width() * 0.03))
            gap = min(gap, max_gap, max((plot_rect.width() - 18 * bar_count) // (bar_count - 1), 6))
        available_width = plot_rect.width() - gap * (bar_count - 1)
        bar_width = max(int(available_width / max(bar_count, 1)), 18)
        for index, class_id in enumerate(self.class_ids):
            probability = self._probabilities[index]
            height = int(probability * plot_rect.height())
            x = int(plot_rect.left() + index * (bar_width + gap))
            y = int(plot_rect.bottom() - height)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#E8EEF5"))
            painter.drawRoundedRect(x, plot_rect.top(), bar_width, plot_rect.height(), 8, 8)

            painter.setBrush(self.class_colors[class_id])
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
                f"{class_id}\n{self.class_labels[class_id]}",
            )

        painter.end()


class ProbabilityTimelineChart(QWidget):
    """Rolling time-series line chart showing per-class probability history."""

    def __init__(
        self,
        *,
        class_ids: tuple[int, ...],
        class_labels: dict[int, str],
        class_colors: dict[int, QColor],
        history_seconds: float = 10.0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.class_ids = tuple(class_ids)
        self.class_labels = dict(class_labels)
        self.class_colors = dict(class_colors)
        self.history_seconds = max(1.0, float(history_seconds))
        self._history: deque[tuple[float, tuple[float, ...]]] = deque()
        self._start_time: float | None = None
        self.setMinimumHeight(200)

    def clear(self) -> None:
        self._history.clear()
        self._start_time = None
        self.update()

    def set_probabilities(
        self,
        probabilities: list[float] | tuple[float, ...] | np.ndarray,
    ) -> None:
        values = tuple(float(v) for v in np.asarray(probabilities, dtype=np.float32).flat)
        if len(values) != len(self.class_ids):
            return
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now
        elapsed = now - self._start_time
        self._history.append((elapsed, values))
        cutoff = elapsed - self.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(8, 4, -8, -4)
        margin_left = 44
        margin_right = 12
        margin_top = 10
        margin_bottom = 28

        plot_rect = rect.adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
        painter.fillRect(rect, QColor("#FBFCFE"))
        painter.fillRect(plot_rect, QColor("#F7F9FC"))

        # ── horizontal grid ──
        painter.setPen(QPen(QColor("#DCE5EE"), 1, Qt.PenStyle.DotLine))
        for level in range(5):
            ratio = level / 4.0
            y = int(plot_rect.bottom() - ratio * plot_rect.height())
            painter.drawLine(int(plot_rect.left()), y, int(plot_rect.right()), y)

        # ── Y-axis labels ──
        painter.setPen(QColor("#7F90A4"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for level in range(5):
            ratio = level / 4.0
            y = int(plot_rect.bottom() - ratio * plot_rect.height())
            painter.drawText(
                int(rect.left()), y - 7, margin_left - 6, 14,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{int(ratio * 100)}%",
            )

        # ── X-axis labels ──
        if self._history:
            t_min = self._history[0][0]
            t_max = self._history[-1][0]
            duration = max(t_max - t_min, 0.5)
            if duration <= 2:
                tick_interval = 0.5
            elif duration <= 5:
                tick_interval = 1.0
            elif duration <= 15:
                tick_interval = 2.0
            else:
                tick_interval = 5.0
            tick = int(t_min / tick_interval) * tick_interval
            while tick <= t_max + tick_interval * 0.01:
                x = int(plot_rect.left() + (tick - t_min) / max(duration, 0.001) * plot_rect.width())
                if plot_rect.left() <= x <= plot_rect.right():
                    painter.setPen(QPen(QColor("#DCE5EE"), 1, Qt.PenStyle.DotLine))
                    painter.drawLine(x, int(plot_rect.top()), x, int(plot_rect.bottom()))
                    painter.setPen(QColor("#7F90A4"))
                    painter.drawText(x - 30, int(plot_rect.bottom()) + 2, 60, 18,
                                     Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                                     f"{tick:.1f}s" if tick_interval < 1.0 else f"{int(tick)}s")
                tick += tick_interval

        # ── plot border ──
        painter.setPen(QPen(QColor("#D0D8E2"), 1))
        painter.drawRect(plot_rect)

        # ── draw lines ──
        if len(self._history) >= 2:
            t_min = self._history[0][0]
            t_max = self._history[-1][0]
            time_span = max(t_max - t_min, 0.001)
            painter.save()
            painter.setClipRect(plot_rect)
            for idx, class_id in reversed(list(enumerate(self.class_ids))):
                color = self.class_colors.get(class_id, QColor("#888888"))
                lw = 2.0 if class_id != 0 else 1.2
                alpha = 220 if class_id != 0 else 140
                c = QColor(color)
                c.setAlpha(alpha)
                painter.setPen(QPen(c, lw))
                points = []
                for elapsed, probs in self._history:
                    x = int(plot_rect.left() + (elapsed - t_min) / time_span * plot_rect.width())
                    y = int(plot_rect.bottom() - probs[idx] * plot_rect.height())
                    points.append((x, y))
                for i in range(len(points) - 1):
                    painter.drawLine(points[i][0], points[i][1],
                                     points[i + 1][0], points[i + 1][1])
            painter.restore()

        # ── legend ──
        legend_y = int(plot_rect.top()) + 6
        for idx, class_id in reversed(list(enumerate(self.class_ids))):
            color = self.class_colors.get(class_id, QColor("#888888"))
            ly = legend_y + idx * 16
            if ly > plot_rect.bottom() - 10:
                break
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            sw = int(plot_rect.right()) - len(self.class_ids) * 72 + idx * 72
            painter.drawRoundedRect(sw, ly, 10, 10, 2, 2)
            painter.setPen(QColor("#44576A"))
            painter.drawText(sw + 14, ly - 1, 56, 12,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f"{class_id}:{self.class_labels.get(class_id, str(class_id))}")

        # ── title ──
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor("#7F90A4"))
        painter.drawText(int(plot_rect.left()), int(rect.top()),
                         plot_rect.width(), 16,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                         "Probability Timeline")
        painter.end()


class WindowClassifierWorker(QThread):
    """Runs model inference off the UI thread; app windows only submit windows and render results."""

    ready = pyqtSignal()
    result_ready = pyqtSignal(object)
    job_failed = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(
        self,
        recognizer_factory: Callable[[], object],
    ):
        super().__init__()
        self.recognizer_factory = recognizer_factory
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
            recognizer = self.recognizer_factory()
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
                self.job_failed.emit(
                    InferenceFailure(ring_name=job.ring_name, error=str(exc))
                )
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
