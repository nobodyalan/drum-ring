from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor

from .runtime import ClassifierSpec, WindowClassifierPreset

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PINCH_CLASS_IDS = (0, 1, 2)
PINCH_CLASS_LABELS = {
    0: "空",
    1: "pinch-down",
    2: "pinch-up",
}
PINCH_CLASS_COLORS = {
    0: QColor("#B0B8C1"),
    1: QColor("#5B8FF9"),
    2: QColor("#61DDAA"),
}
PINCH_EVENT_MAP = {
    1: "pinch-down",
    2: "pinch-up",
}
PINCH_SPEC = ClassifierSpec(
    checkpoint_dir=PROJECT_ROOT / "checkpoint" / "swipe" / "cls-pinch",
    class_ids=PINCH_CLASS_IDS,
    class_labels=PINCH_CLASS_LABELS,
    class_colors=PINCH_CLASS_COLORS,
    model_label="checkpoint/swipe/cls-pinch",
    default_window_size=60,
    default_step_frames=2,
    default_stable_window_seconds=0.05,
    default_positive_ratio=1.0,
    default_reset_delay_frames=2,
    enable_result_fps=True,
)


def build_pinch_recognizer(*, backend: str = "openvino", model_path: Path | None = None):
    from models.pinch_cnn_recognizer import PinchCNNRecognizer

    source = model_path or PINCH_SPEC.checkpoint_dir
    return PinchCNNRecognizer(
        checkpoint_dir=source,
        backend=backend,
    )


PINCH_PRESET = WindowClassifierPreset(
    name="pinch",
    window_title="Pinch 实时检测",
    panel_title="Pinch 实时检测",
    description_text=(
        "每隔若干帧取最近一个固定窗口做一次分类。分类结果进入短队列，"
        "当某个非空类别在稳定窗口内占比超过阈值时，记为一次触发；"
        "触发后暂停若干帧，再把 IMU 缓冲收缩到最新一帧。"
    ),
    spec=PINCH_SPEC,
    event_map=PINCH_EVENT_MAP,
    recognizer_factory=build_pinch_recognizer,
)
