from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor

from .runtime import ClassifierSpec, WindowClassifierPreset

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SWIPE_CLASS_IDS = (0, 1, 2, 3, 4, 5)
SWIPE_CLASS_LABELS = {
    0: "空",
    1: "swipe-up",
    2: "swipe-down",
    3: "swipe-left",
    4: "swipe-right",
    5: "tap",
}
SWIPE_CLASS_COLORS = {
    0: QColor("#B0B8C1"),
    1: QColor("#5B8FF9"),
    2: QColor("#61DDAA"),
    3: QColor("#F6BD16"),
    4: QColor("#E8684A"),
    5: QColor("#6DC8EC"),
}
SWIPE_EVENT_MAP = {
    1: "swipe-up",
    2: "swipe-down",
    3: "swipe-left",
    4: "swipe-right",
    5: "tap",
}
SWIPE_SPEC = ClassifierSpec(
    checkpoint_dir=PROJECT_ROOT / "checkpoint" / "swipe" / "cls",
    class_ids=SWIPE_CLASS_IDS,
    class_labels=SWIPE_CLASS_LABELS,
    class_colors=SWIPE_CLASS_COLORS,
    model_label="checkpoint/swipe/cls",
    default_window_size=60,
    default_step_frames=5,
    default_stable_window_seconds=0.10,
    default_positive_ratio=1.0,
    default_reset_delay_frames=10,
    allow_compact_probabilities=True,
)


def build_swipe_recognizer(*, backend: str = "openvino", model_path: Path | None = None):
    from models.swipe_cnn_recognizer import SwipeCNNRecognizer

    source = model_path or SWIPE_SPEC.checkpoint_dir
    return SwipeCNNRecognizer(
        checkpoint_dir=source,
        backend=backend,
    )


SWIPE_PRESET = WindowClassifierPreset(
    name="swipe",
    window_title="Swipe 实时检测",
    panel_title="Swipe 实时检测",
    description_text=(
        "每隔若干帧取最近一个固定窗口做一次分类。分类结果进入短队列，"
        "当某个非空类别在稳定窗口内占比超过阈值时，记为一次触发；"
        "触发后暂停若干帧，再把 IMU 缓冲收缩到最新一帧。"
    ),
    spec=SWIPE_SPEC,
    event_map=SWIPE_EVENT_MAP,
    recognizer_factory=build_swipe_recognizer,
    keyboard_action_map={
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "space",
    },
    keyboard_tooltip="触发手势时发送方向键或空格",
)
