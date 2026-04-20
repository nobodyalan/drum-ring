from __future__ import annotations
"""Composable UI blocks for classifier-style desktop apps."""

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


@dataclass(slots=True)
class WindowClassifierControlPanel:
    panel: QFrame
    layout: QVBoxLayout
    settings_layout: QGridLayout
    action_row: QHBoxLayout


@dataclass(slots=True)
class WindowClassifierResultPanel:
    panel: QFrame
    result_label: QLabel
    meta_label: QLabel
    count_label: QLabel


def create_window_classifier_control_panel(
    *,
    title_text: str,
    description_text: str,
    status_widgets: list[QWidget],
) -> WindowClassifierControlPanel:
    panel = QFrame()
    panel.setProperty("class", "panel")

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(10)

    title = QLabel(title_text)
    title.setProperty("class", "title")

    description = QLabel(description_text)
    description.setProperty("class", "sub-text")
    description.setWordWrap(True)

    layout.addWidget(title)
    layout.addWidget(description)
    for widget in status_widgets:
        layout.addWidget(widget)

    settings_layout = QGridLayout()
    settings_layout.setHorizontalSpacing(10)
    settings_layout.setVerticalSpacing(10)
    layout.addLayout(settings_layout)

    action_row = QHBoxLayout()
    layout.addLayout(action_row)

    return WindowClassifierControlPanel(
        panel=panel,
        layout=layout,
        settings_layout=settings_layout,
        action_row=action_row,
    )


def create_window_classifier_result_panel(
    *,
    chart: QWidget,
    title_text: str = "实时概率柱状图",
) -> WindowClassifierResultPanel:
    panel = QFrame()
    panel.setProperty("class", "panel")

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(8)

    title = QLabel(title_text)
    title.setProperty("class", "title")

    result_label = QLabel("--")
    result_label.setProperty("class", "title")
    result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    meta_label = QLabel("最近结果：无")
    meta_label.setWordWrap(True)

    count_label = QLabel()
    count_label.setWordWrap(True)

    layout.addWidget(title)
    layout.addWidget(result_label)
    layout.addWidget(meta_label)
    layout.addWidget(count_label)
    layout.addWidget(chart, 1)

    return WindowClassifierResultPanel(
        panel=panel,
        result_label=result_label,
        meta_label=meta_label,
        count_label=count_label,
    )
