from dataclasses import dataclass
from typing import Callable

from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from apps.registry import AppLaunchSpec


@dataclass
class MainToolbar:
    panel: QFrame
    btn_scan: QPushButton
    port_input: QSpinBox
    chk_imu_stream: QCheckBox
    chk_audio_stream: QCheckBox
    chk_auto_reconnect: QCheckBox
    lbl_msg: QLabel


@dataclass
class DeviceListView:
    scroll: QScrollArea
    content: QWidget
    card_layout: QHBoxLayout


@dataclass
class AppLaunchPanel:
    panel: QFrame


def create_main_toolbar(
    *,
    initial_port: int,
    on_scan: Callable[[], None],
    on_port_changed: Callable[[int], None],
    on_stream_mode_changed: Callable[[int], None],
    on_auto_reconnect_changed: Callable[[int], None],
) -> MainToolbar:
    # 顶部控制区只负责输入与状态展示，不承载业务逻辑。
    panel = QFrame()
    panel.setProperty("class", "panel")

    root = QVBoxLayout()
    root.setContentsMargins(16, 14, 16, 14)
    root.setSpacing(10)

    row_primary = QHBoxLayout()
    row_primary.setSpacing(10)
    row_secondary = QHBoxLayout()
    row_secondary.setSpacing(10)

    btn_scan = QPushButton("开始扫描")
    btn_scan.setProperty("class", "btn-primary")
    btn_scan.clicked.connect(on_scan)

    port_input = QSpinBox()
    port_input.setRange(1024, 65535)
    port_input.setValue(initial_port)
    port_input.setFixedWidth(110)
    port_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    port_input.valueChanged.connect(on_port_changed)

    chk_imu_stream = QCheckBox("IMU")
    chk_imu_stream.setChecked(True)
    chk_imu_stream.stateChanged.connect(on_stream_mode_changed)

    chk_audio_stream = QCheckBox("音频")
    chk_audio_stream.setChecked(False)
    chk_audio_stream.stateChanged.connect(on_stream_mode_changed)

    chk_auto_reconnect = QCheckBox("自动重连")
    chk_auto_reconnect.setChecked(True)
    chk_auto_reconnect.stateChanged.connect(on_auto_reconnect_changed)

    lbl_msg = QLabel("就绪")
    lbl_msg.setObjectName("statusTag")

    row_primary.addWidget(btn_scan)
    row_primary.addWidget(QLabel("Port"))
    row_primary.addWidget(port_input)
    row_primary.addWidget(lbl_msg, 1)
    row_primary.addStretch()

    row_secondary.addWidget(QLabel("转发流"))
    row_secondary.addWidget(chk_imu_stream)
    row_secondary.addWidget(chk_audio_stream)
    row_secondary.addSpacing(12)
    row_secondary.addWidget(chk_auto_reconnect)
    row_secondary.addStretch()

    root.addLayout(row_primary)
    root.addLayout(row_secondary)
    panel.setLayout(root)

    return MainToolbar(
        panel=panel,
        btn_scan=btn_scan,
        port_input=port_input,
        chk_imu_stream=chk_imu_stream,
        chk_audio_stream=chk_audio_stream,
        chk_auto_reconnect=chk_auto_reconnect,
        lbl_msg=lbl_msg,
    )


def create_device_list_view() -> DeviceListView:
    # 设备列表保持纯容器职责，具体卡片交给 DeviceCard 渲染。
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)

    content = QWidget()
    card_layout = QHBoxLayout(content)
    card_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    card_layout.setSpacing(15)

    scroll.setWidget(content)
    return DeviceListView(scroll=scroll, content=content, card_layout=card_layout)


def create_app_launch_panel(
    *,
    app_specs: list[AppLaunchSpec],
    on_launch: Callable[[AppLaunchSpec], None],
) -> AppLaunchPanel:
    # 应用入口统一从 registry 渲染，避免主窗口硬编码按钮。
    panel = QFrame()
    panel.setProperty("class", "panel")

    layout = QGridLayout()
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setHorizontalSpacing(10)
    layout.setVerticalSpacing(10)

    title = QLabel("应用")
    title.setProperty("class", "title")
    layout.addWidget(title, 0, 0)

    for column, spec in enumerate(app_specs, start=1):
        button = QPushButton(spec.title)
        button.setProperty("class", spec.button_class)
        button.clicked.connect(
            lambda _checked=False, app_spec=spec: on_launch(app_spec)
        )
        layout.addWidget(button, 0, column)

    layout.setColumnStretch(len(app_specs) + 1, 1)
    panel.setLayout(layout)
    return AppLaunchPanel(panel=panel)
