from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from .device_runtime import RingDevice


class DeviceCard(QFrame):
    sig_action = pyqtSignal(str, str)
    sig_swipe_toggled = pyqtSignal(str, bool)

    def __init__(self, device: RingDevice):
        super().__init__()
        self.device = device
        self._build_ui()
        self.refresh_view()

    def _build_ui(self) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(360, 300)
        layout = QVBoxLayout(self)

        title_row = QHBoxLayout()
        self.lbl_name = QLabel(self.device.name)
        self.lbl_name.setProperty("class", "title")
        title_row.addWidget(self.lbl_name)
        title_row.addStretch()
        layout.addLayout(title_row)

        self.lbl_addr = QLabel(self.device.address)
        self.lbl_addr.setProperty("class", "sub-text")
        layout.addWidget(self.lbl_addr)

        self.lbl_status = QLabel("● 未连接")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

        button_layout = QGridLayout()
        self.btn_connect = QPushButton("连接")
        self.btn_connect.clicked.connect(
            lambda: self.sig_action.emit(self.device.address, "connect")
        )

        self.btn_sync = QPushButton("同步")
        self.btn_sync.setProperty("class", "btn-gray")
        self.btn_sync.clicked.connect(
            lambda: self.sig_action.emit(self.device.address, "sync")
        )

        button_layout.addWidget(self.btn_connect, 0, 0, 1, 2)
        button_layout.addWidget(self.btn_sync, 1, 0, 1, 2)
        layout.addLayout(button_layout)

        stat_row = QHBoxLayout()
        stat_row.setSpacing(8)
        self.lbl_battery = QLabel("电量：--%")
        self.lbl_battery.setProperty("class", "stat-tag")
        self.lbl_imu_fps = QLabel("IMU：--Hz")
        self.lbl_imu_fps.setProperty("class", "stat-tag")
        self.lbl_audio_fps = QLabel("音频：--Hz")
        self.lbl_audio_fps.setProperty("class", "stat-tag")
        stat_row.addWidget(self.lbl_battery)
        stat_row.addWidget(self.lbl_imu_fps)
        stat_row.addWidget(self.lbl_audio_fps)
        layout.addLayout(stat_row)

        gesture_row = QHBoxLayout()
        gesture_row.setSpacing(8)
        self.lbl_gesture_fps = QLabel("识别：0.0Hz")
        self.lbl_gesture_fps.setProperty("class", "stat-tag")
        gesture_row.addWidget(self.lbl_gesture_fps)
        gesture_row.addStretch()
        layout.addLayout(gesture_row)

        self.chk_swipe = QCheckBox("启用 Swipe 识别")
        self.chk_swipe.setChecked(self.device.swipe_enabled)
        self.chk_swipe.toggled.connect(
            lambda enabled: self.sig_swipe_toggled.emit(self.device.address, enabled)
        )
        layout.addWidget(self.chk_swipe)

    def update_progress(self, current: int, total: int) -> None:
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        if current >= total:
            self.lbl_status.setText("同步完成")
            self.progress_bar.setVisible(False)

    def refresh_view(self) -> None:
        self.lbl_name.setText(self.device.name)
        ring = self.device.ring_logic

        if self.device.is_reconnecting:
            self.setProperty("class", "device-card")
            self.lbl_status.setText("● 正在重连")
            self.lbl_status.setStyleSheet("color: #dd6b20; font-weight: bold;")
            self.btn_connect.setText("取消重连")
            self.btn_connect.setProperty("class", "btn-gray")
            self.btn_sync.setEnabled(False)
            self.btn_sync.setProperty("class", "btn-gray")
        elif self.device.is_connected:
            self.setProperty("class", "device-card-connected")
            self.lbl_status.setText(
                f"● 已连接（{'已校准' if self.device.is_calibrated else '未校准'}）"
            )
            self.lbl_status.setStyleSheet("color: #e53e3e; font-weight: bold;")
            self.btn_connect.setText("断开")
            self.btn_connect.setProperty("class", "btn-danger")
            self.btn_sync.setEnabled(True)
            self.btn_sync.setProperty("class", "btn-primary")
        else:
            self.setProperty("class", "device-card")
            self.lbl_status.setText("● 未连接")
            self.lbl_status.setStyleSheet("color: #a0aec0;")
            self.btn_connect.setText("连接")
            self.btn_connect.setProperty("class", "btn-primary")
            self.btn_sync.setEnabled(False)
            self.btn_sync.setProperty("class", "btn-gray")

        battery_level = getattr(ring, "battery_level", -1) if ring is not None else -1
        imu_fps = float(getattr(ring, "imu_fps", 0.0)) if ring is not None else 0.0
        audio_fps = float(getattr(ring, "audio_fps", 0.0)) if ring is not None else 0.0
        recognizer_fps = float(getattr(self.device, "recognizer_fps", 0.0))
        self.lbl_battery.setText(f"电量：{battery_level if battery_level > 0 else '--'}%")
        self.lbl_imu_fps.setText(f"IMU：{imu_fps:.1f}Hz")
        self.lbl_audio_fps.setText(f"音频：{audio_fps:.1f}Hz")
        self.lbl_gesture_fps.setText(f"识别：{recognizer_fps:.1f}Hz")
        self.chk_swipe.blockSignals(True)
        self.chk_swipe.setChecked(self.device.swipe_enabled)
        self.chk_swipe.blockSignals(False)

        self.style().unpolish(self)
        self.style().polish(self)
        self.style().polish(self.btn_sync)
