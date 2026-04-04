import sys
import os
import subprocess
from pathlib import Path

import numpy as np
from tqdm import trange

os.environ["QT_API"] = "pyqt6"
import json
import asyncio
import time
from typing import Dict, List, Any

# ----------------------------------------------------------------------------
# 引入 qasync 核心组件
# ----------------------------------------------------------------------------
from qasync import QEventLoop, asyncSlot

from bleak import BleakScanner
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QFrame,
    QScrollArea,
    QGridLayout,
    QProgressBar,
    QSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from apps.common.ui_style import MAIN_STYLE_SHEET
from core.ble_ring_v2 import BLERing
from core.stream_hub import ImuStreamHub

# ============================================================================
# 1. 样式表
# ============================================================================
STYLE_SHEET = MAIN_STYLE_SHEET

# ============================================================================
# 2. 数据模型 (Model)
# ============================================================================


class RingDevice:
    """
    单个设备的逻辑实体。
    """

    def __init__(self, address: str, name: str):
        self.address = address
        self.name = name
        self.is_connected = False
        self.is_reconnecting = False
        self.is_recording = False
        self.is_calibrated = False

        # 逻辑成员
        self.ring_logic: BLERing | None = None
        # 布局成员
        self.ui_layout: Any = None

    def update_info(self, name: str):
        self.name = name


# ============================================================================
# 3. 业务服务层 (Service) - 修改为纯 Async
# ============================================================================


class DeviceService:
    """
    处理扫描、连接、IO等耗时操作。
    改为原生 async 方法，移除 asyncio.run()
    """

    @staticmethod
    async def scan_ble() -> List[dict]:
        """执行蓝牙扫描"""
        # BleakScanner.discover 本身就是 awaitable
        devices = await BleakScanner.discover(timeout=3.0)
        results = []
        for d in devices:
            name = d.name or "Unknown"
            if name.startswith("BCL"):
                results.append({"name": name, "address": d.address})

        return results

    @staticmethod
    async def connect_device(device: RingDevice):
        """异步连接"""
        await device.ring_logic.connect()
        if device.ring_logic.connected:
            return True
        else:
            raise Exception("连接失败")

    @staticmethod
    async def disconnect_device(device: RingDevice):
        """异步断开"""
        await device.ring_logic.disconnect()
        return True

    @staticmethod
    async def start_recording(device: RingDevice):
        """异步开始录制"""
        await device.ring_logic.start_recording()
        return True

    @staticmethod
    async def stop_recording(device: RingDevice):
        """异步停止录制"""
        await device.ring_logic.stop_recording()
        return True

    @staticmethod
    async def open_imu(device: RingDevice):
        await device.ring_logic.open_imu()
        return True

    @staticmethod
    async def close_imu(device: RingDevice):
        await device.ring_logic.close_imu()
        return True

    @staticmethod
    async def open_audio(device: RingDevice):
        await device.ring_logic.open_audio()
        return True

    @staticmethod
    async def close_audio(device: RingDevice):
        await device.ring_logic.close_audio()
        return True

    @staticmethod
    async def sync_time(device: RingDevice, progress_callback=None):
        """异步同步数据"""
        if not device.ring_logic.connected:
            raise Exception("设备未连接")

        ring = device.ring_logic
        ring.reset_time_calib_data()
        ring.clear_time_calibration()

        calib_packet_num = 300
        calib_packet_interval = 0.1
        max_calib_delta = 0.016
        cnt = 0
        pc_timestamps = []
        ring_timestamps = []
        pc_delta_time = []

        for _ in trange(calib_packet_num):
            if progress_callback:
                progress_callback(cnt, calib_packet_num)
            await device.ring_logic.calib_time(index=cnt)
            await asyncio.sleep(calib_packet_interval)
            cnt += 1
        await asyncio.sleep(1)

        ring.start_calib_timestamps = np.sort(ring.start_calib_timestamps)
        ring.end_calib_timestamps = np.sort(ring.end_calib_timestamps)
        ring.ring_timestamps = np.sort(ring.ring_timestamps)
        ring.end_indices = np.sort(np.array(ring.end_indices))
        ring.start_calib_timestamps = ring.start_calib_timestamps[ring.end_indices]
        assert len(ring.start_calib_timestamps) == len(ring.ring_timestamps)
        assert len(ring.end_calib_timestamps) == len(ring.ring_timestamps)

        pc_timestamps = (ring.start_calib_timestamps + ring.end_calib_timestamps) / 2
        ring_timestamps = ring.ring_timestamps
        pc_delta_time = ring.end_calib_timestamps - ring.start_calib_timestamps
        timestamps = list(zip(pc_timestamps, ring_timestamps, pc_delta_time))
        # sort with pc_delta
        timestamps = sorted(timestamps, key=lambda x: x[2])
        timestamps = [x for x in timestamps if x[2] < max_calib_delta]
        print(
            f"used calib packet num: {len(timestamps)}, percentage: {len(timestamps) / calib_packet_num}"
        )

        pc_timestamps, ring_timestamps, pc_delta_time = zip(*timestamps)

        # linear regression
        ring_timestamps = np.array(ring_timestamps)
        pc_timestamps = np.array(pc_timestamps)
        pc_delta_time = np.array(pc_delta_time)
        X_b = np.c_[np.ones((len(ring_timestamps), 1)), ring_timestamps]
        intercept, slope = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(pc_timestamps)
        print("intercept:", intercept, "slope:", slope)

        # reset ring params
        ring.reset_time_calib_data()

        if progress_callback:
            progress_callback(calib_packet_num, calib_packet_num)  # end

        if len(timestamps) < 20:
            raise Exception("校准数据不足，可能同步失败，最好重新同步")

        # save
        os.makedirs("data", exist_ok=True)
        ring.set_time_calibration(intercept, slope)
        with open(f"data/calib_time_{int(time.time())}.json", "w") as f:
            json.dump(
                {
                    "intercept": intercept,
                    "slope": slope,
                    "pc_timestamps": pc_timestamps.tolist(),
                    "ring_timestamps": ring_timestamps.tolist(),
                    "pc_delta_time": pc_delta_time.tolist(),
                    "ring_mac": ring.address,
                    "saved_perf_counter": time.perf_counter(),
                    "saved_epoch": time.time(),
                },
                f,
            )

    @staticmethod
    def find_latest_calib_record(ring_mac: str) -> dict | None:
        data_dir = Path("data")
        if not data_dir.exists():
            return None
        files = sorted(
            data_dir.glob("calib_time_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for file in files:
            try:
                with open(file, "r") as f:
                    payload = json.load(f)
                if str(payload.get("ring_mac", "")).lower() == ring_mac.lower():
                    payload["__file"] = str(file)
                    return payload
            except Exception:
                continue
        return None

    @staticmethod
    async def load_calib_if_consistent(
        device: RingDevice, tolerance_ms: float = 8.0
    ) -> tuple[bool, str]:
        if device.ring_logic is None:
            return False, "无设备逻辑"
        ring = device.ring_logic
        record = DeviceService.find_latest_calib_record(ring.address)
        if record is None:
            ring.clear_time_calibration()
            return False, "未找到历史校准记录"

        try:
            intercept = float(record["intercept"])
            slope = float(record["slope"])
        except Exception:
            ring.clear_time_calibration()
            return False, "历史校准记录格式错误"

        if abs(slope) < 1e-12:
            ring.clear_time_calibration()
            return False, "历史校准 slope 无效"

        ring.reset_time_calib_data()

        calib_packet_num = 3
        calib_packet_interval = 0.1
        cnt = 0

        for _ in trange(calib_packet_num):
            await device.ring_logic.calib_time(index=cnt)
            await asyncio.sleep(calib_packet_interval)
            cnt += 1
        await asyncio.sleep(1)

        ring.start_calib_timestamps = np.sort(ring.start_calib_timestamps)
        ring.end_calib_timestamps = np.sort(ring.end_calib_timestamps)
        ring.ring_timestamps = np.sort(ring.ring_timestamps)
        ring.end_indices = np.sort(np.array(ring.end_indices))
        ring.start_calib_timestamps = ring.start_calib_timestamps[ring.end_indices]
        assert len(ring.start_calib_timestamps) == len(ring.ring_timestamps)
        assert len(ring.end_calib_timestamps) == len(ring.ring_timestamps)

        expected_host_ts = intercept + slope * ring.ring_timestamps
        current_host_ts = (ring.start_calib_timestamps + ring.end_calib_timestamps) / 2
        delta_ms = np.mean(np.abs(current_host_ts - expected_host_ts)) * 1000.0
        if delta_ms <= tolerance_ms:
            ring.set_time_calibration(intercept, slope)
            return True, f"已加载历史校准（偏差 {delta_ms:.2f}ms）"

        ring.clear_time_calibration()
        return False, f"历史校准失效（偏差 {delta_ms:.2f}ms），请手动同步"


# ============================================================================
# 4. UI 组件 (View)
# ============================================================================


class DeviceCard(QFrame):
    """
    设备卡片 UI 组件。
    通过信号与 MainWindow 交互，避免直接持有 MainWindow 引用。
    通过回调函数与 RingLogic 交互，避免直接持有 RingLogic 引用。
    通过刷新函数更新 UI 状态，避免 MainWindow 直接操作 UI组件。
    """

    sig_action = pyqtSignal(str, str)  # address, action

    def __init__(self, device: RingDevice):
        super().__init__()
        self.device = device  # 持有设备引用，方便访问状态和逻辑
        self.init_ui()
        self.refresh_view()

    def init_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(360, 240)
        layout = QVBoxLayout(self)

        # 第一行：名称
        row1 = QHBoxLayout()
        self.lbl_name = QLabel(self.device.name)
        self.lbl_name.setProperty("class", "title")
        row1.addWidget(self.lbl_name)
        row1.addStretch()
        layout.addLayout(row1)

        # 第二行：地址
        self.lbl_addr = QLabel(self.device.address)
        self.lbl_addr.setProperty("class", "sub-text")
        layout.addWidget(self.lbl_addr)

        # 第三行：状态
        self.lbl_status = QLabel("● 未连接")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

        # 4. 按钮区
        btn_layout = QGridLayout()
        self.btn_connect = QPushButton("连接")
        self.btn_connect.clicked.connect(
            lambda: self.sig_action.emit(self.device.address, "connect")
        )

        self.btn_sync = QPushButton("同步")
        self.btn_sync.setProperty("class", "btn-gray")
        self.btn_sync.clicked.connect(
            lambda: self.sig_action.emit(self.device.address, "sync")
        )

        btn_layout.addWidget(self.btn_connect, 0, 0, 1, 2)
        btn_layout.addWidget(self.btn_sync, 1, 0, 1, 2)
        layout.addLayout(btn_layout)

        # 5. 状态信息栏 (电量 | IMU FPS | 音频 FPS)
        stat_row = QHBoxLayout()
        stat_row.setSpacing(8)

        self.lbl_battery = QLabel("电量：--%")
        self.lbl_battery.setProperty("class", "stat-tag")
        self.lbl_battery.setVisible(True)

        self.lbl_imu_fps = QLabel("IMU：--Hz")
        self.lbl_imu_fps.setProperty("class", "stat-tag")
        self.lbl_imu_fps.setVisible(True)

        self.lbl_audio_fps = QLabel("音频：--Hz")
        self.lbl_audio_fps.setProperty("class", "stat-tag")
        self.lbl_audio_fps.setVisible(True)

        stat_row.addWidget(self.lbl_battery)
        stat_row.addWidget(self.lbl_imu_fps)
        stat_row.addWidget(self.lbl_audio_fps)
        # stat_row.addStretch()
        layout.addLayout(stat_row)

    def update_progress(self, current, total):
        """由外部调用的槽函数，用于更新进度条"""
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)

        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

        if current >= total:
            self.lbl_status.setText(f"同步完成")
            self.progress_bar.setVisible(False)

    def refresh_view(self):
        self.lbl_name.setText(self.device.name)

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

        self.lbl_battery.setText(
            f"电量：{self.device.ring_logic.battery_level if self.device.ring_logic.battery_level > 0 else '--'}%"
        )
        self.lbl_imu_fps.setText(f"IMU：{self.device.ring_logic.imu_fps:.1f}Hz")
        self.lbl_audio_fps.setText(f"音频：{self.device.ring_logic.audio_fps:.1f}Hz")

        self.style().unpolish(self)
        self.style().polish(self)
        self.style().polish(self.btn_sync)


# ============================================================================
# 5. 主窗口 (Controller) - Logic Refactored
# ============================================================================


class MainWindow(QMainWindow):
    STREAM_HOST = "127.0.0.1"
    STREAM_PORT = 17888

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ring Device Manager")
        self.resize(900, 600)

        self.devices: Dict[str, RingDevice] = {}
        self.cards: Dict[str, DeviceCard] = {}
        self.stream_hub = ImuStreamHub(host=self.STREAM_HOST, port=self.STREAM_PORT)
        self.app_processes: list[subprocess.Popen] = []
        self._stream_started = False
        self._port_switch_task: asyncio.Task | None = None
        self._stream_mode_task: asyncio.Task | None = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # 顶部操作栏
        top_panel = QFrame()
        top_panel.setProperty("class", "panel")
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(16, 14, 16, 14)
        top_layout.setSpacing(10)
        row_primary = QHBoxLayout()
        row_primary.setSpacing(10)
        row_secondary = QHBoxLayout()
        row_secondary.setSpacing(10)
        self.btn_scan = QPushButton("开始扫描")
        self.btn_scan.setProperty("class", "btn-primary")
        self.btn_scan.clicked.connect(self.start_scan)  # 连接到 asyncSlot

        self.port_input = QSpinBox()
        self.port_input.setRange(1024, 65535)
        self.port_input.setValue(self.STREAM_PORT)
        self.port_input.setFixedWidth(110)
        self.port_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.port_input.valueChanged.connect(self.on_port_changed)

        self.chk_imu_stream = QCheckBox("IMU")
        self.chk_imu_stream.setChecked(True)
        self.chk_imu_stream.stateChanged.connect(self.on_stream_mode_changed)

        self.chk_audio_stream = QCheckBox("音频")
        self.chk_audio_stream.setChecked(False)
        self.chk_audio_stream.stateChanged.connect(self.on_stream_mode_changed)

        self.chk_auto_reconnect = QCheckBox("自动重连")
        self.chk_auto_reconnect.setChecked(True)
        self.chk_auto_reconnect.stateChanged.connect(self.on_auto_reconnect_changed)

        self.lbl_msg = QLabel("就绪")
        self.lbl_msg.setObjectName("statusTag")

        row_primary.addWidget(self.btn_scan)
        row_primary.addWidget(QLabel("Port"))
        row_primary.addWidget(self.port_input)
        row_primary.addWidget(self.lbl_msg, 1)
        row_primary.addStretch()
        row_secondary.addWidget(QLabel("转发流"))
        row_secondary.addWidget(self.chk_imu_stream)
        row_secondary.addWidget(self.chk_audio_stream)
        row_secondary.addSpacing(12)
        row_secondary.addWidget(self.chk_auto_reconnect)
        row_secondary.addStretch()
        top_layout.addLayout(row_primary)
        top_layout.addLayout(row_secondary)
        top_panel.setLayout(top_layout)
        main_layout.addWidget(top_panel)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.scroll_content = QWidget()
        self.card_layout = QHBoxLayout(self.scroll_content)
        self.card_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.card_layout.setSpacing(15)

        scroll.setWidget(self.scroll_content)
        main_layout.addWidget(scroll)

        apps_panel = QFrame()
        apps_panel.setProperty("class", "panel")
        apps_row = QGridLayout()
        apps_row.setContentsMargins(16, 14, 16, 14)
        apps_row.setHorizontalSpacing(10)
        apps_row.setVerticalSpacing(10)
        lbl_apps = QLabel("应用")
        lbl_apps.setProperty("class", "title")
        apps_row.addWidget(lbl_apps, 0, 0)

        btn_demo = QPushButton("示范应用")
        btn_demo.setProperty("class", "btn-primary")
        btn_demo.clicked.connect(
            lambda: self.launch_app("apps/demo_processor_app.py", "示范应用")
        )
        apps_row.addWidget(btn_demo, 0, 1)

        btn_visual = QPushButton("IMU 可视化")
        btn_visual.setProperty("class", "btn-primary")
        btn_visual.clicked.connect(
            lambda: self.launch_app("apps/imu_visualizer_app.py", "IMU 可视化")
        )
        apps_row.addWidget(btn_visual, 0, 2)

        btn_audio = QPushButton("音频监控")
        btn_audio.setProperty("class", "btn-primary")
        btn_audio.clicked.connect(
            lambda: self.launch_app("apps/audio_monitor_app.py", "音频监控")
        )
        apps_row.addWidget(btn_audio, 0, 3)

        btn_recorder = QPushButton("录制中心")
        btn_recorder.setProperty("class", "btn-primary")
        btn_recorder.clicked.connect(
            lambda: self.launch_app("apps/recorder_app.py", "录制中心")
        )
        apps_row.addWidget(btn_recorder, 0, 4)

        btn_gesture = QPushButton("手势识别")
        btn_gesture.setProperty("class", "btn-primary")
        btn_gesture.clicked.connect(
            lambda: self.launch_app("apps/gesture_recognition_app.py", "手势识别")
        )
        apps_row.addWidget(btn_gesture, 0, 5)

        btn_touchpad = QPushButton("触控板")
        btn_touchpad.setProperty("class", "btn-primary")
        btn_touchpad.clicked.connect(
            lambda: self.launch_app("apps/touchpad_app.py", "触控板")
        )
        apps_row.addWidget(btn_touchpad, 0, 6)

        btn_whisper_number = QPushButton("数字书写识别")
        btn_whisper_number.setProperty("class", "btn-primary")
        btn_whisper_number.clicked.connect(
            lambda: self.launch_app("apps/whisper_number_touch_app.py", "数字书写识别")
        )
        apps_row.addWidget(btn_whisper_number, 0, 7)

        btn_whisper_word = QPushButton("英文小写识别")
        btn_whisper_word.setProperty("class", "btn-primary")
        btn_whisper_word.clicked.connect(
            lambda: self.launch_app("apps/whisper_word_touch_app.py", "英文小写识别")
        )
        apps_row.addWidget(btn_whisper_word, 0, 8)
        apps_row.setColumnStretch(9, 1)
        apps_panel.setLayout(apps_row)
        main_layout.addWidget(apps_panel)

    async def start_background_services(self):
        await self.stream_hub.start()
        self._stream_started = True
        self.lbl_msg.setText(
            f"就绪（数据流服务 {self.STREAM_HOST}:{self.STREAM_PORT} 已启动）"
        )

    def on_port_changed(self, port: int):
        self.STREAM_PORT = int(port)
        if not self._stream_started:
            return
        if self._port_switch_task and not self._port_switch_task.done():
            self._port_switch_task.cancel()
        self._port_switch_task = asyncio.create_task(self.restart_stream_hub())

    async def restart_stream_hub(self):
        try:
            self.lbl_msg.setText(f"正在切换数据流端口到 {self.STREAM_PORT}...")
            await self.stream_hub.stop()
            self.stream_hub = ImuStreamHub(host=self.STREAM_HOST, port=self.STREAM_PORT)
            for device in self.devices.values():
                self.stream_hub.upsert_ring(
                    device.address, device.name, connected=device.is_connected
                )
            await self.stream_hub.start()
            self.lbl_msg.setText(
                f"就绪（数据流服务 {self.STREAM_HOST}:{self.STREAM_PORT} 已启动）"
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.lbl_msg.setText(f"端口切换失败: {exc}")

    def on_stream_mode_changed(self, _state: int):
        if self._stream_mode_task and not self._stream_mode_task.done():
            self._stream_mode_task.cancel()
        self._stream_mode_task = asyncio.create_task(self.apply_stream_mode_to_connected())

    def on_auto_reconnect_changed(self, _state: int):
        enabled = self.chk_auto_reconnect.isChecked()
        for device in self.devices.values():
            if device.ring_logic is not None:
                device.ring_logic.set_auto_reconnect(enabled)
        self.lbl_msg.setText(f"自动重连已{'开启' if enabled else '关闭'}")

    async def apply_stream_mode_to_connected(self):
        enabled_imu = self.chk_imu_stream.isChecked()
        enabled_audio = self.chk_audio_stream.isChecked()
        tasks = []
        for device in self.devices.values():
            if device.is_connected:
                tasks.append(
                    self.apply_stream_mode_to_device(
                        device=device,
                        enabled_imu=enabled_imu,
                        enabled_audio=enabled_audio,
                    )
                )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.lbl_msg.setText(
            f"流设置已应用（IMU: {'开' if enabled_imu else '关'}，音频: {'开' if enabled_audio else '关'}）"
        )

    async def apply_stream_mode_to_device(
        self, device: RingDevice, enabled_imu: bool, enabled_audio: bool
    ):
        if not device.is_connected:
            return
        try:
            if enabled_imu:
                await DeviceService.open_imu(device)
            else:
                await DeviceService.close_imu(device)
        except Exception as exc:
            print(f"切换 IMU 流失败({device.name}): {exc}")

        try:
            if enabled_audio:
                await DeviceService.open_audio(device)
            else:
                await DeviceService.close_audio(device)
        except Exception as exc:
            print(f"切换音频流失败({device.name}): {exc}")

    def launch_app(self, app_rel_path: str, app_name: str):
        script = Path(__file__).resolve().parent / app_rel_path
        if not script.exists():
            self.lbl_msg.setText(f"{app_name} 启动失败：未找到 {app_rel_path}")
            return

        cmd = [
            sys.executable,
            str(script),
            "--host",
            self.STREAM_HOST,
            "--port",
            str(self.STREAM_PORT),
        ]
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            start_new_session=True,
        )
        self.app_processes.append(process)
        self.lbl_msg.setText(f"{app_name} 已启动（PID: {process.pid}）")

    async def finalize_device_connection(
        self, device: RingDevice, *, is_reconnect: bool
    ) -> tuple[bool, str]:
        device.is_connected = True
        device.is_calibrated = False
        await self.apply_stream_mode_to_device(
            device=device,
            enabled_imu=self.chk_imu_stream.isChecked(),
            enabled_audio=self.chk_audio_stream.isChecked(),
        )
        loaded, calib_msg = await DeviceService.load_calib_if_consistent(device)
        device.is_calibrated = loaded
        self.stream_hub.upsert_ring(device.address, device.name, connected=True)
        self.refresh_card(device.address)
        if is_reconnect:
            if loaded:
                self.lbl_msg.setText(f"{device.name} 已自动重连，{calib_msg}")
            else:
                self.lbl_msg.setText(f"{device.name} 已自动重连，未同步（{calib_msg}）")
        return loaded, calib_msg

    async def handle_reconnect_success(self, address: str):
        try:
            device = self.devices.get(address)
            if device is None or device.ring_logic is None or not device.ring_logic.connected:
                return
            await self.finalize_device_connection(device, is_reconnect=True)
        except Exception as exc:
            device = self.devices.get(address)
            device_name = device.name if device is not None else address
            self.lbl_msg.setText(f"{device_name} 自动重连后恢复失败: {exc}")
            import traceback

            traceback.print_exc()

    # ------------------------------------------------------------------------
    # 异步逻辑核心
    # ------------------------------------------------------------------------

    @asyncSlot()
    async def start_scan(self):
        """
        异步扫描槽函数。
        """
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("扫描中...")
        self.lbl_msg.setText("正在扫描蓝牙环境...")

        try:
            # 直接 await 异步服务
            scanned_list = await DeviceService.scan_ble()

            # 扫描完成，直接在下方处理 UI 更新（因为我们仍在主线程上下文中）
            self.process_scan_results(scanned_list)

        except Exception as e:
            self.lbl_msg.setText(f"扫描错误: {e}")
        finally:
            self.btn_scan.setEnabled(True)
            self.btn_scan.setText("开始扫描")

    def process_scan_results(self, scanned_list: List[dict]):
        """处理扫描结果"""
        found_addresses = set()

        for data in scanned_list:
            addr = data["address"]
            name = data["name"]
            found_addresses.add(addr)

            if addr in self.devices:
                self.devices[addr].update_info(name)
                self.stream_hub.upsert_ring(
                    addr, name, connected=self.devices[addr].is_connected
                )
                if addr in self.cards:
                    self.cards[addr].refresh_view()
            else:
                new_device = RingDevice(addr, name)
                new_device.ring_logic = BLERing(
                    addr,
                    index=len(self.devices),
                    name=name,
                    imu_callback=lambda _, imu, address=addr: self.handle_imu_data(
                        address, imu
                    ),
                    touch_callback=lambda _, action_code, address=addr: self.handle_touch_data(
                        address, action_code
                    ),
                    audio_callback=lambda _, audio_type, audio, local_ts, host_ts, address=addr: self.handle_audio_data(
                        address, audio_type, audio, local_ts, host_ts
                    ),
                    update_view_callback=lambda address: self.refresh_card(address),
                    reconnect_success_callback=lambda address, self=self: self.handle_reconnect_success(
                        address
                    ),
                    auto_reconnect=self.chk_auto_reconnect.isChecked(),
                )
                self.devices[addr] = new_device
                self.stream_hub.upsert_ring(addr, name, connected=False)
                self.add_card_ui(new_device)

        current_addresses = set(self.devices.keys())
        missing_addresses = current_addresses - found_addresses

        for addr in missing_addresses:
            device = self.devices[addr]
            if not device.is_connected:
                self.remove_card_ui(addr)
                self.stream_hub.remove_ring(addr)
                del self.devices[addr]

        self.lbl_msg.setText(f"扫描结束。当前设备数: {len(self.devices)}")

    def refresh_card(self, address: str):
        device = self.devices.get(address)
        if device is None:
            return

        ring = device.ring_logic
        ring_connected = bool(
            ring is not None and getattr(ring, "connected", False)
        )
        device.is_reconnecting = bool(
            ring is not None and getattr(ring, "is_reconnecting", False)
        )
        if device.is_connected != ring_connected:
            device.is_connected = ring_connected
            if not ring_connected:
                device.is_recording = False
                device.is_calibrated = False
                self.lbl_msg.setText(f"{device.name} 已自动断开")
                if (
                    ring is not None
                    and ring.auto_reconnect
                    and not getattr(ring, "_manual_disconnect", False)
                ):
                    self.lbl_msg.setText(f"{device.name} 已断开，正在自动重连...")
            self.stream_hub.upsert_ring(
                device.address, device.name, connected=device.is_connected
            )

        device.is_calibrated = bool(
            ring is not None and getattr(ring, "is_time_calibrated", False)
        )

        if address in self.cards:
            self.cards[address].refresh_view()

    def handle_imu_data(self, address: str, imu):
        device = self.devices.get(address)
        if device is None:
            return
        self.stream_hub.publish_imu(
            address=address,
            name=device.name,
            local_timestamp=float(getattr(imu, "local_timestamp", 0.0)),
            host_timestamp=float(getattr(imu, "host_timestamp", 0.0)),
            acc_x=imu.acc_x,
            acc_y=imu.acc_y,
            acc_z=imu.acc_z,
            gyr_x=imu.gyr_x,
            gyr_y=imu.gyr_y,
            gyr_z=imu.gyr_z,
        )

    def handle_audio_data(
        self,
        address: str,
        audio_type: int,
        audio: bytes,
        local_timestamp: float,
        host_timestamp: float,
    ):
        device = self.devices.get(address)
        if device is None:
            return
        self.stream_hub.publish_audio(
            address=address,
            name=device.name,
            local_timestamp=local_timestamp,
            host_timestamp=host_timestamp,
            audio_type=audio_type,
            audio_bytes=bytes(audio),
        )

    def handle_touch_data(self, address: str, action_code: int):
        device = self.devices.get(address)
        if device is None:
            return
        self.stream_hub.publish_touch(
            address=address,
            name=device.name,
            host_timestamp=time.time(),
            action_code=int(action_code),
        )

    def add_card_ui(self, device: RingDevice):
        card = DeviceCard(device)
        # 信号连接到 asyncSlot
        card.sig_action.connect(self.handle_card_action)
        self.card_layout.addWidget(card)
        self.cards[device.address] = card

    def remove_card_ui(self, address: str):
        if address in self.cards:
            card = self.cards.pop(address)
            self.card_layout.removeWidget(card)
            card.deleteLater()

    # ------------------------------------------------------------------------
    # 交互处理 - 使用 asyncSlot
    # ------------------------------------------------------------------------

    @asyncSlot(str, str)
    async def handle_card_action(self, address: str, action: str):
        """
        统一处理卡片动作的异步槽
        """
        device = self.devices.get(address)
        if not device:
            return

        # 找到 UI 卡片进行交互锁定
        card = self.cards.get(address)
        if card:
            card.setEnabled(False)

        try:
            if action == "connect":
                if device.is_reconnecting:
                    self.lbl_msg.setText(f"正在取消 {device.name} 的自动重连...")
                    if device.ring_logic is not None:
                        device.ring_logic.cancel_reconnect()
                    device.is_reconnecting = False
                    device.is_connected = False
                    device.is_calibrated = False
                    self.stream_hub.upsert_ring(
                        device.address, device.name, connected=False
                    )
                    self.lbl_msg.setText(f"{device.name} 已取消自动重连")
                elif device.is_connected:
                    self.lbl_msg.setText(f"正在断开 {device.name}...")
                    await DeviceService.disconnect_device(device)
                    device.is_connected = False
                    device.is_calibrated = False
                    self.stream_hub.upsert_ring(
                        device.address, device.name, connected=False
                    )
                    self.lbl_msg.setText(f"{device.name} 已断开")
                else:
                    self.lbl_msg.setText(f"正在连接 {device.name}...")
                    await DeviceService.connect_device(device)
                    loaded, calib_msg = await self.finalize_device_connection(
                        device, is_reconnect=False
                    )
                    if loaded:
                        self.lbl_msg.setText(f"{device.name} 连接成功，{calib_msg}")
                    else:
                        self.lbl_msg.setText(f"{device.name} 连接成功，未同步（{calib_msg}）")

            elif action == "sync":
                self.lbl_msg.setText(f"正在同步 {device.name}...")
                await DeviceService.sync_time(
                    device, progress_callback=card.update_progress
                )
                device.is_calibrated = True
                self.lbl_msg.setText(f"{device.name} 同步完成")

        except Exception as e:
            self.lbl_msg.setText(f"操作失败: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # 无论成功失败，恢复 UI 状态
            if card:
                card.setEnabled(True)
                card.refresh_view()

    def closeEvent(self, event):
        """
        拦截窗口关闭事件。
        如果是用户点击关闭，先忽略该事件，启动异步清理任务。
        """
        # 检查是否已经清理完毕
        if getattr(self, "_is_cleaned_up", False):
            event.accept()  # 已清理，允许关闭
            return

        # 1. 忽略当前的关闭请求（防止窗口立即消失）
        event.ignore()

        # 2. 启动清理任务，并传入当前窗口作为引用
        self.lbl_msg.setText("正在清理资源，请稍候...")
        asyncio.create_task(self.shutdown_and_close())

    async def shutdown_and_close(self):
        """
        执行异步清理：断开蓝牙、取消任务
        """
        print("正在开始清理...")

        tasks = []
        for addr, device in self.devices.items():
            if device.is_connected:
                print(f"正在断开: {device.name}")
                # 即使断开出错也不要卡住退出
                tasks.append(self.safe_disconnect(device))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # 2. (可选) 这里可以取消其他正在运行的后台任务
        # pending = asyncio.all_tasks()
        # ... cancel logic ...

        print("清理完成，正在退出。")

        # 3. 标记清理完成，并重新手动关闭窗口
        if self._port_switch_task and not self._port_switch_task.done():
            self._port_switch_task.cancel()
        if self._stream_mode_task and not self._stream_mode_task.done():
            self._stream_mode_task.cancel()
        await self.stream_hub.stop()
        self._stream_started = False
        self._is_cleaned_up = True

        # 再次调用 close()，这次会触发 closeEvent 并进入 if 分支
        self.close()

    async def safe_disconnect(self, device):
        """辅助函数：安全断开，忽略错误"""
        try:
            await DeviceService.disconnect_device(device)
        except Exception as e:
            print(f"断开 {device.name} 时出错 (可忽略): {e}")


# ============================================================================
# 6. 程序入口 - 启动 qasync 循环
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    # 创建 qasync 事件循环
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    win = MainWindow()
    win.show()
    asyncio.get_event_loop().create_task(win.start_background_services())

    # 使用 loop.run_forever() 替代 app.exec()
    with loop:
        loop.run_forever()
