from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qasync import asyncSlot

from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from apps.registry import APP_LAUNCH_SPECS, AppLaunchSpec
from .app_launcher import AppLauncher
from .device_manager import DeviceManager
from .main_window_ui import create_app_launch_panel, create_device_list_view, create_main_toolbar
from .stream_runtime import StreamMode, StreamRuntime


class MainWindow(QMainWindow):
    """Desktop shell: only wires UI events to the dedicated controllers."""

    STREAM_HOST = "127.0.0.1"
    STREAM_PORT = 17888

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ring Device Manager")
        self.resize(900, 600)

        workspace_root = Path(__file__).resolve().parents[1]
        self._is_cleaned_up = False
        self._port_switch_task: asyncio.Task | None = None
        self._stream_mode_task: asyncio.Task | None = None
        self.stream_runtime = StreamRuntime(
            host=self.STREAM_HOST,
            port=self.STREAM_PORT,
            command_handler=self.handle_stream_hub_command,
        )
        self.app_launcher = AppLauncher(workspace_root)
        self._build_ui()
        self.device_manager = DeviceManager(
            card_layout=self.card_layout,
            stream_runtime=self.stream_runtime,
            get_stream_mode=self.current_stream_mode,
            auto_reconnect_enabled=lambda: self.chk_auto_reconnect.isChecked(),
            set_status=self.set_status,
            on_card_action=self.handle_card_action,
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        toolbar = create_main_toolbar(
            initial_port=self.STREAM_PORT,
            on_scan=self.start_scan,
            on_port_changed=self.on_port_changed,
            on_stream_mode_changed=self.on_stream_mode_changed,
            on_auto_reconnect_changed=self.on_auto_reconnect_changed,
        )
        self.btn_scan = toolbar.btn_scan
        self.port_input = toolbar.port_input
        self.chk_imu_stream = toolbar.chk_imu_stream
        self.chk_audio_stream = toolbar.chk_audio_stream
        self.chk_auto_reconnect = toolbar.chk_auto_reconnect
        self.lbl_msg = toolbar.lbl_msg
        main_layout.addWidget(toolbar.panel)

        device_list = create_device_list_view()
        self.card_layout = device_list.card_layout
        main_layout.addWidget(device_list.scroll)

        apps_panel = create_app_launch_panel(
            app_specs=APP_LAUNCH_SPECS,
            on_launch=self.launch_app,
        )
        main_layout.addWidget(apps_panel.panel)

    def set_status(self, message: str) -> None:
        self.lbl_msg.setText(message)

    def current_stream_mode(self) -> StreamMode:
        return StreamMode(
            enabled_imu=self.chk_imu_stream.isChecked(),
            enabled_audio=self.chk_audio_stream.isChecked(),
        )

    async def start_background_services(self) -> None:
        await self.stream_runtime.start()
        self.set_status(
            f"就绪（数据流服务 {self.STREAM_HOST}:{self.STREAM_PORT} 已启动）"
        )

    def on_port_changed(self, port: int) -> None:
        self.STREAM_PORT = int(port)
        if not self.stream_runtime.started:
            return
        if self._port_switch_task and not self._port_switch_task.done():
            self._port_switch_task.cancel()
        self._port_switch_task = asyncio.create_task(self.restart_stream_runtime())

    async def restart_stream_runtime(self) -> None:
        try:
            self.set_status(f"正在切换数据流端口到 {self.STREAM_PORT}...")
            await self.stream_runtime.restart(
                port=self.STREAM_PORT,
                rings=self.device_manager.rings_snapshot(),
            )
            self.set_status(
                f"就绪（数据流服务 {self.STREAM_HOST}:{self.STREAM_PORT} 已启动）"
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.set_status(f"端口切换失败: {exc}")

    def on_stream_mode_changed(self, _state: int) -> None:
        if self._stream_mode_task and not self._stream_mode_task.done():
            self._stream_mode_task.cancel()
        self._stream_mode_task = asyncio.create_task(self.apply_stream_mode())

    async def apply_stream_mode(self) -> None:
        mode = self.current_stream_mode()
        await self.device_manager.apply_stream_mode_to_connected(mode)
        self.set_status(
            f"流设置已应用（IMU: {'开' if mode.enabled_imu else '关'}，音频: {'开' if mode.enabled_audio else '关'}）"
        )

    async def handle_stream_hub_command(self, _session, command: dict[str, Any]):
        if str(command.get("cmd", "")).strip() != "set_main_audio":
            return None
        enabled = bool(command.get("enabled", False))
        source = str(command.get("source", "remote") or "remote")
        await self.set_audio_stream_from_remote(enabled=enabled, source=source)
        return {
            "type": "ack",
            "cmd": "set_main_audio",
            "enabled": enabled,
            "source": source,
        }

    async def set_audio_stream_from_remote(self, *, enabled: bool, source: str) -> None:
        changed = self.chk_audio_stream.isChecked() != enabled
        if changed:
            self.chk_audio_stream.blockSignals(True)
            self.chk_audio_stream.setChecked(enabled)
            self.chk_audio_stream.blockSignals(False)
        if self.stream_runtime.started:
            if self._stream_mode_task and not self._stream_mode_task.done():
                self._stream_mode_task.cancel()
            await self.apply_stream_mode()
            return
        self.set_status(
            f"收到 {source} 的音频流请求（音频: {'开' if enabled else '关'}）"
        )

    def on_auto_reconnect_changed(self, _state: int) -> None:
        enabled = self.chk_auto_reconnect.isChecked()
        self.device_manager.set_auto_reconnect(enabled)
        self.set_status(f"自动重连已{'开启' if enabled else '关闭'}")

    def launch_app(self, spec: AppLaunchSpec) -> None:
        try:
            process = self.app_launcher.launch(
                spec,
                host=self.STREAM_HOST,
                port=self.STREAM_PORT,
            )
        except FileNotFoundError:
            self.set_status(f"{spec.title} 启动失败：未找到 {spec.script_path}")
            return

        self.set_status(f"{spec.title} 已启动（PID: {process.pid}）")

    @asyncSlot()
    async def start_scan(self) -> None:
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("扫描中...")
        self.set_status("正在扫描蓝牙环境...")
        try:
            await self.device_manager.scan()
        except Exception as exc:
            self.set_status(f"扫描错误: {exc}")
        finally:
            self.btn_scan.setEnabled(True)
            self.btn_scan.setText("开始扫描")

    @asyncSlot(str, str)
    async def handle_card_action(self, address: str, action: str) -> None:
        await self.device_manager.handle_card_action(address, action)

    def closeEvent(self, event) -> None:
        if self._is_cleaned_up:
            event.accept()
            return
        event.ignore()
        self.set_status("正在清理资源，请稍候...")
        asyncio.create_task(self.shutdown_and_close())

    async def shutdown_and_close(self) -> None:
        await self.device_manager.shutdown()
        if self._port_switch_task and not self._port_switch_task.done():
            self._port_switch_task.cancel()
        if self._stream_mode_task and not self._stream_mode_task.done():
            self._stream_mode_task.cancel()
        await self.stream_runtime.stop()
        self._is_cleaned_up = True
        self.close()
