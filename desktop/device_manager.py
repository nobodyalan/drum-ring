"""Coordinates device discovery, BLE callbacks, and device-card widgets."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from PyQt6.QtWidgets import QHBoxLayout

from core.ble_ring_v2 import BLERing

from .device_card import DeviceCard
from .device_runtime import DeviceService, RingDevice
from .swipe_runtime import SwipeGestureEvent, SwipeRuntime
from .stream_runtime import StreamMode, StreamRuntime

logger = logging.getLogger(__name__)


class DeviceManager:
    def __init__(
        self,
        *,
        card_layout: QHBoxLayout,
        stream_runtime: StreamRuntime,
        get_stream_mode: Callable[[], StreamMode],
        auto_reconnect_enabled: Callable[[], bool],
        set_status: Callable[[str], None],
        on_card_action: Callable[[str, str], None],
    ):
        self.card_layout = card_layout
        self.stream_runtime = stream_runtime
        self.get_stream_mode = get_stream_mode
        self.auto_reconnect_enabled = auto_reconnect_enabled
        self.set_status = set_status
        self.on_card_action = on_card_action
        self.devices: dict[str, RingDevice] = {}
        self.cards: dict[str, DeviceCard] = {}
        self.swipe_runtime = SwipeRuntime(
            on_event=self.handle_swipe_event,
            on_status=self.set_status,
        )

    async def scan(self) -> None:
        scanned_list = await DeviceService.scan_ble()
        self.process_scan_results(scanned_list)

    def process_scan_results(self, scanned_list: list[dict]) -> None:
        found_addresses = set()

        for data in scanned_list:
            address = data["address"]
            name = data["name"]
            found_addresses.add(address)

            if address in self.devices:
                device = self.devices[address]
                device.update_info(name)
                self.stream_runtime.upsert_ring(
                    address, name, connected=device.is_connected
                )
                self.refresh_card(address)
                continue

            device = self._build_device(address=address, name=name)
            self.devices[address] = device
            self.stream_runtime.upsert_ring(address, name, connected=False)
            self._add_card(device)

        self._remove_missing_devices(found_addresses)
        self.set_status(f"扫描结束。当前设备数: {len(self.devices)}")

    def set_auto_reconnect(self, enabled: bool) -> None:
        for device in self.devices.values():
            if device.ring_logic is not None:
                device.ring_logic.set_auto_reconnect(enabled)

    async def apply_stream_mode_to_connected(self, mode: StreamMode) -> None:
        tasks = [
            self.apply_stream_mode_to_device(device=device, mode=mode)
            for device in self.devices.values()
            if device.is_connected
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def apply_stream_mode_to_device(
        self,
        *,
        device: RingDevice,
        mode: StreamMode,
    ) -> None:
        if not device.is_connected:
            return

        imu_enabled = bool(mode.enabled_imu or device.swipe_enabled)
        try:
            if imu_enabled:
                await DeviceService.open_imu(device)
            else:
                await DeviceService.close_imu(device)
        except Exception as exc:
            logger.warning("切换 IMU 流失败(%s): %s", device.name, exc)

        try:
            if mode.enabled_audio:
                await DeviceService.open_audio(device)
            else:
                await DeviceService.close_audio(device)
        except Exception as exc:
            logger.warning("切换音频流失败(%s): %s", device.name, exc)

    async def handle_card_action(self, address: str, action: str) -> None:
        device = self.devices.get(address)
        if device is None:
            return

        card = self.cards.get(address)
        if card is not None:
            card.setEnabled(False)

        try:
            if action == "connect":
                await self._handle_connect_action(device)
            elif action == "sync" and card is not None:
                await self._handle_sync_action(device, card)
        except Exception as exc:
            self.set_status(f"操作失败: {exc}")
            logger.exception("处理设备动作失败: %s %s", address, action)
        finally:
            if card is not None:
                card.setEnabled(True)
                card.refresh_view()

    def refresh_card(self, address: str) -> None:
        device = self.devices.get(address)
        if device is None:
            return

        ring = device.ring_logic
        ring_connected = bool(ring is not None and getattr(ring, "connected", False))
        device.is_reconnecting = bool(
            ring is not None and getattr(ring, "is_reconnecting", False)
        )
        if device.is_connected != ring_connected:
            device.is_connected = ring_connected
            if not ring_connected:
                device.is_calibrated = False
                device.recognizer_fps = 0.0
                self.swipe_runtime.reset_device(address)
                self.set_status(f"{device.name} 已自动断开")
                if (
                    ring is not None
                    and ring.auto_reconnect
                    and not getattr(ring, "_manual_disconnect", False)
                ):
                    self.set_status(f"{device.name} 已断开，正在自动重连...")
            self.stream_runtime.upsert_ring(
                device.address, device.name, connected=device.is_connected
            )

        device.is_calibrated = bool(
            ring is not None and getattr(ring, "is_time_calibrated", False)
        )
        device.recognizer_fps = self.swipe_runtime.recognizer_fps(address)

        card = self.cards.get(address)
        if card is not None:
            card.refresh_view()

    async def handle_reconnect_success(self, address: str) -> None:
        try:
            device = self.devices.get(address)
            if device is None or device.ring_logic is None or not device.ring_logic.connected:
                return
            await self._finalize_device_connection(device, is_reconnect=True)
        except Exception as exc:
            device = self.devices.get(address)
            device_name = device.name if device is not None else address
            self.set_status(f"{device_name} 自动重连后恢复失败: {exc}")
            logger.exception("自动重连恢复失败: %s", address)

    async def shutdown(self) -> None:
        tasks = [
            self._safe_disconnect(device)
            for device in self.devices.values()
            if device.is_connected
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.swipe_runtime.shutdown()

    def publish_imu(self, address: str, imu) -> None:
        device = self.devices.get(address)
        if device is None:
            return
        local_timestamp = float(getattr(imu, "local_timestamp", 0.0))
        host_timestamp = float(getattr(imu, "host_timestamp", 0.0))
        sample = [
            float(imu.acc_x),
            float(imu.acc_y),
            float(imu.acc_z),
            float(imu.gyr_x),
            float(imu.gyr_y),
            float(imu.gyr_z),
        ]
        event_timestamp = host_timestamp if host_timestamp > 0 else time.time()
        self.swipe_runtime.handle_imu(
            address=address,
            ring_name=device.name,
            sample=sample,
            event_timestamp=event_timestamp,
        )
        self.stream_runtime.publish_imu(
            address=address,
            name=device.name,
            local_timestamp=local_timestamp,
            host_timestamp=host_timestamp,
            acc_x=imu.acc_x,
            acc_y=imu.acc_y,
            acc_z=imu.acc_z,
            gyr_x=imu.gyr_x,
            gyr_y=imu.gyr_y,
            gyr_z=imu.gyr_z,
        )

    def publish_audio(
        self,
        address: str,
        audio_type: int,
        audio: bytes,
        local_timestamp: float,
        host_timestamp: float,
    ) -> None:
        device = self.devices.get(address)
        if device is None:
            return
        self.stream_runtime.publish_audio(
            address=address,
            name=device.name,
            local_timestamp=local_timestamp,
            host_timestamp=host_timestamp,
            audio_type=audio_type,
            audio_bytes=bytes(audio),
        )

    def publish_touch(self, address: str, action_code: int) -> None:
        device = self.devices.get(address)
        if device is None:
            return

        ring = device.ring_logic
        tap_timestamp = (
            ring.get_tap_timestamp() if hasattr(ring, "get_tap_timestamp") else None
        )
        release_timestamp = (
            ring.get_release_timestamp()
            if hasattr(ring, "get_release_timestamp")
            else None
        )
        self.stream_runtime.publish_touch(
            address=address,
            name=device.name,
            host_timestamp=time.time(),
            action_code=int(action_code),
            tap_timestamp=tap_timestamp,
            release_timestamp=release_timestamp,
        )
        if tap_timestamp is not None and hasattr(ring, "clear_tap_timestamp"):
            ring.clear_tap_timestamp()
        if release_timestamp is not None and hasattr(ring, "clear_release_timestamp"):
            ring.clear_release_timestamp()

    async def set_swipe_enabled(self, address: str, enabled: bool) -> None:
        device = self.devices.get(address)
        if device is None:
            return
        device.swipe_enabled = bool(enabled)
        self.swipe_runtime.set_enabled(address, device.name, device.swipe_enabled)
        if device.is_connected:
            await self.apply_stream_mode_to_device(device=device, mode=self.get_stream_mode())
        self.refresh_card(address)

    def handle_swipe_event(self, event: SwipeGestureEvent) -> None:
        device = self.devices.get(event.address)
        if device is not None:
            device.recognizer_fps = self.swipe_runtime.recognizer_fps(event.address)
            self.refresh_card(event.address)
        self.stream_runtime.publish_swipe(
            address=event.address,
            name=event.ring_name,
            host_timestamp=event.timestamp,
            event=event.event,
            class_id=event.class_id,
            confidence=event.confidence,
            probabilities=list(event.probabilities),
            total_count=event.total_count,
        )

    def rings_snapshot(self) -> list[tuple[str, str, bool]]:
        return [
            (device.address, device.name, device.is_connected)
            for device in self.devices.values()
        ]

    def _build_device(self, *, address: str, name: str) -> RingDevice:
        device = RingDevice(address, name)
        device.ring_logic = BLERing(
            address,
            index=len(self.devices),
            name=name,
            imu_callback=lambda _, imu, ring_address=address: self.publish_imu(
                ring_address, imu
            ),
            touch_callback=lambda _, action_code, ring_address=address: self.publish_touch(
                ring_address, action_code
            ),
            audio_callback=lambda _, audio_type, audio, local_ts, host_ts, ring_address=address: self.publish_audio(
                ring_address, audio_type, audio, local_ts, host_ts
            ),
            update_view_callback=lambda ring_address: self.refresh_card(ring_address),
            reconnect_success_callback=lambda ring_address, self=self: self.handle_reconnect_success(
                ring_address
            ),
            auto_reconnect=self.auto_reconnect_enabled(),
        )
        return device

    def _add_card(self, device: RingDevice) -> None:
        card = DeviceCard(device)
        card.sig_action.connect(self.on_card_action)
        card.sig_swipe_toggled.connect(
            lambda address, enabled, self=self: asyncio.create_task(
                self.set_swipe_enabled(address, enabled)
            )
        )
        self.card_layout.addWidget(card)
        self.cards[device.address] = card

    def _remove_missing_devices(self, found_addresses: set[str]) -> None:
        current_addresses = set(self.devices.keys())
        missing_addresses = current_addresses - found_addresses
        for address in missing_addresses:
            device = self.devices[address]
            if device.is_connected:
                continue
            self._remove_card(address)
            self.swipe_runtime.remove_device(address)
            self.stream_runtime.remove_ring(address)
            del self.devices[address]

    def _remove_card(self, address: str) -> None:
        card = self.cards.pop(address, None)
        if card is None:
            return
        self.card_layout.removeWidget(card)
        card.deleteLater()

    async def _handle_connect_action(self, device: RingDevice) -> None:
        if device.is_reconnecting:
            self.set_status(f"正在取消 {device.name} 的自动重连...")
            if device.ring_logic is not None:
                device.ring_logic.cancel_reconnect()
            device.is_reconnecting = False
            device.is_connected = False
            device.is_calibrated = False
            self.stream_runtime.upsert_ring(device.address, device.name, connected=False)
            self.set_status(f"{device.name} 已取消自动重连")
            return

        if device.is_connected:
            self.set_status(f"正在断开 {device.name}...")
            await DeviceService.disconnect_device(device)
            device.is_connected = False
            device.is_calibrated = False
            self.stream_runtime.upsert_ring(device.address, device.name, connected=False)
            self.set_status(f"{device.name} 已断开")
            return

        self.set_status(f"正在连接 {device.name}...")
        await DeviceService.connect_device(device)
        loaded, calib_msg = await self._finalize_device_connection(
            device, is_reconnect=False
        )
        if loaded:
            self.set_status(f"{device.name} 连接成功，{calib_msg}")
        else:
            self.set_status(f"{device.name} 连接成功，未同步（{calib_msg}）")

    async def _handle_sync_action(self, device: RingDevice, card: DeviceCard) -> None:
        self.set_status(f"正在同步 {device.name}...")
        await DeviceService.sync_time(device, progress_callback=card.update_progress)
        device.is_calibrated = True
        self.set_status(f"{device.name} 同步完成")

    async def _finalize_device_connection(
        self,
        device: RingDevice,
        *,
        is_reconnect: bool,
    ) -> tuple[bool, str]:
        device.is_connected = True
        device.is_calibrated = False
        await self.apply_stream_mode_to_device(device=device, mode=self.get_stream_mode())
        loaded, calib_msg = await DeviceService.load_calib_if_consistent(device)
        device.is_calibrated = loaded
        self.stream_runtime.upsert_ring(device.address, device.name, connected=True)
        self.refresh_card(device.address)
        if is_reconnect:
            if loaded:
                self.set_status(f"{device.name} 已自动重连，{calib_msg}")
            else:
                self.set_status(f"{device.name} 已自动重连，未同步（{calib_msg}）")
        return loaded, calib_msg

    async def _safe_disconnect(self, device: RingDevice) -> None:
        try:
            await DeviceService.disconnect_device(device)
        except Exception as exc:
            logger.warning("断开 %s 时出错，可忽略: %s", device.name, exc)
