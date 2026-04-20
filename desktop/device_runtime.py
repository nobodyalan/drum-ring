"""Device-side runtime helpers shared by the desktop manager."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bleak import BleakScanner
from tqdm import trange

from core.ble_ring_v2 import BLERing

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RingDevice:
    address: str
    name: str
    is_connected: bool = False
    is_reconnecting: bool = False
    is_calibrated: bool = False
    swipe_enabled: bool = False
    recognizer_fps: float = 0.0
    ring_logic: BLERing | None = None

    def update_info(self, name: str) -> None:
        self.name = name


class DeviceService:
    """Pure async BLE operations kept outside the Qt window layer."""

    @staticmethod
    async def scan_ble() -> list[dict]:
        devices = await BleakScanner.discover(timeout=3.0)
        results = []
        for device in devices:
            name = device.name or "Unknown"
            if name.startswith("BCL"):
                results.append({"name": name, "address": device.address})
        return results

    @staticmethod
    async def connect_device(device: RingDevice) -> bool:
        await device.ring_logic.connect()
        if device.ring_logic.connected:
            return True
        raise Exception("连接失败")

    @staticmethod
    async def disconnect_device(device: RingDevice) -> bool:
        await device.ring_logic.disconnect()
        return True

    @staticmethod
    async def open_imu(device: RingDevice) -> bool:
        await device.ring_logic.open_imu()
        return True

    @staticmethod
    async def close_imu(device: RingDevice) -> bool:
        await device.ring_logic.close_imu()
        return True

    @staticmethod
    async def open_audio(device: RingDevice) -> bool:
        await device.ring_logic.open_audio()
        return True

    @staticmethod
    async def close_audio(device: RingDevice) -> bool:
        await device.ring_logic.close_audio()
        return True

    @staticmethod
    async def sync_time(device: RingDevice, progress_callback=None) -> None:
        if not device.ring_logic.connected:
            raise Exception("设备未连接")

        ring = device.ring_logic
        ring.reset_time_calib_data()
        ring.clear_time_calibration()

        calib_packet_num = 300
        calib_packet_interval = 0.1
        max_calib_delta = 0.016
        count = 0

        for _ in trange(calib_packet_num):
            if progress_callback:
                progress_callback(count, calib_packet_num)
            await device.ring_logic.calib_time(index=count)
            await asyncio.sleep(calib_packet_interval)
            count += 1
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
        timestamps = sorted(timestamps, key=lambda item: item[2])
        timestamps = [item for item in timestamps if item[2] < max_calib_delta]
        logger.info(
            f"used calib packet num: {len(timestamps)}, percentage: {len(timestamps) / calib_packet_num}"
        )

        pc_timestamps, ring_timestamps, pc_delta_time = zip(*timestamps)

        ring_timestamps = np.array(ring_timestamps)
        pc_timestamps = np.array(pc_timestamps)
        pc_delta_time = np.array(pc_delta_time)
        x_with_bias = np.c_[np.ones((len(ring_timestamps), 1)), ring_timestamps]
        intercept, slope = np.linalg.inv(x_with_bias.T.dot(x_with_bias)).dot(
            x_with_bias.T
        ).dot(pc_timestamps)
        logger.info("time calibration ready: intercept=%s slope=%s", intercept, slope)

        ring.reset_time_calib_data()

        if progress_callback:
            progress_callback(calib_packet_num, calib_packet_num)

        if len(timestamps) < 20:
            raise Exception("校准数据不足，可能同步失败，最好重新同步")

        os.makedirs("data", exist_ok=True)
        ring.set_time_calibration(intercept, slope)
        with open(f"data/calib_time_{int(time.time())}.json", "w") as handle:
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
                handle,
            )

    @staticmethod
    def find_latest_calib_record(ring_mac: str) -> dict | None:
        data_dir = Path("data")
        if not data_dir.exists():
            return None

        files = sorted(
            data_dir.glob("calib_time_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for file_path in files:
            try:
                with open(file_path, "r") as handle:
                    payload = json.load(handle)
                if str(payload.get("ring_mac", "")).lower() == ring_mac.lower():
                    payload["__file"] = str(file_path)
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
        count = 0

        for _ in trange(calib_packet_num):
            await device.ring_logic.calib_time(index=count)
            await asyncio.sleep(calib_packet_interval)
            count += 1
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
