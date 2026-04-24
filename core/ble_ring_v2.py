import gc
import os
import time
import math
import asyncio
import inspect
import struct
from bleak import BleakClient
import queue
from types import FunctionType
from typing import Awaitable, Callable, Tuple

import math
import numpy as np


class MemmapStreamer:
    def __init__(
        self, filename, max_records, shape=(7,), dtype="float64", flush_interval=1000
    ):
        """
        初始化流式记录器
        :param filename: 保存的文件名
        :param max_records: 预估的最大行数（宁大勿小，最后会裁剪）
        :param shape: 单条数据的维度，例如 (7,)
        :param dtype: 数据类型
        :param flush_interval: 刷盘间隔，单位为写入条数
        """
        self.filename = filename
        self.temp_filename = filename + ".temp.dat"
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.idx = 0
        self.flush_interval = flush_interval

        # 1. 计算完整的文件大小并创建 Memmap
        # mode='w+' 会创建新文件或覆盖旧文件
        self.full_shape = (max_records,) + shape
        self.memmap_arr = np.memmap(
            self.temp_filename, dtype=self.dtype, mode="w+", shape=self.full_shape
        )

    def append(self, data):
        """
        写入一条数据
        :param data: 输入数据，可以是 (7,) 或 (1, 7)
        """
        if self.idx >= self.full_shape[0]:
            raise IndexError("预分配空间已满，请增大 max_records")

        if self.memmap_arr is not None:
            self.memmap_arr[self.idx] = data
            self.idx += 1
            if self.flush_interval > 0 and self.idx % self.flush_interval == 0:
                self.flush()

    def flush(self):
        """
        手动刷盘：建议每写入 N 条或每隔 T 秒调用一次
        """
        if self.memmap_arr is not None:
            self.memmap_arr.flush()

    def _close_memmap(self):
        mm = getattr(self, "memmap_arr", None)
        if mm is None:
            return
        try:
            mm.flush()
        except Exception:
            pass
        # memmap 底层 mmap 句柄显式 close
        try:
            if hasattr(mm, "_mmap") and mm._mmap is not None:
                mm._mmap.close()
        except Exception:
            pass
        self.memmap_arr = None

    def close(self):
        """
        关闭文件并裁剪多余空间
        """
        # 强制刷盘
        self.flush()

        print(f"录制结束，开始转换为 .npy (共 {self.idx} 条数据)...")
        valid_data = np.asarray(self.memmap_arr[: self.idx]).copy()

        self._close_memmap()

        np.save(self.filename, valid_data)

        del valid_data
        gc.collect()
        os.remove(self.temp_filename)


class IMUData:
    def __init__(
        self,
        acc_x: float,
        acc_y: float,
        acc_z: float,
        gyr_x: float,
        gyr_y: float,
        gyr_z: float,
        timestamp: float,
        local_timestamp: float | None = None,
        host_timestamp: float | None = None,
    ):
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.gyr_x = gyr_x
        self.gyr_y = gyr_y
        self.gyr_z = gyr_z
        self.local_timestamp = timestamp if local_timestamp is None else local_timestamp
        self.host_timestamp = 0.0 if host_timestamp is None else float(host_timestamp)
        # backward-compatible alias
        self.timestamp = self.local_timestamp
        self.acc_np = np.array([self.acc_x, self.acc_y, self.acc_z])
        self.gyr_np = np.array([self.gyr_x, self.gyr_y, self.gyr_z])
        self.imu_np = np.concatenate((self.acc_np, self.gyr_np), axis=0)
        self.plane_directions = np.array(
            [
                [-0.23709712, 0.20552186, -0.93578157],
                [-0.94889871, 0.04797022, 0.28491208],
                [-0.0617075, 0.95126743, 0.29782995],
                [0.99204434, -0.06982192, 0.00740044],
            ]
        )

    def __getitem__(self, index):
        return [self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z][
            index
        ]

    def __str__(self):
        return "Acc [x: {:.2f}  y: {:.2f}  z: {:.2f}]  Gyr[x: {:.2f}  y: {:.2f}  z: {:.2f}]  Timestamp {}".format(
            self.acc_x,
            self.acc_y,
            self.acc_z,
            self.gyr_x,
            self.gyr_y,
            self.gyr_z,
            self.timestamp,
        )

    @property
    def gyr_norm(self):
        return math.sqrt(
            self.gyr_x * self.gyr_x + self.gyr_y * self.gyr_y + self.gyr_z * self.gyr_z
        )

    @property
    def acc_norm(self):
        return math.sqrt(
            self.acc_x * self.acc_x + self.acc_y * self.acc_y + self.acc_z * self.acc_z
        )

    def scale(self):
        return IMUData(
            self.acc_x / 9.8,
            -self.acc_y / 9.8,
            -self.acc_z / 9.8,
            self.gyr_x / math.pi * 180,
            -self.gyr_y / math.pi * 180,
            -self.gyr_z / math.pi * 180,
            self.local_timestamp,
            self.local_timestamp,
            self.host_timestamp,
        )

    def direction(self):
        for i in range(len(self.plane_directions)):
            if (
                np.dot(self.acc_np / self.acc_norm, self.plane_directions[i])
                >= math.sqrt(2) / 2
            ):
                return i
        return -1

    def to_numpy(self):
        return np.array(
            [self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z],
            dtype=np.float32,
        )

    def to_numpy_with_timestamp(self):
        return np.array(
            [
                self.acc_x,
                self.acc_y,
                self.acc_z,
                self.gyr_x,
                self.gyr_y,
                self.gyr_z,
                self.local_timestamp,
            ],
            dtype=np.float64,
        )

    def assigned_by(self, data):
        # remain timestamp
        self.acc_x = data.acc_x
        self.acc_y = data.acc_y
        self.acc_z = data.acc_z
        self.gyr_x = data.gyr_x
        self.gyr_y = data.gyr_y
        self.gyr_z = data.gyr_z
        self.acc_np = np.array([self.acc_x, self.acc_y, self.acc_z])
        self.gyr_np = np.array([self.gyr_x, self.gyr_y, self.gyr_z])
        self.imu_np = np.concatenate((self.acc_np, self.gyr_np), axis=0)


class NotifyProtocol:
    READ_CHARACTERISTIC = "BAE80011-4F05-4503-8E65-3AF1F7329D1F"
    WRITE_CHARACTERISTIC = "BAE80010-4F05-4503-8E65-3AF1F7329D1F"

    GET_SOFTWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x00])
    GET_HARDWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x01])
    GET_BATTERY_LEVEL = bytearray([0x00, 0x00, 0x12, 0x00])
    GET_BATTERY_STATUS = bytearray([0x00, 0x00, 0x12, 0x01])
    OPEN_6AXIS_IMU = bytearray([0x00, 0x00, 0x40, 0x06, 0x00, 0x07, 0x00, 0x07, 0x10])
    CLOSE_6AXIS_IMU = bytearray([0x00, 0x00, 0x40, 0x00])
    CALIB_IMU = bytearray([0x00, 0x00, 0x40, 0x02])
    CALIB_TIME = bytearray([0x00, 0x00, 0x99, 0x00])
    GET_TOUCH = bytearray([0x00, 0x00, 0x61, 0x00])
    OPEN_MIC = bytearray([0x00, 0x00, 0x71, 0x01, 0x01])
    CLOSE_MIC = bytearray([0x00, 0x00, 0x71, 0x00])
    GET_NFC = bytearray([0x00, 0x00, 0x82, 0x00])


class BLERing:

    def __init__(
        self,
        address: str,
        index: int,
        name: str = "",
        gyro_bias: Tuple[float] = None,
        imu_callback=None,
        battery_callback=None,
        touch_callback=None,
        gesture_callback=None,
        audio_callback=None,
        imu_int8_callback=None,
        update_view_callback=None,
        reconnect_success_callback: Callable[[str], Awaitable[None] | None] | None = None,
        imu_freq=200.0,
        auto_reconnect: bool = True,
        reconnect_interval: float = 2.0,
    ):
        self.address = address
        self.index = index
        self.name = name
        if gyro_bias is None:
            self.gyro_bias = (0, 0, 0)
        else:
            self.gyro_bias = gyro_bias
        self.imu_mode = False
        self.raw_imu_data = bytearray()
        self.touch_callback = touch_callback
        self.gesture_callback = gesture_callback
        self.battery_callback = battery_callback
        self.imu_callback = imu_callback
        self.audio_callback = audio_callback
        self.imu_int8_callback = imu_int8_callback
        self.update_view_callback = update_view_callback
        self.reconnect_success_callback = reconnect_success_callback
        self.acc_fsr = "0"
        self.gyro_fsr = "0"
        self.imu_freq = imu_freq
        self.client = None
        self.connected = False
        self.action_queue = queue.Queue()
        self.auto_reconnect = bool(auto_reconnect)
        self.reconnect_interval = float(reconnect_interval)
        self._manual_disconnect = False
        self._connect_lock = asyncio.Lock()
        self._reconnect_task: asyncio.Task | None = None

        # timestamp related
        self.ring_timestamps = []
        self.end_calib_timestamps = []
        self.start_calib_timestamps = []
        self.end_indices = []

        # imu recording related
        self.is_recording = False
        self.imu_data_streamer = None

        # status
        self.battery_level = -1
        self.IMU_FPS_INTERVAL = 5.0  # seconds
        self.imu_fps = 0.0
        self.imu_fps_start_time = None
        self.imu_fps_count = 0
        self.AUDIO_FPS_INTERVAL = 2.0  # seconds
        self.audio_fps = 0.0
        self.audio_fps_start_time = None
        self.audio_fps_count = 0
        self.time_calib_intercept = 0.0
        self.time_calib_slope = 0.0
        self.is_time_calibrated = False
        self.tap_timestamp = None
        self.release_timestamp = None

    @property
    def is_reconnecting(self) -> bool:
        return self._reconnect_task is not None and not self._reconnect_task.done()

    def on_disconnect(self, clients):
        _ = clients
        self.connected = False
        self.is_recording = False
        self.imu_fps = 0.0
        self.audio_fps = 0.0
        self.imu_fps_count = 0
        self.audio_fps_count = 0
        self.imu_fps_start_time = None
        self.audio_fps_start_time = None
        self.is_time_calibrated = False
        if self.auto_reconnect and not self._manual_disconnect:
            self._schedule_reconnect()
        if self.update_view_callback is not None:
            self.update_view_callback(self.address)

    def set_auto_reconnect(self, enabled: bool):
        self.auto_reconnect = bool(enabled)
        if not self.auto_reconnect:
            self._cancel_reconnect_task()
        elif not self.connected and not self._manual_disconnect:
            self._schedule_reconnect()

    def _cancel_reconnect_task(self):
        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        self._reconnect_task = None

    def cancel_reconnect(self):
        self._manual_disconnect = True
        self._cancel_reconnect_task()
        if self.update_view_callback is not None:
            self.update_view_callback(self.address)

    def _schedule_reconnect(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        self._reconnect_task = loop.create_task(self._reconnect_loop())

    async def _invoke_reconnect_success_callback(self):
        if self.reconnect_success_callback is None:
            return
        result = self.reconnect_success_callback(self.address)
        if inspect.isawaitable(result):
            await result

    async def _reconnect_loop(self):
        while self.auto_reconnect and not self._manual_disconnect and not self.connected:
            try:
                print(f"尝试自动重连 {self.address}")
                await self.connect(is_reconnect=True)
                if self.connected:
                    print(f"自动重连成功 {self.address}")
                    self._reconnect_task = None
                    await self._invoke_reconnect_success_callback()
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"自动重连失败 {self.address}: {exc}")

            try:
                await asyncio.sleep(self.reconnect_interval)
            except asyncio.CancelledError:
                raise

    def notify_callback(self, sender, data: bytearray):
        # print(len(data))
        bias = self.gyro_bias
        if data[2] == 0x62 and data[3] == 0x1:
            print("呼吸灯结果")
        if data[2] == 0x62 and data[3] == 0x2:
            print("自定义灯结果")
        if data[2] == 0x62 and data[3] == 0x3:
            print("自定义灯pwm空闲结果")

        if data[2] == 0x10 and data[3] == 0x0:
            print(data[4])
        if data[2] == 0x11 and data[3] == 0x0:
            print("Software version:", data[4:])
        if data[2] == 0x11 and data[3] == 0x1:
            print("Hardware version:", data[4:])

        if data[2] == 0x40 and data[3] == 0x06:
            acc_scale = 32768 / 16 * (2 ** ((data[4] >> 2) & 3)) / 9.8
            gyr_scale = 32768 / 2000 * (2 ** (data[4] & 3)) / (math.pi / 180)
            if len(data) > 20:
                imu_data = []
                head_length = 4 + len(data) % 2

                imu_start_time = 0
                imu_end_time = 0
                if (len(data) - head_length) % 12 != 0:
                    # end with 2 timestamps
                    imu_start_time = struct.unpack("i", data[-8:-4])[0]
                    imu_end_time = struct.unpack("i", data[-4:])[0]
                    imu_packet_num = (len(data) - head_length - 8) // 12
                else:
                    imu_packet_num = (len(data) - head_length) // 12

                for i in range(head_length, len(data), 12):
                    if len(data) - i < 12:
                        break
                    acc_x, acc_y, acc_z = struct.unpack("hhh", data[i : i + 6])
                    acc_x, acc_y, acc_z = (
                        acc_x / acc_scale,
                        acc_y / acc_scale,
                        acc_z / acc_scale,
                    )
                    gyr_x, gyr_y, gyr_z = struct.unpack("hhh", data[i + 6 : i + 12])
                    gyr_x, gyr_y, gyr_z = (
                        gyr_x / gyr_scale,
                        gyr_y / gyr_scale,
                        gyr_z / gyr_scale,
                    )
                    if imu_start_time != 0 and imu_end_time != 0:
                        local_timestamp = (imu_start_time + (
                            (imu_end_time - imu_start_time) / (imu_packet_num - 1)
                        ) * ((i - head_length) // 12)
                        ) / 16384.0
                    else:
                        local_timestamp = 0.0
                    imu = IMUData(
                        -1 * acc_y,
                        acc_z,
                        -1 * acc_x,
                        -1 * gyr_y - bias[0],
                        gyr_z - bias[1],
                        -1 * gyr_x - bias[2],
                        local_timestamp,
                        local_timestamp,
                        self.to_host_timestamp(local_timestamp),
                    )
                    imu_data.append(imu)

                # IMU callback function
                if self.imu_callback is not None:
                    for i, imu in enumerate(imu_data):
                        self.imu_callback(self.index, imu)

                # recording
                if self.is_recording and self.imu_data_streamer is not None:
                    for imu in imu_data:
                        self.imu_data_streamer.append(imu.to_numpy_with_timestamp())

                # imu fps calculation
                self.imu_fps_count += len(imu_data)
                if self.imu_fps_start_time is None:
                    self.imu_fps_start_time = time.perf_counter()
                else:
                    elapsed = time.perf_counter() - self.imu_fps_start_time
                    if elapsed >= (1.0 if not self.imu_fps else self.IMU_FPS_INTERVAL):
                        self.imu_fps = self.imu_fps_count / elapsed
                        self.imu_fps_count = 0
                        self.imu_fps_start_time = time.perf_counter()
                        if self.update_view_callback is not None:
                            self.update_view_callback(self.address)

        elif data[2] == 0x61 and data[3] == 0x0:
            pass

        elif data[2] == 0x61 and data[3] == 0x1:
            # detect touch event
            pass

        elif data[2] == 0x61 and data[3] == 0x2:
            if data[4] in (0, 1):
                self.tap_timestamp = time.perf_counter()
            elif data[4] == 5:
                self.release_timestamp = time.perf_counter()
            if self.touch_callback is not None:
                self.touch_callback(self.index, data[4])

        elif data[2] == 0x12 and data[3] == 0x0:
            self.battery_level = data[4]
            if self.battery_callback is not None:
                self.battery_callback(self.index, self.battery_level)
            if self.update_view_callback is not None:
                self.update_view_callback(self.address)
            print(f"Ring {self.index} battery level: {self.battery_level}")

        elif data[2] == 0x99 and data[3] == 0x0:
            # timestamp
            self.end_calib_timestamps.append(time.perf_counter())
            self.end_indices.append(struct.unpack("<H", data[0:2])[0])
            self.ring_timestamps.append(struct.unpack("i", data[4:])[0] / 16384)

        elif data[2] == 0x71 and data[3] == 0x0:
            audio_type = 0
            audio = data[-200:]
            local_timestamp = 0.0
            if len(data) >= 8:
                local_timestamp = struct.unpack("i", data[4:8])[0] / 16384.0
            if self.audio_callback is not None:
                self.audio_callback(
                    self.index,
                    audio_type,
                    audio,
                    local_timestamp,
                    self.to_host_timestamp(local_timestamp),
                )
            self.update_audio_fps()

        elif data[2] == 0x71 and data[3] == 0x1:
            audio_type = 1
            audio = data[-200:]
            local_timestamp = 0.0
            if len(data) >= 8:
                local_timestamp = struct.unpack("i", data[4:8])[0] / 16384.0
            if self.audio_callback is not None:
                self.audio_callback(
                    self.index,
                    audio_type,
                    audio,
                    local_timestamp,
                    self.to_host_timestamp(local_timestamp),
                )
            self.update_audio_fps()

        elif data[2] == 0x97 and data[3] == 0x0:
            gesture_code = data[4]
            print(f"Ring {self.index} gesture code: {gesture_code}")
            if self.gesture_callback is not None:
                self.gesture_callback(
                    self.index,
                    gesture_code,
                    time.time(),
                )

        elif data[2] == 0x97 and data[3] == 0x01:
            packet_size = len(data[4:])  # should be multiple of 6
            imu_data_int8 = data[4:]
            # print(f"Ring {self.index} imu int8 data: {imu_data_int8}")

            z_m = (-0.143, -2.291, -5.800, 0.034, 0.049, -0.036)
            z_s = (5.403, 4.641, 5.621, 1.996, 2.390, 1.354)

            for i in range(packet_size // 6):
                raw = struct.unpack("6b", imu_data_int8[i * 6 : (i + 1) * 6])
                calibrated = [raw[j] / 16 * z_s[j] + z_m[j] for j in range(6)]

                imu = IMUData(*calibrated, time.perf_counter())
                self.imu_callback(self.index, imu)

    async def connect(self, callback: FunctionType = None, is_reconnect: bool = False):
        async with self._connect_lock:
            if self.connected and self.client is not None and self.client.is_connected:
                return

            self._manual_disconnect = False
            if not is_reconnect:
                self._cancel_reconnect_task()

            self.client = BleakClient(
                self.address, disconnected_callback=self.on_disconnect
            )
            try:
                print(f"Try to connect to {self.address}")
                await self.client.connect()
                print("Start notify")
                await self.client.start_notify(
                    NotifyProtocol.READ_CHARACTERISTIC, self.notify_callback
                )
                await asyncio.sleep(0.5)
                await self.client.write_gatt_char(
                    NotifyProtocol.WRITE_CHARACTERISTIC, NotifyProtocol.OPEN_6AXIS_IMU
                )
                await self.client.write_gatt_char(
                    NotifyProtocol.WRITE_CHARACTERISTIC, NotifyProtocol.GET_BATTERY_LEVEL
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                print(e)
                print(f"Failed to connect to {self.address}")
                self.connected = False
                try:
                    if self.client is not None and self.client.is_connected:
                        await self.client.disconnect()
                except Exception:
                    pass
                return

            if self.client.is_connected:
                print("Connected")
                self.connected = True
                if callback is not None:
                    callback()

    async def send_command(self, command: bytearray):
        await self.client.write_gatt_char(NotifyProtocol.WRITE_CHARACTERISTIC, command)
        await asyncio.sleep(0.1)

    async def calibrate_imu(self):
        await self.send_command(NotifyProtocol.CALIB_IMU)
        print("Calibrating IMU...")
        await asyncio.sleep(3.0)
        print("Calibration done")

    async def open_audio(self):
        await self.send_command(NotifyProtocol.OPEN_MIC)
        print("Audio opened")

    async def close_audio(self):
        await self.send_command(NotifyProtocol.CLOSE_MIC)
        print("Audio closed")

    async def open_imu(self):
        await self.send_command(NotifyProtocol.OPEN_6AXIS_IMU)
        print("IMU opened")

    async def close_imu(self):
        await self.send_command(NotifyProtocol.CLOSE_6AXIS_IMU)
        print("IMU closed")

    async def calib_time(self, index: int):
        # convert index to bytearray
        index = index & 0xFFFF
        index_bytes = index.to_bytes(2, byteorder="little")
        start_time = time.perf_counter()
        await self.client.write_gatt_char(
            NotifyProtocol.WRITE_CHARACTERISTIC,
            index_bytes + NotifyProtocol.CALIB_TIME[2:],
        )
        end_time = time.perf_counter()
        self.start_calib_timestamps.append((start_time + end_time) / 2)

    def reset_time_calib_data(self):
        self.ring_timestamps = []
        self.end_calib_timestamps = []
        self.start_calib_timestamps = []
        self.end_indices = []

    async def start_recording(self, filename=""):
        self.is_recording = True
        if filename:
            self.imu_filename = filename
        else:
            os.makedirs("data", exist_ok=True)
            self.imu_filename = f"data/{self.name}_{int(time.time())}.npy"
        print(f"Started recording IMU to {self.imu_filename}")
        self.imu_data_streamer = MemmapStreamer(
            self.imu_filename,
            max_records=3000000,
            shape=(7,),
            dtype="float64",
            flush_interval=1000,
        )

    async def stop_recording(self):
        self.is_recording = False
        if self.imu_data_streamer is not None:
            self.imu_data_streamer.close()
            self.imu_data_streamer = None
        print(f"Stopped recording IMU to {self.imu_filename}")

    async def disconnect(self):
        self._manual_disconnect = True
        self._cancel_reconnect_task()
        if self.client is None:
            self.connected = False
            return
        try:
            if self.client.is_connected:
                await self.client.stop_notify(NotifyProtocol.READ_CHARACTERISTIC)
        except Exception:
            pass
        try:
            if self.client.is_connected:
                await self.client.disconnect()
        finally:
            self.connected = False
        print(f"Disconnected from {self.address}")

    def get_tap_timestamp(self):
        return self.tap_timestamp

    def clear_tap_timestamp(self):
        self.tap_timestamp = None

    def get_release_timestamp(self):
        return self.release_timestamp

    def clear_release_timestamp(self):
        self.release_timestamp = None

    def update_audio_fps(self):
        self.audio_fps_count += 1
        if self.audio_fps_start_time is None:
            self.audio_fps_start_time = time.perf_counter()
            return

        elapsed = time.perf_counter() - self.audio_fps_start_time
        if elapsed >= (1.0 if not self.audio_fps else self.AUDIO_FPS_INTERVAL):
            self.audio_fps = self.audio_fps_count / elapsed
            self.audio_fps_count = 0
            self.audio_fps_start_time = time.perf_counter()
            if self.update_view_callback is not None:
                self.update_view_callback(self.address)

    def set_time_calibration(self, intercept: float, slope: float):
        self.time_calib_intercept = float(intercept)
        self.time_calib_slope = float(slope)
        self.is_time_calibrated = True
        if self.update_view_callback is not None:
            self.update_view_callback(self.address)

    def clear_time_calibration(self):
        self.time_calib_intercept = 0.0
        self.time_calib_slope = 0.0
        self.is_time_calibrated = False
        if self.update_view_callback is not None:
            self.update_view_callback(self.address)

    def to_host_timestamp(self, local_timestamp: float) -> float:
        if not self.is_time_calibrated or local_timestamp <= 0:
            return 0.0
        return self.time_calib_intercept + self.time_calib_slope * local_timestamp

async def connect():
    coroutines = []
    ring_addr = f"C6D638AF-DEE1-7F21-E5DC-EB31C271A3D7"
    ring = BLERing(
        ring_addr,
        index=0,
        imu_callback=lambda idx, imu: print(f"[Ring {idx}] {imu}"),
        audio_callback=lambda idx, t, audio, *_: print(
            f"[Ring {idx}] Audio type {t} length {len(audio)}"
        ),
    )
    coroutines.append(ring.connect())
    await asyncio.gather(*coroutines)

    # open mic
    await ring.open_audio()
    # wait for a while to receive audio data
    await asyncio.sleep(5)
    # close mic
    await ring.close_audio()


if __name__ == "__main__":
    asyncio.run(connect())
