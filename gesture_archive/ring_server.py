import socket
import struct
import asyncio
import time
import numpy as np
import selectors
import threading
import torch

from tsai.models.InceptionTime import InceptionTime

from core.ring.utils.imu_data import IMUData
from core.ring.ble_ring_v2 import BLERing
from core.utils.window import Window

from config import cfg
from utils.utils import Butterworth, MadgwickHelper
from constants import GESTURE_CODE_TO_NAME

import fire

class SocketServer:
    def __init__(
        self, host="localhost", port=12345, receive_callback=None, connect_callback=None
    ):
        self.host = host
        self.port = port
        self.receive_callback = receive_callback
        self.connect_callback = connect_callback
        self.sel = selectors.DefaultSelector()
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind((host, port))
        self.server_sock.listen()
        self.server_sock.setblocking(False)
        self.sel.register(self.server_sock, selectors.EVENT_READ, self.accept)
        self.clients = []

    def accept(self, sock):
        conn, addr = sock.accept()
        print(f"Accepted connection from {addr}")
        conn.setblocking(False)
        self.sel.register(conn, selectors.EVENT_READ, self.read)
        self.clients.append(conn)
        if self.connect_callback:
            self.connect_callback()

    def read(self, conn):
        try:
            data = conn.recv(1024)
            if data:
                data = data.decode()
                if self.receive_callback:
                    self.receive_callback(data)
            else:
                print(f"Closing connection to {conn.getpeername()}")
                self.sel.unregister(conn)
                self.clients.remove(conn)
                conn.close()
        except Exception as e:
            print(f"Error reading data from client: {e}")
            self.sel.unregister(conn)
            self.clients.remove(conn)
            conn.close()

    def broadcast(self, bytes: bytes):
        for client in self.clients:
            try:
                client.sendall(bytes)
            except Exception as e:
                print(f"Error sending bytes to client: {e}")
                try:
                    self.clients.remove(client)
                    self.sel.unregister(client)
                    client.close()
                except:
                    pass

    def start_server(self):
        print(f"Server started, listening on ({self.host}, {self.port})")
        while True:
            events = self.sel.select()
            for key, _ in events:
                try:
                    callback = key.data
                    callback(key.fileobj)
                except Exception as e:
                    print(f"Error in callback: {e}")


class GestureHandler:
    def __init__(
        self, modelnames, data_queue=None, result_queue=None, cmd_queue=None
    ) -> None:
        self.modelnames = modelnames
        self.device = cfg.device
        self.models = [
            InceptionTime(6, 39, nf=32).to(self.device) for modelname in modelnames
            # SimpleCNN(6, 26, 32, 0.1, 0.1).to(self.device) for modelname in modelnames
        ]
        # self.model = SimpleCNN(6, 26, 32, 0.1, 0.1).to(self.device)
        for i, modelname in enumerate(modelnames):
            self.models[i].load_state_dict(
                torch.load(modelname, map_location=self.device)["model"]
            )
            self.models[i].eval()
        self.ring_result_windows: list[Window[int]] = [
            Window(cfg.result_window_length) for _ in range(len(modelnames))
        ]  # store gesture result
        self.model_pointer = 0
        self.ring_last_touch_time = (
            0  # this time is updated when touch_xxx gesture is detected
        )
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.cmd_queue = cmd_queue

    def clear_all_result_windows(self):
        for window in self.ring_result_windows:
            window.clear()

    def _detect_gesture(self, data: np.ndarray):
        """
        Detect and push into result window.
        """
        start_time = time.time()
        for _ in range(len(self.models)):
            model = self.models[self.model_pointer]
            outputs = model(
                torch.from_numpy(data)
                .float()
                .view(1, 6, cfg.window_size)
                # .permute(0, 2, 1)
                .to(self.device)
            )
            # only keep allowed gestures in outputs
            outputs[:, [i for i in range(1, outputs.shape[1]) if i not in cfg.allowed_gestures]] = -1e9
            _, predictions = torch.max(outputs, 1)
            gesture = predictions[0].item()
            outputs = torch.nn.functional.softmax(outputs, dim=1)[0]
            self.ring_result_windows[self.model_pointer].push(gesture)
            self.model_pointer = (self.model_pointer + 1) % len(self.models)

    def _detect_gesture_with_window(self, data: np.ndarray) -> int:
        self._detect_gesture(data)
        for i in range(len(self.models)):
            if self.ring_result_windows[i].capacity() == 0:
                return -1
            gesture = self.ring_result_windows[i].last()
            if (
                self.ring_result_windows[i].count(lambda x: x == gesture)
                >= self.ring_result_windows[i].window_length * cfg.positive_ratio
            ):
                if (
                    gesture in cfg.allowed_gestures
                    and gesture in cfg.models_allowed_gestures[i]
                ):
                    self.clear_all_result_windows()
                    return gesture
        return -1

    def _check_if_skipped(self, gesture: int) -> bool:
        """
        skipped or touched in a short time.
        """
        if gesture in cfg.skipped_gestures:
            return True
        if GESTURE_CODE_TO_NAME[gesture].startswith("touch"):
            if time.time() - self.ring_last_touch_time < 0.2:
                self.ring_last_touch_time = time.time()
                return True
            self.ring_last_touch_time = time.time()
        if GESTURE_CODE_TO_NAME[gesture].startswith("pinch"):
            if time.time() - self.ring_last_touch_time < 0.2:
                return True
        return False

    def detect(self, data: np.ndarray) -> int:
        """
        Detect gesture from data. -1 for nothing, -2 for skipped
        """
        gesture = self._detect_gesture_with_window(data)
        if gesture >= 0:
            if self._check_if_skipped(gesture):
                self.clear_all_result_windows()
                return -2
        return gesture

    def run(self):
        while True:
            if self.data_queue.empty() and self.cmd_queue.empty():
                continue
            if not self.cmd_queue.empty():
                cmd = self.cmd_queue.get()
                if cmd == "clear":
                    self.clear_all_result_windows()
            if not self.data_queue.empty():
                data = self.data_queue.get()
                gesture = self.detect(data)
                self.result_queue.put(gesture)

class GestureHandlerCNN:
    def __init__(self, modelname: str) -> None:
        self.device = cfg.device
        self.model = SimpleCNN(6, 26, 32, 0.1, 0.1).to(self.device)
        self.model.load_state_dict(
            torch.load(modelname, map_location=self.device)
        )
        self.model.eval()

    def detect(self, data: np.ndarray) -> int:
        with torch.no_grad():
            outputs = self.model(
                torch.from_numpy(data)
                .float().view(1, 6, 200).permute(0, 2, 1)
                .to(self.device)
            )
            _, predictions = torch.max(outputs, 1)
            gesture = predictions[0].item()
            # if gesture != 0:
            #     print(f"Detected gesture: {[gesture]}")
            return gesture

class RingServer:
    def __init__(self, macs: list[str] = []):
        self.macs = macs
        self.macs_to_change = None
        self.initialize_rings()
        self.sk = SocketServer(
            host=cfg.server_ip,
            port=cfg.server_port,
            receive_callback=None,
            connect_callback=self.send_status,
        )
        self.enable_gesture = cfg.enable_gesture
        self.allowed_gestures = cfg.allowed_gestures
        self.filter = Butterworth(cfg.window_size, 10, mode="highpass", order=4)
        self.gesture_handler = (
            GestureHandler(cfg.models) if cfg.enable_gesture else None
        )
        # self.gesture_handler = GestureHandlerCNN("checkpoint/gesture_cnn.pt")
        self.selected_index = cfg.selected_index
        self.last_receive_imu_time = time.time()
        self.imu_data_num = 0
        self.madgwick_helper = MadgwickHelper(Dt=1 / cfg.window_size)
        self.start_move = False
        
        self.theta = 0.0
        self.auto_calibration_enabled = True  # Whether auto-calibration is enabled
        self.calibration_samples = []  # Store IMU samples that meet the criteria
        self.max_calibration_samples = 100  # Maximum number of samples
        self.calibration_interval = 30  # Calibration interval (seconds)
        self.last_calibration_time = 0  # Time of last calibration
        self.min_samples_for_calibration = 10  # Minimum samples required for calibration
        self.acc_x_threshold = 0.2  # Acceleration threshold in x direction
        self.angular_velocity_threshold = 0.3  # Angular velocity threshold

    def initialize_rings(self):
        self.rings: list[BLERing] = []
        self.ring_imu_data_windows: list[Window[IMUData]] = []  # store imu data

        for index, mac in enumerate(self.macs):
            self.rings.append(
                BLERing(
                    mac,
                    index=index,
                    imu_callback=self.imu_callback,
                    touch_callback=self.touch_callback,
                )
            )
            self.ring_imu_data_windows.append(Window(cfg.window_size))

    def send_status(self, index=0):
        # send ring info to all clients
        # be called when ring connected or client connected
        data_to_send = bytearray([0x44, 0x55, 0x00])
        for index, ring in enumerate(self.rings):
            if index == self.selected_index:
                data_to_send.extend(struct.pack("i", index))
                data_to_send.extend(struct.pack("i", ring.connected))
                break
        self.send_all(data_to_send)

    async def detect_and_send_gesture(self):
        while True:
            while True:
                ring_connected = False
                if len(self.rings) > 0:
                    for ring in self.rings:
                        if ring.index == self.selected_index and ring.connected:
                            ring_connected = True
                            break
                if ring_connected:
                    break
                await asyncio.sleep(0.5)
            start_time = time.time()
            for index, imu_window in enumerate(
                [self.ring_imu_data_windows[self.selected_index]]
            ):
                if imu_window.capacity() != imu_window.window_length:
                    continue
                data = imu_window.to_numpy_inside().T
                gesture = self.gesture_handler.detect(data)

                # special check for snap 16
                if gesture == 16:
                    if np.max(data) - np.min(data) < 25:
                        gesture = -2

                if gesture >= 0:
                    data_to_send = bytearray([0x44, 0x55, 0x88])
                    data_to_send.extend(struct.pack("i", index))
                    data_to_send.extend(struct.pack("i", gesture))
                    self.send_all(data_to_send)
                    imu_window.set_to_last_value()
                elif gesture == -2:
                    pass
                    imu_window.set_to_last_value()
                elif gesture == -1:
                    # print("No gesture detected")
                    pass
            await asyncio.sleep(0.001)

    async def connect_ring(self, ring):
        while True:
            try:
                await ring.connect(
                    callback=(
                        self.send_status if ring.index == self.selected_index else None
                    )
                )
            except Exception as e:
                print(f"Error connecting to ring: {e}")
                pass
            while ring.connected:
                await asyncio.sleep(0.1)
            # disconnected, send to all apps
            print("Disconnected, send to all apps")
            self.send_status()
            await asyncio.sleep(0.1)

    async def main_func(self):
        server_thread = threading.Thread(target=self.sk.start_server)
        server_thread.start()
        tasks = [self.connect_ring(ring) for ring in self.rings]
        tasks.append(self.detect_and_send_gesture())
        await asyncio.gather(*tasks)
        server_thread.join()

    def send_all(self, data: bytearray, pad_to=40):
        data = data + bytearray(pad_to - len(data))
        self.sk.broadcast(data)

    def imu_callback(self, index: int, data: IMUData):
        if index != self.selected_index:
            return

        # update data
        self.ring_imu_data_windows[index].push(data)
        self.madgwick_helper.update(data.to_numpy())

        if time.time() - self.last_receive_imu_time > 1:
            if cfg.enable_imu_fps:
                print(
                    "imu fps:",
                    self.imu_data_num / (time.time() - self.last_receive_imu_time),
                )
            self.imu_data_num = 0
            self.last_receive_imu_time = time.time()
        self.imu_data_num += 1

        data_to_send = bytearray([0x44, 0x55, 0x66])
        data_to_send.extend(struct.pack("i", index))
        data_to_send.extend(struct.pack("f", data.acc_x))
        data_to_send.extend(struct.pack("f", data.acc_y))
        data_to_send.extend(struct.pack("f", data.acc_z))
        data_to_send.extend(struct.pack("f", data.gyr_x))
        data_to_send.extend(struct.pack("f", data.gyr_y))
        data_to_send.extend(struct.pack("f", data.gyr_z))
        data_to_send.extend(struct.pack("d", data.timestamp))
        self.send_all(data_to_send)

        if self.ring_imu_data_windows[index].capacity() != cfg.window_size:
            return

        # send intensity
        data_to_send = bytearray([0x44, 0x55, 0x99])
        intensity = np.abs(
            self.filter(
                self.ring_imu_data_windows[index]
                .to_numpy_inside()
                .T[:3, cfg.window_size // 2 :]
            )
        ).mean()
        data_to_send.extend(struct.pack("i", index))
        data_to_send.extend(struct.pack("d", intensity))
        self.send_all(data_to_send)

        # send pose
        data_to_send = bytearray([0x44, 0x55, 0x11])
        data_to_send.extend(struct.pack("i", index))
        data_to_send.extend(struct.pack("f", self.madgwick_helper.get_orientation()[0]))
        data_to_send.extend(struct.pack("f", self.madgwick_helper.get_orientation()[1]))
        data_to_send.extend(struct.pack("f", self.madgwick_helper.get_orientation()[2]))
        data_to_send.extend(struct.pack("f", self.madgwick_helper.get_orientation()[3]))
        self.send_all(data_to_send)

    def touch_callback(self, index: int, action_code):
        if index != self.selected_index:
            return
        action_code = int(action_code)
        data_to_send = bytearray([0x44, 0x55, 0x77])
        data_to_send.extend(struct.pack("i", index))
        data_to_send.extend(struct.pack("i", action_code))
        self.send_all(data_to_send)

    def run(self):
        asyncio.run(self.main_func())
        
    def _transform_imu_data(self,x, y, z, gx, gy, gz, theta):
        """
        Convert six-axis sensor data from the rotated coordinate system back to the original coordinate system

        Parameters:
        x, y, z: Acceleration values in the rotated coordinate system
        gx, gy, gz: Angular velocity values in the rotated coordinate system
        theta: Rotation angle (in radians), positive rotation means rotation from the direction of the light towards the index finger direction to the middle finger direction

        Returns:
        x_orig, y_orig, z_orig: Acceleration values in the original coordinate system
        gx_orig, gy_orig, gz_orig: Angular velocity values in the original coordinate system
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x_orig = x
        gx_orig = gx
        
        y_orig = y * cos_theta - z * sin_theta
        z_orig = y * sin_theta + z * cos_theta
        
        gy_orig = gy * cos_theta - gz * sin_theta
        gz_orig = gy * sin_theta + gz * cos_theta
        
        return x_orig, y_orig, z_orig, gx_orig, gy_orig, gz_orig
    
    
    def collect_calibration_sample(self, data: IMUData):
        if not self.auto_calibration_enabled:
            return
            
 
        total_acc_magnitude = np.sqrt(data.acc_x**2 + data.acc_y**2 + data.acc_z**2)
        gravity_threshold_min = 9.5 
        gravity_threshold_max = 10.1 
        
        angular_velocity_magnitude = np.sqrt(
            data.gyr_x**2 + data.gyr_y**2 + data.gyr_z**2
        )
        
        if (abs(data.acc_x) < self.acc_x_threshold and 
            gravity_threshold_min < total_acc_magnitude < gravity_threshold_max and
            angular_velocity_magnitude < self.angular_velocity_threshold):
            

            self.calibration_samples.append({
                'acc_y': data.acc_y,
                'acc_z': data.acc_z,
                'timestamp': time.time()
            })
                
            if len(self.calibration_samples) > self.max_calibration_samples:
                self.calibration_samples.pop(0)

    
    def auto_calibrate(self):
        current_time = time.time()
        
        if (current_time - self.last_calibration_time < self.calibration_interval or
            len(self.calibration_samples) < self.min_samples_for_calibration):
            return
            

        total_y = 0
        total_z = 0
        valid_samples = 0
        
        
        recent_time = current_time - 300
        for sample in self.calibration_samples:
            if sample['timestamp'] >= recent_time:
                total_y += sample['acc_y']
                total_z += sample['acc_z']
                valid_samples += 1
        
        if valid_samples >= self.min_samples_for_calibration:
            avg_y = total_y / valid_samples
            avg_z = total_z / valid_samples
            
            new_theta = (np.degrees(np.arctan2(avg_y, avg_z)) + 180) % 360
            print(avg_y,avg_z,new_theta)
            if self.last_calibration_time == 0:
                self.theta = new_theta
            else:
                self.theta = 0.3 * new_theta + 0.7 * self.theta

            print(f"Auto calibration completed, theta updated to {self.theta:.2f} degrees (based on {valid_samples} samples)")
            self.last_calibration_time = current_time

def main(ring: str = None):
    print("start server")
    if ring != None:
        if isinstance(ring, str):
            cfg['ring_group'] = [ring]
        else:
            cfg['ring_group'] = list(ring)
    ring_group = RingServer(cfg.ring_group)
    ring_group.run()


if __name__ == "__main__":
    fire.Fire(main)
