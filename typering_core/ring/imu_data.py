from __future__ import annotations

import math
import numpy as np
from multipledispatch import dispatch

class IMUData():
    @dispatch(float, float, float, float, float, float, float, float)
    def __init__(self, acc_x:float, acc_y:float, acc_z:float, gyr_x:float,
            gyr_y:float, gyr_z:float, client_timestamp:float, server_timestamp:float):
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.gyr_x = gyr_x
        self.gyr_y = gyr_y
        self.gyr_z = gyr_z
        self.client_timestamp = client_timestamp
        self.server_timestamp = server_timestamp
        self.acc_np = np.array([self.acc_x, self.acc_y, self.acc_z])
        self.gyr_np = np.array([self.gyr_x, self.gyr_y, self.gyr_z])
        self.imu_np = np.concatenate((self.acc_np, self.gyr_np), axis=0)
        self.plane_directions = np.array([
            [-0.23709712, 0.20552186, -0.93578157],
            [-0.94889871, 0.04797022, 0.28491208],
            [-0.0617075, 0.95126743, 0.29782995],
            [ 0.99204434, -0.06982192, 0.00740044]
        ])

    @dispatch(dict)
    def __init__(self, data:dict):
        self.__init__(data['acc_x'], data['acc_y'], data['acc_z'], data['gyr_x'],
            data['gyr_y'], data['gyr_z'], data['client_timestamp'], data['server_timestamp'])
    
    def __getitem__(self, index):
        return [self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z][index]

    def __str__(self):
        return f'Acc [x: {self.acc_x:.3f}  y: {self.acc_y:.3f}  z: {self.acc_z:.3f}]' \
            + f'Gyr[x: {self.gyr_x:.3f}  y: {self.gyr_y:.3f}  z: {self.gyr_z:.3f}] ' \
            + f'Client Timestamp: {self.client_timestamp:.6f} Server Timestamp: {self.server_timestamp:.6f}'

    @property
    def gyr_norm(self):
        return math.sqrt(self.gyr_x * self.gyr_x + self.gyr_y * self.gyr_y + self.gyr_z * self.gyr_z)
    
    @property
    def acc_norm(self):
        return math.sqrt(self.acc_x * self.acc_x + self.acc_y * self.acc_y + self.acc_z * self.acc_z)

    def scale(self) -> IMUData:
        return IMUData(self.acc_x / 9.8, -self.acc_y / 9.8, -self.acc_z / 9.8, self.gyr_x / math.pi * 180,
            -self.gyr_y / math.pi * 180, -self.gyr_z / math.pi * 180, self.client_timestamp, self.server_timestamp)
                    
    def direction(self) -> int: # TODO: -> str
        for i in range(len(self.plane_directions)):
            if np.dot(self.acc_np / self.acc_norm, self.plane_directions[i]) >= math.sqrt(2) / 2:
                return i
        return -1
  
    def to_numpy(self):
        return np.array([self.acc_x, self.acc_y, self.acc_z,
            self.gyr_x, self.gyr_y, self.gyr_z], dtype=np.float32)
    
    def to_numpy_with_timestamp(self):
        return np.array([self.acc_x, self.acc_y, self.acc_z, self.gyr_x,
            self.gyr_y, self.gyr_z, self.client_timestamp, self.server_timestamp], dtype=np.float64)
    
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
    

class IMUDataGroup():
    
    def __init__(self, data:list[IMUData]):
        self.data = data

    def __getitem__(self, index) -> IMUData:
        return self.data[index]

    def __str__(self):
        return '\n'.join(map(str, self.data)) + '\n'

    def scale(self) -> IMUDataGroup:
        return IMUDataGroup([x.scale() for x in self.data])

    def to_numpy(self):
        return np.array([x[j] for x in self.data for j in range(6)])

    @property
    def timestamp(self):
        return self.data[0].timestamp
