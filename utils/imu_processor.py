import numpy as np
from ahrs import Quaternion
from ahrs.common.orientation import acc2q
from ahrs.filters import Madgwick
from vqf import PyVQF as VQF


class IMUProcessor:
    def __init__(self, frequency=200.0, core="VQF"):
        """
        初始化IMU处理器。
        :param frequency: 采样频率，默认为60 Hz。
        """
        self.frequency = frequency
        self.dt = 1.0 / frequency  # 采样时间间隔
        self.madgwick = Madgwick(frequency=frequency)  # 初始化 Madgwick 滤波器
        self.vqf = VQF(self.dt)  # 初始化 VQF 滤波器
        self.gravity = np.array([0.0, 0.0, 9.81])  # 全局坐标系中的重力向量 (m/s²)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 初始化四元数
        self.angle = None
        self.pre_timestamp = 0
        assert core in ["Madgwick", "VQF"]
        self.core = core

    def init_quaternion_and_timestamp(self, accel_data, timestamp):
        """
        初始化四元数和上一次读取的时间戳。
        :param accel_data: 加速度计数据，格式为[x, y, z]。
        :param timestamp: 上一次读取的时间戳，单位为毫秒(ms)。
        :return: None
        """
        self.quaternion = Quaternion(acc2q(accel_data).astype(np.float64))
        self.angle = self.quaternion.to_angles()
        self.pre_timestamp = timestamp

    def update_attitude_by_data(self, gyro_data, accel_data, timestamp=None, quaternion=None):
        """
        使用滤波器更新姿态估计。
        :param quaternion: 当前姿态的四元数 [w, x, y, z]
        :param gyro_data: 陀螺仪数据 [GX, GY, GZ] (单位: rad/s)
        :param accel_data: 加速度计数据 [AX, AY, AZ] (单位: m/s²)
        :param timestamp: 时间戳,单位ms
        :return: 更新后的四元数 [w, x, y, z]
        """
        # print("*************** update_attitude ***************")
        # 将陀螺仪数据从 °/s 转换为 rad/s
        # 计算时间间隔
        if timestamp is None:
            dt = self.dt
        else:
            dt = (timestamp - self.pre_timestamp) / 1000
            self.pre_timestamp = timestamp

        # 更新滤波器
        if self.core == "Madgwick":
            q = self.madgwick.updateIMU(
                q=quaternion if quaternion is not None else self.quaternion,
                gyr=gyro_data,  # 陀螺仪数据 (rad/s)
                acc=accel_data,  # 加速度计数据 (m/s²)
                dt=dt  # 采样时间间隔
            )
        else:
            self.vqf.update(
                gyr=gyro_data,  # 陀螺仪数据 (rad/s)
                acc=accel_data,  # 加速度计数据 (m/s²)
            )
            q = self.vqf.getQuat6D()

        self.quaternion = Quaternion(q)
        self.angle = self.quaternion.to_angles()
        return q

    def calculate_linear_acceleration(self, accel_data, quaternion=None):
        """
        计算线性加速度（去除重力分量）。
        :param accel_data: 原始加速度数据 (含重力) [AX, AY, AZ] (单位: m/s²)
        :param quaternion: 当前姿态的四元数 [w, x, y, z]
        :return: 线性加速度 [aX_linear, aY_linear, aZ_linear] (单位: m/s²)
        """
        # print("*************************** calculate_linear_acceleration ***************************")
        if quaternion is None:
            quaternion = self.quaternion
        gravity_imu = Quaternion(quaternion.conjugate).rotate(self.gravity)  # 重力在局部坐标系的投影
        linear_acc = accel_data - gravity_imu  # 去除重力分量
        return linear_acc
