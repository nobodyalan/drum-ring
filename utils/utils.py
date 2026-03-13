import numpy as np
import pyquaternion as pyq
import numpy as np

from scipy import signal, ndimage
from typing import Union, Tuple
from scipy import signal
from ahrs.filters import Madgwick
from collections import deque
from threading import Timer
from copy import deepcopy

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

class Butterworth:

    def __init__(
        self,
        fs: float,
        cut: Union[float, Tuple[float, float]],
        mode: str = "lowpass",
        order: int = 4,
    ) -> None:
        """Init the parameters of the Butterworth filter.
        args:
            fs: float, the sampling frequency in Hz.
            cut: Union[float,Tuple[float,float]], the cut-off frequencies of the
                transition band. Use low and high cut-off frequencies or both of
                them depending on the filter mode.
            mode: str, in {'lowpass', 'highpass', 'bandpass', 'bandstop'}.
                Default is 'lowpass'.
            order: int, the order of the Butterworth filter.
        """
        nyq = 0.5 * fs
        if mode == "lowpass" or mode == "highpass":
            self.sos = signal.butter(order, cut / nyq, btype=mode, output="sos")
        elif mode == "bandpass" or mode == "bandstop":
            self.sos = signal.butter(
                order, [cut[0] / nyq, cut[1] / nyq], btype=mode, output="sos"
            )
        else:
            raise Exception(f"Wrong filter mode: {mode}.")

    def __call__(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Filter signals.
        args:
            data: np.ndarray, of any shape and any dtype.
            axis: int, the axis to perform 1d filtering. Default is -1.
        returns:
            np.ndarray, with the same shape as data, the filtered signals.
        """
        return signal.sosfiltfilt(self.sos, data, axis=axis)


class ButterworthRealTime:
    def __init__(self, sampling_rate, cutoff_frequency, order=4):
        """
        初始化实时低通滤波器
        
        :param sampling_rate: 采样率 (Hz)
        :param cutoff_frequency: 截止频率 (Hz)
        :param order: 滤波器阶数 (默认4阶)
        """
        # 计算奈奎斯特频率并标准化截止频率
        nyquist = 0.5 * sampling_rate
        normalized_cutoff = cutoff_frequency / nyquist
        
        # 获取巴特沃斯低通滤波器的系数
        self.b, self.a = signal.butter(order, normalized_cutoff, btype='low')
        
        # 获取滤波器的初始状态
        self.zi = signal.lfilter_zi(self.b, self.a)
    
    def filter_data(self, new_data_point):
        """
        对新的数据点进行滤波
        
        :param new_data_point: 新的IMU数据点
        :return: 滤波后的数据点
        """
        # 使用lfilter进行滤波
        filtered_point, self.zi = signal.lfilter(self.b, self.a, [new_data_point], zi=self.zi)
        return filtered_point[0]
    
    def filter_data_list(self, new_data_list):
        """
        对新的数据点序列进行滤波
        
        :param new_data_list: 新的IMU数据点
        :return: 滤波后的数据点
        """
        filtered_point_list = []
        for new_data_point in new_data_list:
            filtered_point_list.append(self.filter_data(new_data_point))
        return filtered_point_list


class MovingAverage:

    def __init__(self, window: int) -> None:
        """Init the parameters of the moving average filter.
        args:
            window: int, the length of the moving average window.
        """
        self.window = window

    def __call__(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Filter signals.
        args:
            data: np.ndarray, of any shape and any dtype.
            axis: int, the axis to perform 1d filtering. Default is -1.
        returns:
            np.ndarray, with the same shape as data, the filtered signals.
        """
        return ndimage.uniform_filter1d(data, self.window, axis=axis)


class IntegralValue:
    def __init__(self, value=0):
        self.value = value
        self.last_input_queue = deque(maxlen=10)
        for _ in range(10):
            self.last_input_queue.append(0)
        self.cur_mean = 0

    def update(self, new_input, dt):
        self.cur_mean = (self.cur_mean * 10 - self.last_input_queue[0] + new_input) / 10
        self.last_input_queue.append(new_input)
        self.value += self.cur_mean * dt
        return self.value

    def reset_value(self):
        self.value = 0


class MadgwickHelper:
    def __init__(
        self,
        q0: np.ndarray = np.array([1, 0, 0, 0]),
        Dt=1 / 200,
        auto_start_calibration=True,
    ) -> None:
        self.madgwick_filter = Madgwick(Dt=Dt) if not auto_start_calibration else None
        self.start_orientation_queue = deque(maxlen=100)
        self.Dt = Dt
        self.orientation = q0
        self.prev_orientation = q0
        self.delta_orientation = np.array([0, 0, 0, 0])
        self.data = None
        self.data_without_gravity = None
        self.data_without_gravity_world = None
        self.vx, self.vy, self.vz = IntegralValue(), IntegralValue(), IntegralValue()
        self.zero_velocity_updater = self.ZeroVelocityUpdater()

    class ZeroVelocityUpdater:
        SAMPLES_THRESHOLD = 10
        ACC_THRESHOLD = 1
        ANGU_THRESHOLD = 0.15

        def __init__(self) -> None:
            self.zero_sample_cnt = 0

        def update(self, data) -> bool:
            """
            data: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            """

            if np.all(np.abs(data[:3]) <= self.ACC_THRESHOLD) and np.all(
                np.abs(data[3:]) <= self.ANGU_THRESHOLD
            ):
                self.zero_sample_cnt += 1
            else:
                self.zero_sample_cnt = 0
            return self.zero_sample_cnt >= self.SAMPLES_THRESHOLD

    def update(self, data: np.ndarray) -> np.ndarray:
        """
        data: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            unit: [m/s^2, m/s^2, m/s^2, rad/s, rad/s, rad/s]
        """
        self.data = data
        self.data_without_gravity = self.cal_data_without_gravity()
        # filter
        # self.data_without_gravity = median_filter(self.data_without_gravity, size=3)
        self.data_without_gravity_world = (
            self.cal_data_without_gravity_world()
        )  # with acc in world coordinate

        if not self.madgwick_filter:
            if len(self.start_orientation_queue) < self.start_orientation_queue.maxlen:
                self.start_orientation_queue.append(data)
                return self.orientation
            accumulated_imu_data = np.array(self.start_orientation_queue)
            self.madgwick_filter = Madgwick(
                Dt=self.Dt,
                acc=accumulated_imu_data[:, :3],
                gyr=accumulated_imu_data[:, 3:],
            )
            self.orientation = self.madgwick_filter.Q[-1]
            return self.orientation

        self.prev_orientation = deepcopy(self.orientation)

        self.orientation = self.madgwick_filter.updateIMU(
            self.orientation,
            gyr=[data[3], data[4], data[5]],
            acc=[data[0], data[1], data[2]],
        )

        self.delta_orientation = (
            pyq.Quaternion(self.orientation).inverse
            * pyq.Quaternion(self.prev_orientation)
        ).elements

        if not self.zero_velocity_updater.update(self.data_without_gravity_world):
            self.vx.update(self.data_without_gravity_world[0], self.Dt)
            self.vy.update(self.data_without_gravity_world[1], self.Dt)
            self.vz.update(self.data_without_gravity_world[2], self.Dt)
        else:
            self.vx.reset_value()
            self.vy.reset_value()
            self.vz.reset_value()

        return self.orientation

    def get_orientation(self) -> np.ndarray:
        return self.orientation

    def get_orientation_as_pyq(self) -> pyq.Quaternion:
        return pyq.Quaternion(
            self.orientation[0],
            self.orientation[1],
            self.orientation[2],
            self.orientation[3],
        )

    def get_orientation_as_euler(self) -> np.ndarray:
        return np.array(self.get_orientation_as_pyq().yaw_pitch_roll)

    def get_orientation_as_coords(self) -> np.ndarray:
        return np.array(self.get_orientation_as_pyq().rotate([1, 0, 0]))

    def get_delta_orientation_as_pyq(self) -> pyq.Quaternion:
        return pyq.Quaternion(
            self.delta_orientation[0],
            self.delta_orientation[1],
            self.delta_orientation[2],
            self.delta_orientation[3],
        )

    def get_delta_orientation_as_euler(self) -> np.ndarray:
        return np.array(self.get_delta_orientation_as_pyq().yaw_pitch_roll)

    def get_delta_orientation_as_coords(self) -> np.ndarray:
        return np.array(
            self.get_delta_orientation_as_pyq().rotate([1, 0, 0])
        ) - np.array([1, 0, 0])

    def cal_data_without_gravity(self) -> np.ndarray:
        GRAVITY = 9.76
        g = []
        g.append(
            2
            * (
                self.orientation[1] * self.orientation[3]
                - self.orientation[0] * self.orientation[2]
            )
            * GRAVITY
        )
        g.append(
            2
            * (
                self.orientation[2] * self.orientation[3]
                + self.orientation[0] * self.orientation[1]
            )
            * GRAVITY
        )
        g.append(
            (
                self.orientation[0] ** 2
                - self.orientation[1] ** 2
                - self.orientation[2] ** 2
                + self.orientation[3] ** 2
            )
            * GRAVITY
        )
        for _ in range(3):
            g.append(0)
        return self.data - g

    def cal_data_without_gravity_world(self) -> np.ndarray:
        # rotate data without gravity to world coordinate
        world_acc = self.get_orientation_as_pyq().rotate(self.data_without_gravity[:3])
        world_data = np.concatenate([world_acc, self.data_without_gravity[3:]])
        return world_data

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx.value, self.vy.value, self.vz.value])


if __name__ == "__main__":
    pass
