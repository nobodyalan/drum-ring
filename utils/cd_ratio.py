import time
import numpy as np

import math
import time


class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq  # Hz
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.last_time = None
        self.x_prev = None
        self.dx_prev = 0.0

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if t is None:
            t = time.perf_counter()

        if self.last_time is None:
            self.last_time = t
            self.x_prev = x
            return x

        dt = t - self.last_time
        if dt <= 0: dt = 1.0 / self.freq
        self.freq = 1.0 / dt
        self.last_time = t

        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)

        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


class CDRatioEstimator:
    def __init__(self,
                 min_cd=0.5,
                 max_cd=2,
                 v0=15,
                 kv=0.2,
                 c=2,
                 ka=0.003,
                 euro_freq=200,
                 euro_min_cutoff=1.0,
                 euro_beta=0.02):
        self.min_cd = min_cd
        self.max_cd = max_cd
        self.v0 = v0
        self.kv = kv
        self.c = c
        self.ka = ka

        self.filter = OneEuroFilter(freq=euro_freq,
                                    min_cutoff=euro_min_cutoff,
                                    beta=euro_beta)

        self.prev_v = None
        self.prev_t = None

    def update(self, v_raw, t=None):
        """
        输入当前控制器速度（单位 px/s），输出当前 CD ratio。
        """
        if t is None:
            t = time.perf_counter()

        v_filtered = self.filter.filter(v_raw, t)

        if self.prev_v is None:
            self.prev_v = v_filtered
            self.prev_t = t
            return self.min_cd

        dt = t - self.prev_t
        if dt <= 0: dt = 1e-3  # 避免除 0

        acc = (v_filtered - self.prev_v) / dt

        # print(f"v_filtered: {v_filtered}, acc: {acc}, dt: {dt}")
        # sigmoid based on speed
        sigmoid = 1.0 / (1.0 + math.exp(-self.kv * (v_filtered - self.v0)))
        accel_term = self.c * math.tanh(self.ka * acc)
        # print(f"sigmoid: {sigmoid}, accel_term: {accel_term}")
        cd = self.min_cd + (self.max_cd - self.min_cd) * sigmoid + accel_term

        # 更新状态
        self.prev_v = v_filtered
        self.prev_t = t

        return cd


estimator = CDRatioEstimator()


if __name__ == "__main__":
    # 模拟一段速度输入
    for v in [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 4.0, 2.0]:
        cd = estimator.update(v)
        print(f"Speed: {v:.1f} px/s => CD Ratio: {cd:.2f}")
        time.sleep(0.01)
