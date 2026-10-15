# 17 通道特征生成说明

本文档说明鼓击手势模型（`rawlin12_handcrafted` 特征模式）如何从原始戒指 IMU 数据一步步生成 **17 通道** 的特征。

## 总览

```
原始 IMU (T, 8) @200Hz
   │
   ▼ ① 重力补偿 (四元数互补滤波)
线性加速度 linear_acc (T, 3)
   │
   ▼ ② 拼接基础通道
[raw_acc×3, raw_gyro×3, linear_acc×3]   = 9 通道
   │
   ▼ ③ 手工特征
[norm×3, delta_norm×3, dot, cos]        = 8 通道
   │
   ▼ 拼接
17 通道 (T, 17)
   │
   ▼ ④ 滑窗 + 标准化
模型输入 (1, 17, 15)
```

| 阶段 | 产物 | 通道数 |
|---|---|---|
| 原始 IMU | acc + gyro 六轴 | 6 |
| 重力补偿 | linear_acc | +3 |
| 基础通道 | raw_acc + raw_gyro + linear_acc | 9 |
| 手工特征 | 模长/差分/夹角 | +8 |
| **合计** | | **17** |

---

## 第 0 步：原始 IMU 数据格式

戒指 IMU 是 `.npy.enc` 加密的 NumPy 数组，解密后为 `(T, 8)`：

| 列 | 内容 |
|---|---|
| `[:, 0]` | 时间戳（计数器，可能全 0） |
| `[:, 1]` | 时间戳（秒） |
| `[:, 2:5]` | 加速度计 `acc = [ax, ay, az]`，单位 m/s² |
| `[:, 5:8]` | 陀螺仪 `gyro = [gx, gy, gz]`，单位 rad/s |

采样率 200Hz（`dt = 1/200 = 5ms`）。代码取 `raw6 = imu[:, 2:8]`（加速度 3 轴 + 陀螺仪 3 轴）。

---

## 第 1 步：重力补偿 → 线性加速度

使用 `OnlineLinearAccelerationEstimator`（`models/drum_gesture_recognizer.py`），一个**四元数互补滤波器**（不是 Kalman——旧的包 README 里写 Kalman 是笔误）：

### 初始化
假设静止时加速度方向即重力方向，用加速度方向估计初始姿态：

```
orientation = quat_between(acc / |acc|, [0, 0, 1])
```

### 每帧更新
```
1. 陀螺仪积分:  orientation = normalize(orientation ⊗ rotvec(gyro · dt))
2. 加速度校正:  计算当前姿态估计的重力方向与 [0,0,1] 的旋转差 correction
               trust = 0.025 · exp(-((|acc| - 9.81) / 2.5)²)
               orientation = normalize(rotvec(correction · trust) ⊗ orientation)
```

`trust` 自适应：当 `|acc|` 接近 9.81（静止或匀速）时 trust 高、更信加速度；运动中 trust 低、主要靠陀螺仪积分。

### 去重力
```
gravity_sensor = rotate(orientation⁻¹, [0,0,1] · 9.81)   # 传感器坐标系下的重力
linear_acc     = acc - gravity_sensor                     # 世界/传感器系下的线性加速度
```

完整实现（`models/drum_gesture_recognizer.py`）：

```python
# ── 四元数工具函数 ──
def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = angle * 0.5
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    inv = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return _quat_multiply(_quat_multiply(q, qv), inv)[1:]


def _quat_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-8 or target_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    s = source / source_norm
    t = target / target_norm
    dot = float(np.clip(np.dot(s, t), -1.0, 1.0))
    if dot < -1.0 + 1e-8:
        axis = np.cross(s, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(s, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    quat = np.array([1.0 + dot, *np.cross(s, t)], dtype=np.float64)
    return _normalize_quat(quat)


def _rotvec_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    quat = _quat_between(source, target)
    quat = _normalize_quat(quat)
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return quat[1:] / sin_half * angle


class OnlineLinearAccelerationEstimator:
    """Lightweight gyro+acc complementary gravity compensation used by the drum app."""

    def __init__(self) -> None:
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.initialized = False
        self.world_gravity_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def reset(self) -> None:
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.initialized = False

    def update(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        acc = np.asarray(acc, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0 or dt > 0.1:
            dt = 1.0 / 200.0

        acc_norm = float(np.linalg.norm(acc))
        if not self.initialized:
            if acc_norm > 1e-6:
                self.orientation = _quat_between(acc / acc_norm, self.world_gravity_dir)
            self.initialized = True
        else:
            self.orientation = _normalize_quat(
                _quat_multiply(self.orientation, _quat_from_rotvec(gyro * dt))
            )
            if acc_norm > 1e-6:
                est_gravity_world = _quat_rotate(self.orientation, acc / acc_norm)
                correction = _rotvec_between(est_gravity_world, self.world_gravity_dir)
                trust = 0.025 * np.exp(-((acc_norm - GRAVITY) / 2.5) ** 2)
                self.orientation = _normalize_quat(
                    _quat_multiply(_quat_from_rotvec(correction * trust), self.orientation)
                )

        inv = np.array(
            [self.orientation[0], -self.orientation[1], -self.orientation[2], -self.orientation[3]],
            dtype=np.float64,
        )
        gravity_sensor = _quat_rotate(inv, self.world_gravity_dir * GRAVITY)
        linear_acc = acc - gravity_sensor
        return linear_acc.astype(np.float32), gravity_sensor.astype(np.float32)
```

---

## 第 2 步：基础 9 通道

直接拼接原始六轴 + 去重力线性加速度：

```
base = [raw_acc(3), raw_gyro(3), linear_acc(3)]   # 9 通道
```

注意：**gyro 不重复**。这是当前 17 通道版本与旧版 21 通道的关键区别（旧版把 gyro 重复了一遍）。

训练侧拼接实现（`scripts/train_local_drum_gestures.py`）：

```python
def feature_recording(raw_imu: np.ndarray, linear_imu: np.ndarray, feature_mode: str) -> np.ndarray:
    raw6 = raw_imu[:, 2:8].astype(np.float32)
    if feature_mode == "lin6":
        return linear_imu.astype(np.float32)
    if feature_mode == "rawlin12":
        # 9 通道: raw_acc(3) + raw_gyro(3) + linear_acc(3), 不重复 gyro
        return np.concatenate([raw6, linear_imu[:, :3]], axis=1).astype(np.float32)
    if feature_mode == "rawlin12_handcrafted":
        # 基础 9 通道: raw_acc(3) + raw_gyro(3) + linear_acc(3), 不重复 gyro
        base = np.concatenate([raw6, linear_imu[:, :3]], axis=1).astype(np.float32)
        return np.concatenate([base, handcrafted_channels(raw6, linear_imu[:, :3])], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")
```

---

## 第 3 步：手工特征 8 通道

由基础通道派生，全部是标量（每帧 1 个值）。辅助函数：

```python
_norm(v)  = ||v||                            # L2 模长 (axis=1)
_delta(v) = np.diff(v, prepend=首帧)          # 帧间差分, 首帧差分补 0
```

| 通道 | 公式 | 物理含义 |
|---|---|---|
| `raw_acc_norm` | $\|a_{raw}\|$ | 原始加速度模长 |
| `linear_acc_norm` | $\|a_{lin}\|$ | 线性加速度模长（**最关键的击打特征**） |
| `raw_gyro_norm` | $\|\omega\|$ | 角速度模长 |
| `raw_acc_delta_norm` | $\|\Delta a_{raw}\|$ | 加速度帧间差分（急动度） |
| `linear_acc_delta_norm` | $\|\Delta a_{lin}\|$ | 线性加速度帧间差分 |
| `gyro_delta_norm` | $\|\Delta \omega\|$ | 角速度帧间差分 |
| `raw_linear_dot` | $a_{raw} \cdot a_{lin}$ | 原始/线性加速度内积 |
| `raw_linear_cos` | $\dfrac{a_{raw} \cdot a_{lin}}{\|a_{raw}\| \cdot \|a_{lin}\|}$ | 两者夹角余弦（分母加 1e-6 防除零） |

实现（`scripts/train_local_drum_gestures.py`）：

```python
def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)


def _delta(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=0, prepend=values[:1]).astype(np.float32)


def handcrafted_channels(raw6: np.ndarray, linear_acc3: np.ndarray) -> np.ndarray:
    """从 raw6 (acc×3+gyro×3) + linear_acc3 (去重力加速度×3) 构建 8 个手工特征。"""
    raw_acc = raw6[:, :3]
    raw_gyro = raw6[:, 3:6]
    linear_acc = linear_acc3[:, :3]

    raw_acc_norm = _norm(raw_acc)
    linear_acc_norm = _norm(linear_acc)
    raw_gyro_norm = _norm(raw_gyro)
    raw_acc_delta_norm = _norm(_delta(raw_acc))
    linear_acc_delta_norm = _norm(_delta(linear_acc))
    gyro_delta_norm = _norm(_delta(raw_gyro))

    raw_linear_dot = np.sum(raw_acc * linear_acc, axis=1, keepdims=True).astype(np.float32)
    denom = np.maximum(raw_acc_norm * linear_acc_norm, 1e-6)
    raw_linear_cos = (raw_linear_dot / denom).astype(np.float32)

    return np.concatenate(
        [
            raw_acc_norm,
            linear_acc_norm,
            raw_gyro_norm,
            raw_acc_delta_norm,
            linear_acc_delta_norm,
            gyro_delta_norm,
            raw_linear_dot,
            raw_linear_cos,
        ],
        axis=1,
    ).astype(np.float32)
```

---

## 17 通道完整表

| # (1-based) | 通道名 | 类别 |
|---|---|---|
| 1 | `raw_acc_x` | 基础 |
| 2 | `raw_acc_y` | 基础 |
| 3 | `raw_acc_z` | 基础 |
| 4 | `raw_gyro_x` | 基础 |
| 5 | `raw_gyro_y` | 基础 |
| 6 | `raw_gyro_z` | 基础 |
| 7 | `linear_acc_x` | 基础 |
| 8 | `linear_acc_y` | 基础 |
| 9 | `linear_acc_z` | 基础 |
| 10 | `raw_acc_norm` | 手工 |
| 11 | `linear_acc_norm` | 手工 |
| 12 | `raw_gyro_norm` | 手工 |
| 13 | `raw_acc_delta_norm` | 手工 |
| 14 | `linear_acc_delta_norm` | 手工 |
| 15 | `gyro_delta_norm` | 手工 |
| 16 | `raw_linear_dot` | 手工 |
| 17 | `raw_linear_cos` | 手工 |

拼接顺序即上表顺序：`[基础 9] + [手工 8]`。

推理侧单窗口构建（`models/drum_gesture_recognizer.py`，输入 `(15, 9)` 输出 `(15, 17)`）：

```python
def build_rawlin12_handcrafted_window(
    raw_linear_window: np.ndarray,
    window_size: int | None = None,
) -> np.ndarray:
    """从 9 通道原始窗口 [raw_acc×3, raw_gyro×3, linear_acc×3] 构建 17 通道特征窗口。"""
    values = np.asarray(raw_linear_window, dtype=np.float32)
    expected_window = int(window_size or WINDOW_SIZE)
    if values.shape != (expected_window, 9):
        raise ValueError(f"输入窗口必须为 ({expected_window}, 9)，收到: {values.shape}")
    raw_acc = values[:, 0:3]
    raw_gyro = values[:, 3:6]
    linear_acc = values[:, 6:9]

    raw_acc_norm = _norm(raw_acc)
    linear_acc_norm = _norm(linear_acc)
    raw_gyro_norm = _norm(raw_gyro)
    raw_acc_delta_norm = _norm(_delta(raw_acc))
    linear_acc_delta_norm = _norm(_delta(linear_acc))
    gyro_delta_norm = _norm(_delta(raw_gyro))
    raw_linear_dot = np.sum(raw_acc * linear_acc, axis=1, keepdims=True).astype(np.float32)
    raw_linear_cos = (raw_linear_dot / np.maximum(raw_acc_norm * linear_acc_norm, 1e-6)).astype(
        np.float32
    )
    return np.concatenate(
        [
            raw_acc,
            raw_gyro,
            linear_acc,
            raw_acc_norm,
            linear_acc_norm,
            raw_gyro_norm,
            raw_acc_delta_norm,
            linear_acc_delta_norm,
            gyro_delta_norm,
            raw_linear_dot,
            raw_linear_cos,
        ],
        axis=1,
    ).astype(np.float32)
```

---

## 第 4 步：滑窗 + 标准化

### 滑窗
`WINDOW_SIZE = 15` 帧 @ 200Hz = **75ms**。模型输入形状 `(1, 17, 15)`（通道 × 时间帧）。

### 标准化（逐通道 z-score）
原始 6 轴（通道 1-9）用固定均值/标准差：

| 通道 | 均值 | 标准差 |
|---|---|---|
| `raw_acc` (×3) | -1.5374 | 15.2859 |
| `raw_gyro` (×3) | -0.2267 | 3.0157 |
| `linear_acc` (×3) | 1.3655 | 15.4273 |

手工 8 通道（通道 10-17）：

| 通道 | 均值 | 标准差 |
|---|---|---|
| `raw_acc_norm` | 12.6973 | 23.3847 |
| `linear_acc_norm` | 15.3177 | 22.0220 |
| `raw_gyro_norm` | 4.0217 | 3.3561 |
| `raw_acc_delta_norm` | 4.6444 | 17.8993 |
| `linear_acc_delta_norm` | 4.5870 | 17.9035 |
| `gyro_delta_norm` | 0.7715 | 2.0652 |
| `raw_linear_dot` | 665.7154 | 3529.4480 |
| `raw_linear_cos` | 0.3612 | 0.6688 |

这些统计量在 `models/drum_gesture_recognizer.py` 顶部（`RAW_ACC_MEAN`、`RAW_GYRO_STD`、`HANDCRAFTED_STATS` 等），训练时由 `scripts/train_local_drum_gestures.py` 计算，推理时硬编码使用。

---

## 代码位置索引

| 功能 | 文件 | 函数 |
|---|---|---|
| 重力补偿（互补滤波） | `models/drum_gesture_recognizer.py` | `OnlineLinearAccelerationEstimator` |
| 推理特征构建（17ch） | `models/drum_gesture_recognizer.py` | `build_rawlin12_handcrafted_window()` |
| 标准化 | `models/drum_gesture_recognizer.py` | `DrumGestureRecognizer.prepare()` |
| 训练特征构建（17ch） | `scripts/train_local_drum_gestures.py` | `feature_recording()` + `handcrafted_channels()` |
| 训练线性加速度估计 | `scripts/train_local_drum_gestures.py` | `estimate_linear_recording()` |

训练与推理两套实现输出**完全一致**的 17 通道，区别仅在批量处理方式（训练一次性整段算，推理逐帧滑窗算）。

---

## 常见问题

**Q: 为什么 CLAUDE.md / 旧文档写的是 21 通道？**

旧版特征模式为 `raw_acc(3) + raw_gyro(3) + linear_acc(3) + gyro(3)` = 12 基础 + 9 手工 = 21 通道，其中 gyro 被重复了一遍，且手工特征多一个 `gyro_norm`。当前代码已改为 9 基础 + 8 手工 = **17 通道**（去掉了重复 gyro 和冗余的 gyro_norm）。

模型代码 `_infer_input_channels()` 仍兼容旧 21 通道 checkpoint：会自动把 17 通道格式扩展回 21 通道（插入重复 gyro 和 `gyro_norm`）后再推理。
