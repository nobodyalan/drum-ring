from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tsai.models.InceptionTime import InceptionTime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoint"
    / "drum_gesture"
    / "my_v6.pth"
)

WINDOW_SIZE = 15
PRE_PEAK_FRAMES = 12
POST_PEAK_FRAMES = 2
GRAVITY = 9.81

DRUM_CLASS_IDS = tuple(range(6))
DRUM_CLASS_LABELS = {
    0: "空",
    1: "踩镲",
    2: "通鼓2",
    3: "军鼓",
    4: "通鼓1",
    5: "吊镲",
}

FEATURE_MODE = "rawlin12_handcrafted"
FEATURE_CHANNELS = 17
RAW_ACC_MEAN = -1.5373504161834717
RAW_ACC_STD = 15.285881996154785
RAW_GYRO_MEAN = -0.22666260600090027
RAW_GYRO_STD = 3.0157175064086914
LINEAR_ACC_MEAN = 1.3655122518539429
LINEAR_ACC_STD = 15.427335739135742
HANDCRAFTED_STATS = {
    9:  ("raw_acc_norm", 12.697272300720215, 23.384702682495117),
    10: ("linear_acc_norm", 15.31769847869873, 22.022035598754883),
    11: ("raw_gyro_norm", 4.021728992462158, 3.3561110496520996),
    12: ("raw_acc_delta_norm", 4.644360065460205, 17.899290084838867),
    13: ("linear_acc_delta_norm", 4.587034702301025, 17.903532028198242),
    14: ("gyro_delta_norm", 0.7714515328407288, 2.0652413368225098),
    15: ("raw_linear_dot", 665.7153930664062, 3529.447998046875),
    16: ("raw_linear_cos", 0.3611887991428375, 0.6688123345375061),
}

DEFAULT_STANDARDIZE_STATS = {
    "raw_acc_mean": RAW_ACC_MEAN,
    "raw_acc_std": RAW_ACC_STD,
    "raw_gyro_mean": RAW_GYRO_MEAN,
    "raw_gyro_std": RAW_GYRO_STD,
    "linear_acc_mean": LINEAR_ACC_MEAN,
    "linear_acc_std": LINEAR_ACC_STD,
    **{name + "_mean": mean for _channel, (name, mean, _std) in HANDCRAFTED_STATS.items()},
    **{name + "_std": std for _channel, (name, _mean, std) in HANDCRAFTED_STATS.items()},
}


@dataclass(frozen=True)
class DrumGesturePrediction:
    class_id: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]


class RegularizedInceptionTime(torch.nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int | None = None,
        nf: int = 32,
        ks: int = 15,
        depth: int = 4,
        bottleneck: bool = False,
        dropout: float = 0.0,
        head: str = "gap",
        gru_hidden: int = 32,
    ):
        super().__init__()
        self.inceptionblock = InceptionTime(
            c_in,
            c_out,
            nf=nf,
            ks=ks,
            depth=depth,
            bottleneck=bottleneck,
        ).inceptionblock
        self.head = head
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.dropout = torch.nn.Dropout(dropout) if dropout and dropout > 0 else torch.nn.Identity()
        feature_dim = nf * 4
        if head == "weighted":
            n_t = int(seq_len or 15)
            init_w = torch.exp(torch.linspace(0, 2.0, n_t))
            self.time_weights = torch.nn.Parameter(init_w)
            self.fc = torch.nn.Linear(feature_dim, c_out)
        elif head == "attention":
            self.attn = torch.nn.Sequential(
                torch.nn.Conv1d(feature_dim, max(feature_dim // 2, 1), kernel_size=1),
                torch.nn.Tanh(),
                torch.nn.Conv1d(max(feature_dim // 2, 1), 1, kernel_size=1),
            )
            self.fc = torch.nn.Linear(feature_dim * 2, c_out)
        elif head == "gru":
            self.gru = torch.nn.GRU(
                input_size=feature_dim,
                hidden_size=gru_hidden,
                batch_first=True,
                bidirectional=True,
            )
            self.attn = torch.nn.Linear(gru_hidden * 2, 1)
            self.fc = torch.nn.Linear(feature_dim + gru_hidden * 2, c_out)
        else:
            self.fc = torch.nn.Linear(feature_dim, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inceptionblock(x)
        gap = self.gap(x).squeeze(-1)
        if self.head == "weighted":
            T = x.shape[-1]
            w = torch.softmax(self.time_weights[:T], dim=0)
            features = (x * w.view(1, 1, -1)).sum(dim=-1)
        elif self.head == "attention":
            attn_weights = torch.softmax(self.attn(x), dim=-1)
            pooled = (x * attn_weights).sum(dim=-1)
            features = torch.cat([gap, pooled], dim=1)
        elif self.head == "gru":
            seq = x.transpose(1, 2)
            seq, _ = self.gru(seq)
            attn_weights = torch.softmax(self.attn(seq), dim=1)
            pooled = (seq * attn_weights).sum(dim=1)
            features = torch.cat([gap, pooled], dim=1)
        else:
            features = gap
        return self.fc(self.dropout(features))


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


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)


def _delta(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=0, prepend=values[:1]).astype(np.float32)


def build_rawlin12_handcrafted_window(
    raw_linear_window: np.ndarray,
    window_size: int | None = None,
) -> np.ndarray:
    """从 9 通道原始窗口 [raw_acc×3, raw_gyro×3, linear_acc×3] 构建 17 通道特征窗口。

    输出 17 通道:
      raw_acc[3] + raw_gyro[3] + linear_acc[3]          = 9  (基础)
      raw_acc_norm + linear_acc_norm + raw_gyro_norm     = 3  (模长)
      raw_acc_delta + linear_acc_delta + gyro_delta       = 3  (急动度)
      raw_linear_dot + raw_linear_cos                     = 2  (运动-重力夹角)
    """
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


class DrumGestureRecognizer:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError(f"该识别器仅允许 CPU 推理，收到: {self.device}")
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"未找到 drum gesture checkpoint: {self.checkpoint_path}")

        self.standardize_stats, self.window_size = self._load_eval_metadata()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        fc_weight = state_dict.get("fc.weight")
        if fc_weight is None:
            raise KeyError("checkpoint 缺少 fc.weight，无法确定类别数")
        class_count = int(fc_weight.shape[0])
        self.n_classes = class_count
        c_in = self._infer_input_channels(state_dict)
        self.input_channels = c_in
        self.model_head = "weighted" if "time_weights" in state_dict else "gap"
        self.model = RegularizedInceptionTime(
            c_in=c_in,
            c_out=class_count,
            seq_len=self.window_size,
            head=self.model_head,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _load_eval_metadata(self) -> tuple[dict[str, float], int]:
        stats = dict(DEFAULT_STANDARDIZE_STATS)
        window_size = WINDOW_SIZE
        eval_path = self.checkpoint_path.with_name(f"{self.checkpoint_path.stem}_eval.json")
        if not eval_path.exists():
            return stats, window_size

        try:
            with eval_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 兼容旧版 GBK 编码的 eval 文件 (如中文路径)
            with eval_path.open("r", encoding="gbk") as file:
                metadata = json.load(file)
        standardize_stats = metadata.get("standardize_stats", {})
        if isinstance(standardize_stats, dict):
            for key, value in standardize_stats.items():
                if isinstance(value, (int, float)):
                    stats[key] = float(value)
        input_shape = metadata.get("input_shape")
        if isinstance(input_shape, list) and len(input_shape) >= 2:
            window_size = int(input_shape[1])
        return stats, window_size

    def _infer_input_channels(self, state_dict: dict[str, torch.Tensor]) -> int:
        for key, value in state_dict.items():
            if key.endswith("bottleneck.weight") and value.ndim == 3:
                return int(value.shape[1])
        return FEATURE_CHANNELS

    def prepare(self, raw_linear_window: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_linear_window, dtype=np.float32)
        if self.input_channels == FEATURE_CHANNELS:
            values = build_rawlin12_handcrafted_window(values, self.window_size)
        elif self.input_channels == 21:
            # 旧版 21 通道格式 (12 base + 9 handcrafted):
            #   0-2: raw_acc    3-5: raw_gyro    6-8: linear_acc    9-11: gyro (=raw_gyro)
            #   12-20: handcrafted (比 17ch 多一个 gyro_norm 在末尾)
            feat17 = build_rawlin12_handcrafted_window(values, self.window_size)  # (W, 17)
            raw_gyro = values[:, 3:6]
            gyro_norm = _norm(raw_gyro)  # = raw_gyro_norm, 重复通道
            values = np.concatenate(
                [feat17[:, :9], raw_gyro, feat17[:, 9:], gyro_norm], axis=1
            ).astype(np.float32)
        elif self.input_channels == 6:
            if values.shape == (self.window_size, 12):
                values = values[:, 6:12]
            if values.shape != (self.window_size, 6):
                raise ValueError(f"输入窗口必须为 ({self.window_size}, 6)，收到: {values.shape}")
        else:
            raise ValueError(f"不支持的 drum gesture 输入通道数: {self.input_channels}")
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        values = values.T.copy()
        stats = self.standardize_stats
        if self.input_channels in (FEATURE_CHANNELS, 21):
            values[0:3, :] = (values[0:3, :] - stats["raw_acc_mean"]) / stats["raw_acc_std"]
            values[3:6, :] = (values[3:6, :] - stats["raw_gyro_mean"]) / stats["raw_gyro_std"]
            values[6:9, :] = (values[6:9, :] - stats["linear_acc_mean"]) / stats["linear_acc_std"]
            if self.input_channels == 21:
                # channels 9-11: gyro (= raw_gyro duplicate), same mean/std
                values[9:12, :] = (values[9:12, :] - stats.get("gyro_mean", stats["raw_gyro_mean"])) / stats.get("gyro_std", stats["raw_gyro_std"])
            for channel, (_name, mean, std) in HANDCRAFTED_STATS.items():
                channel_mean = stats.get(f"{_name}_mean", mean)
                channel_std = stats.get(f"{_name}_std", std)
                if self.input_channels == 21:
                    # 21ch: handcrafted channels 12-20 (offset by +3 from 17ch's 9-16)
                    ch = channel + 3
                else:
                    ch = channel
                values[ch, :] = (values[ch, :] - channel_mean) / channel_std
            if self.input_channels == 21:
                # 额外通道 20: gyro_norm (= raw_gyro_norm duplicate)
                gnm = stats.get("gyro_norm_mean", stats.get("raw_gyro_norm_mean", 4.0))
                gns = stats.get("gyro_norm_std", stats.get("raw_gyro_norm_std", 3.4))
                values[20, :] = (values[20, :] - gnm) / gns
        else:
            values[:3, :] = (values[:3, :] - stats["linear_acc_mean"]) / stats["linear_acc_std"]
            values[3:, :] = (values[3:, :] - stats["gyro_mean"]) / stats["gyro_std"]
        return values

    @torch.inference_mode()
    def predict(self, raw_linear_window: np.ndarray) -> DrumGesturePrediction:
        prepared = self.prepare(raw_linear_window)
        inputs = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
        logits = self.model(inputs)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy().astype(np.float32, copy=False)
        class_id = int(np.argmax(probs))
        return DrumGesturePrediction(
            class_id=class_id,
            label=DRUM_CLASS_LABELS.get(class_id, str(class_id)),
            confidence=float(probs[class_id]),
            probabilities=tuple(float(value) for value in probs.tolist()),
        )
