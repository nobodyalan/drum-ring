from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tsai.models.InceptionTime import InceptionTime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoint" / "drum_gesture" / "linacc_early_stream_lead6_deep_gap_raw6.pth"
)

WINDOW_SIZE = 15
PRE_PEAK_FRAMES = 12
POST_PEAK_FRAMES = 2
GRAVITY = 9.81

DRUM_CLASS_IDS = tuple(range(8))
DRUM_CLASS_LABELS = {
    0: "空",
    1: "踩镲",
    2: "通鼓2",
    3: "军鼓",
    4: "通鼓1",
    5: "通鼓3",
    6: "吊镲",
    7: "节奏镲",
}

ACC_MEAN = 1.1428884267807007
ACC_STD = 12.995551109313965
GYRO_MEAN = 0.5434064269065857
GYRO_STD = 3.6908280849456787


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
        nf: int = 32,
        ks: int = 40,
        bottleneck: bool = True,
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
            bottleneck=bottleneck,
        ).inceptionblock
        self.head = head
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.dropout = torch.nn.Dropout(dropout) if dropout and dropout > 0 else torch.nn.Identity()
        feature_dim = nf * 4
        if head == "attention":
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
        if self.head == "attention":
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


class DrumGestureRecognizer:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError(f"该识别器仅允许 CPU 推理，收到: {self.device}")
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"未找到 drum gesture checkpoint: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        fc_weight = state_dict.get("fc.weight")
        if fc_weight is None:
            raise KeyError("checkpoint 缺少 fc.weight，无法确定类别数")
        class_count = int(fc_weight.shape[0])
        self.model = RegularizedInceptionTime(c_in=6, c_out=class_count, head="gap")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def prepare(self, linear_window: np.ndarray) -> np.ndarray:
        values = np.asarray(linear_window, dtype=np.float32)
        if values.shape != (WINDOW_SIZE, 6):
            raise ValueError(f"输入窗口必须为 ({WINDOW_SIZE}, 6)，收到: {values.shape}")
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        values = values.T.copy()
        values[:3, :] = (values[:3, :] - ACC_MEAN) / ACC_STD
        values[3:, :] = (values[3:, :] - GYRO_MEAN) / GYRO_STD
        return values

    @torch.inference_mode()
    def predict(self, linear_window: np.ndarray) -> DrumGesturePrediction:
        prepared = self.prepare(linear_window)
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
