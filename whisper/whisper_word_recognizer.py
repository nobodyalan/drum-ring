from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string

import numpy as np
import torch

from whisper.whisper_imu import (
    WHISPER_IMU_END_IDX,
    WHISPER_IMU_START_IDX,
    WhisperIMUTextRecognizer,
)

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


CONFIG_FILENAME = "config.yaml"
CHECKPOINT_FILENAME = "whisper_imu_writing_best.pt"
CONFIG_KEYS = (
    "input_dim",
    "n_audio_ctx",
    "n_audio_state",
    "n_audio_head",
    "n_audio_layer",
    "encoder_kernel_size",
    "n_text_ctx",
    "n_text_state",
    "n_text_head",
    "n_text_layer",
)

LOWERCASE_LETTERS = string.ascii_lowercase
WHISPER_WORD_VOCAB = ["<pad>", "<start>", "<end>", *list(LOWERCASE_LETTERS)]
WHISPER_WORD_CHAR_OFFSET = 3

WRITING_SET_MEAN = np.array([-2.702, -2.574, -5.256, 0.002, -0.003, -0.010], dtype=np.float32)
WRITING_SET_STD = np.array([3.901, 3.638, 4.771, 0.641, 1.536, 1.076], dtype=np.float32)

@dataclass(frozen=True)
class PreparedIMUSample:
    values: np.ndarray
    original_length: int
    used_length: int
    was_resampled: bool


@dataclass(frozen=True)
class WhisperWordPrediction:
    text: str
    token_ids: tuple[int, ...]
    step_confidences: tuple[float, ...]
    average_confidence: float
    original_length: int
    used_length: int
    was_resampled: bool


def load_whisper_word_config(config_path: str | Path) -> dict[str, int]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")

    if yaml is not None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        task_cfg = dict(payload.get("task", {}))
        return {key: int(task_cfg[key]) for key in CONFIG_KEYS}

    return _load_whisper_word_config_fallback(path)


def _load_whisper_word_config_fallback(config_path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    in_task_block = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("task:"):
            in_task_block = True
            continue
        if in_task_block and line and not line.startswith(" "):
            break
        if not in_task_block:
            continue
        if not line.startswith("  ") or line.startswith("    "):
            continue
        text = line.strip()
        if not text or text.startswith("#") or ":" not in text:
            continue
        key, value = text.split(":", 1)
        if key not in CONFIG_KEYS:
            continue
        values[key] = int(value.strip())

    missing = [key for key in CONFIG_KEYS if key not in values]
    if missing:
        raise KeyError(f"配置缺少字段: {', '.join(missing)}")
    return values


class WhisperWordRecognizer:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu"):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError(f"该识别器仅允许 CPU 推理，收到: {self.device}")

        config = load_whisper_word_config(self.checkpoint_dir / CONFIG_FILENAME)
        self.max_decode_steps = max(int(config["n_text_ctx"]) - 1, 1)
        self.max_input_length = int(config["n_audio_ctx"]) * 2

        self.model = WhisperIMUTextRecognizer(
            input_dim=int(config["input_dim"]),
            n_audio_ctx=int(config["n_audio_ctx"]),
            n_audio_state=int(config["n_audio_state"]),
            n_audio_head=int(config["n_audio_head"]),
            n_audio_layer=int(config["n_audio_layer"]),
            encoder_kernel_size=int(config["encoder_kernel_size"]),
            n_text_ctx=int(config["n_text_ctx"]),
            n_text_state=int(config["n_text_state"]),
            n_text_head=int(config["n_text_head"]),
            n_text_layer=int(config["n_text_layer"]),
            vocab_size=len(WHISPER_WORD_VOCAB),
        )

        state_dict = torch.load(self.checkpoint_dir / CHECKPOINT_FILENAME, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def prepare(self, imu_values: np.ndarray) -> PreparedIMUSample:
        values = np.asarray(imu_values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"IMU 数据形状必须为 (N, 6)，收到: {values.shape}")
        if values.shape[0] == 0:
            raise ValueError("IMU 数据为空，无法推理")

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        original_length = int(values.shape[0])
        was_resampled = False
        if original_length > self.max_input_length:
            values = self._resample(values, self.max_input_length)
            was_resampled = True

        values = (values - WRITING_SET_MEAN) / WRITING_SET_STD
        return PreparedIMUSample(
            values=values.astype(np.float32, copy=False),
            original_length=original_length,
            used_length=int(values.shape[0]),
            was_resampled=was_resampled,
        )

    @torch.inference_mode()
    def predict(self, imu_values: np.ndarray) -> WhisperWordPrediction:
        prepared = self.prepare(imu_values)
        inputs = torch.from_numpy(prepared.values).unsqueeze(0).to(self.device)
        lengths = torch.tensor([prepared.used_length], dtype=torch.long, device=self.device)
        encoded = self.model.encode(inputs, lengths)

        tokens = torch.full(
            (1, 1),
            fill_value=WHISPER_IMU_START_IDX,
            dtype=torch.long,
            device=self.device,
        )
        token_ids: list[int] = []
        step_confidences: list[float] = []

        for _ in range(self.max_decode_steps):
            logits = self.model.decode(
                tokens,
                encoded["audio_features"],
                audio_padding_mask=encoded["audio_padding_mask"],
            )
            probs = torch.softmax(logits[:, -1], dim=-1)
            next_token = int(torch.argmax(probs, dim=-1).item())
            step_confidences.append(float(probs[0, next_token].item()))

            if next_token == WHISPER_IMU_END_IDX:
                break
            if next_token < WHISPER_WORD_CHAR_OFFSET or next_token >= len(WHISPER_WORD_VOCAB):
                break

            token_ids.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            tokens = torch.cat([tokens, next_token_tensor], dim=1)

        return WhisperWordPrediction(
            text=self._decode_tokens(token_ids),
            token_ids=tuple(token_ids),
            step_confidences=tuple(step_confidences),
            average_confidence=float(np.mean(step_confidences)) if step_confidences else 0.0,
            original_length=prepared.original_length,
            used_length=prepared.used_length,
            was_resampled=prepared.was_resampled,
        )

    @staticmethod
    def _resample(values: np.ndarray, target_length: int) -> np.ndarray:
        if values.shape[0] <= target_length:
            return values
        source_x = np.arange(values.shape[0], dtype=np.float32)
        target_x = np.linspace(0.0, float(values.shape[0] - 1), num=target_length, dtype=np.float32)
        return np.stack(
            [np.interp(target_x, source_x, values[:, channel_idx]) for channel_idx in range(values.shape[1])],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _decode_tokens(token_ids: list[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            if WHISPER_WORD_CHAR_OFFSET <= token_id < len(WHISPER_WORD_VOCAB):
                chars.append(WHISPER_WORD_VOCAB[token_id])
        return "".join(chars)
