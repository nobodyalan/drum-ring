from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - surfaced as runtime error by caller
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINGERSCRIBE_TRAIN_ROOT = PROJECT_ROOT / "FingerScribe" / "train"
if str(FINGERSCRIBE_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(FINGERSCRIBE_TRAIN_ROOT))

from modules.whisper_pretrain import (  # noqa: E402
    SPECIAL_END_ID,
    SPECIAL_PAD_ID,
    SPECIAL_START_ID,
    SPECIAL_TOKEN_OFFSET,
    WhisperWithPretrain,
)


DIGIT_VOCAB = "0123456789"
LOWERCASE_LETTER_VOCAB = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE_LETTER_VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_STANDARDIZE_MEAN = [-1.854, -1.730, -2.379, -0.005, -0.005, -0.008]
DEFAULT_STANDARDIZE_STD = [3.451, 3.310, 4.526, 0.622, 1.457, 1.059]


@dataclass(frozen=True)
class WhisperWithPretrainPrediction:
    text: str
    token_ids: tuple[int, ...]
    step_confidences: tuple[float, ...]
    average_confidence: float
    original_length: int
    used_length: int
    was_resampled: bool


@dataclass(frozen=True)
class PreparedIMUSample:
    values: np.ndarray
    original_length: int
    used_length: int
    was_resampled: bool


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to load Whisper-with-pretrain configs.")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("task"), dict):
        raise ValueError(f"Invalid Whisper-with-pretrain config: {path}")
    return payload


def _resolve_checkpoint_path(run_dir: Path) -> Path:
    candidates = (
        run_dir / "models" / "whisper_with_pretrain_best.pt",
        run_dir / "models" / "whisper_with_pretrain_final.pt",
        run_dir / "whisper_with_pretrain_best.pt",
        run_dir / "whisper_with_pretrain_final.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No Whisper-with-pretrain checkpoint found under {run_dir}:\n{searched}")


def _normalize_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload is not a state dict.")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = str(key)
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        normalized[normalized_key] = value
    if not normalized:
        raise ValueError("Checkpoint contains no tensor weights.")
    return normalized


def _state_emb_len(state_dict: dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key == "encoder.emb_time.time_encodings" or key.endswith(".encoder.emb_time.time_encodings"):
            return int(value.shape[0])
    raise KeyError("Unable to infer encoder time-embedding length from checkpoint.")


def _build_vocab(content_types: list[str], load_uppercase_letters: bool) -> str:
    selected = set(content_types)
    parts: list[str] = []
    if selected & {"single_digit", "number_string"}:
        parts.append(DIGIT_VOCAB)
    if selected & {"single_letter", "word"}:
        parts.append(LOWERCASE_LETTER_VOCAB + (UPPERCASE_LETTER_VOCAB if load_uppercase_letters else ""))
    return "".join(parts) or (DIGIT_VOCAB + LOWERCASE_LETTER_VOCAB)


def _ceil_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return int(length)
    return int(math.ceil(max(int(length), 1) / float(multiple)) * multiple)


class WhisperWithPretrainRecognizer:
    """Small inference wrapper for checkpoint/whisper-*-exp runs."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "cpu",
        *,
        checkpoint_path: str | Path | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.config_path = self.checkpoint_dir / ".hydra" / "config.yaml"
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else _resolve_checkpoint_path(self.checkpoint_dir)
        )
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError(f"This recognizer is intentionally CPU-only, got: {self.device}")

        config = _load_yaml(self.config_path)
        task_cfg = dict(config["task"])
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = _normalize_state_dict(checkpoint)

        self.vocab = _build_vocab(
            [str(item) for item in task_cfg.get("train_content_types", [])],
            bool(task_cfg.get("load_uppercase_letters", True)),
        )
        self.max_decode_steps = int(task_cfg.get("token_len", 15))
        self.pad_to_multiple = int(task_cfg.get("pad_to_multiple") or math.prod(task_cfg.get("tokenizer_strides", [1])))
        self.stride_product = int(math.prod([int(value) for value in task_cfg.get("tokenizer_strides", [1])]))
        self.max_input_length = _state_emb_len(state_dict) * self.stride_product

        self.mean = np.asarray(task_cfg.get("standardize_mean", DEFAULT_STANDARDIZE_MEAN), dtype=np.float32).reshape(1, 6)
        self.std = np.asarray(task_cfg.get("standardize_std", DEFAULT_STANDARDIZE_STD), dtype=np.float32).reshape(1, 6)
        self.std = np.clip(self.std, a_min=1e-6, a_max=None)

        self.model = WhisperWithPretrain(
            input_dim=int(task_cfg.get("input_dim", 6)),
            max_seq_len=self.max_input_length,
            vocab_size=SPECIAL_TOKEN_OFFSET + len(self.vocab),
            token_len=self.max_decode_steps,
            tokenizer_filters=[int(value) for value in task_cfg.get("tokenizer_filters", [128, 128])],
            tokenizer_kernel_sizes=[int(value) for value in task_cfg.get("tokenizer_kernel_sizes", [7, 7])],
            tokenizer_strides=[int(value) for value in task_cfg.get("tokenizer_strides", [1, 6])],
            encoder_n_blocks=int(task_cfg.get("encoder_n_blocks", 12)),
            encoder_n_heads=int(task_cfg.get("encoder_n_heads", 8)),
            decoder_n_blocks=int(task_cfg.get("decoder_n_blocks", 6)),
            decoder_n_heads=int(task_cfg.get("decoder_n_heads", 4)),
            cls_loss_scale=float(task_cfg.get("cls_loss_scale", 1.0)),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def prepare(self, imu_values: np.ndarray) -> PreparedIMUSample:
        values = np.asarray(imu_values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"IMU values must have shape (N, 6), got {values.shape}")
        if values.shape[0] <= 0:
            raise ValueError("IMU values are empty.")

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        original_length = int(values.shape[0])
        was_resampled = False
        if original_length > self.max_input_length:
            values = self._resample(values, self.max_input_length)
            was_resampled = True

        used_length = int(values.shape[0])
        values = (values - self.mean) / self.std
        padded_length = min(_ceil_to_multiple(used_length, self.pad_to_multiple), self.max_input_length)
        if padded_length > used_length:
            values = np.pad(values, ((0, padded_length - used_length), (0, 0)), mode="constant", constant_values=0.0)

        return PreparedIMUSample(
            values=values.astype(np.float32, copy=False),
            original_length=original_length,
            used_length=used_length,
            was_resampled=was_resampled,
        )

    @torch.inference_mode()
    def predict(self, imu_values: np.ndarray) -> WhisperWithPretrainPrediction:
        prepared = self.prepare(imu_values)
        inputs = torch.from_numpy(prepared.values).unsqueeze(0).to(self.device)
        lengths = torch.tensor([prepared.used_length], dtype=torch.long, device=self.device)
        encoded, encoded_lengths = self.model.encode(inputs, lengths)

        tokens = torch.full((1, 1), SPECIAL_START_ID, dtype=torch.long, device=self.device)
        token_ids: list[int] = []
        step_confidences: list[float] = []
        for _ in range(self.max_decode_steps):
            logits = self.model.decoder(tokens, encoded, encoded_lengths)[:, -1]
            probs = torch.softmax(logits, dim=-1)
            next_token = int(torch.argmax(probs, dim=-1).item())
            step_confidences.append(float(probs[0, next_token].item()))
            if next_token in {SPECIAL_END_ID, SPECIAL_PAD_ID}:
                break
            token_ids.append(next_token)
            tokens = torch.cat(
                [tokens, torch.tensor([[next_token]], dtype=torch.long, device=self.device)],
                dim=1,
            )

        return WhisperWithPretrainPrediction(
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
        source_x = np.arange(values.shape[0], dtype=np.float32)
        target_x = np.linspace(0.0, float(values.shape[0] - 1), num=int(target_length), dtype=np.float32)
        return np.stack(
            [np.interp(target_x, source_x, values[:, channel_idx]) for channel_idx in range(values.shape[1])],
            axis=1,
        ).astype(np.float32)

    def _decode_tokens(self, token_ids: list[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            vocab_idx = int(token_id) - SPECIAL_TOKEN_OFFSET
            if 0 <= vocab_idx < len(self.vocab):
                chars.append(self.vocab[vocab_idx])
        return "".join(chars)
