from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from models.cnn import SimpleCNN
from models.swipe_cnn_recognizer import (
    HAS_OPENVINO_RUNTIME,
    OPENVINO_RUNTIME_ERROR,
    Core,
    load_swipe_config,
    _partial_shape_to_dims,
    resolve_openvino_model_path,
    _softmax,
)


CONFIG_FILENAME = "config.yaml"
CHECKPOINT_FILENAME = "swipe_cnn_classification_best.pt"
PINCH_VISIBLE_CLASS_IDS = (0, 1, 2)
PINCH_CLASS_LABELS = {
    0: "空",
    1: "pinch-down",
    2: "pinch-up",
    3: "move",
    4: "rotate",
}
PINCH_CLASS_ID_SETS = {
    # 3 类 checkpoint: 仅包含 空 / pinch-down / pinch-up
    3: (0, 1, 2),
    4: (1, 2, 3, 4),
    5: (0, 1, 2, 3, 4),
}

PINCH_ZSCORE_MEAN = np.array([-0.491, -1.203, -6.149, 0.066, 0.070, -0.027], dtype=np.float32)
PINCH_ZSCORE_STD = np.array([5.056, 5.035, 5.676, 2.171, 2.549, 1.258], dtype=np.float32)

@dataclass(frozen=True)
class PinchPrediction:
    class_id: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]


class PinchCNNRecognizer:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu", backend: str = "pytorch"):
        self.backend = str(backend).strip().lower()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.model_source = self.checkpoint_dir
        self.window_size: int | None = None

        if self.backend == "pytorch":
            self._init_pytorch(device)
            return
        if self.backend == "openvino":
            self._init_openvino(device)
            return
        raise ValueError(f"不支持的后端: {backend}")

    def _init_pytorch(self, device: str) -> None:
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError(f"该识别器仅允许 CPU 推理，收到: {self.device}")

        config = load_swipe_config(self.checkpoint_dir / CONFIG_FILENAME)
        state_dict = torch.load(self.checkpoint_dir / CHECKPOINT_FILENAME, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        final_weight = state_dict.get("net.14.weight")
        if final_weight is None:
            raise KeyError("checkpoint 缺少 net.14.weight，无法确定类别数")

        self.input_dim = int(config["input_dim"])
        self.class_ids = PINCH_CLASS_ID_SETS.get(
            int(final_weight.shape[0]), tuple(range(int(final_weight.shape[0])))
        )
        self.model = SimpleCNN(
            input_dim=self.input_dim,
            output_dim=len(self.class_ids),
            conv_dim=int(config["conv_dim"]),
            input_dropout1d=float(config["input_dropout1d"]),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.model_source = self.checkpoint_dir / CHECKPOINT_FILENAME

    def _init_openvino(self, device: str) -> None:
        if not HAS_OPENVINO_RUNTIME or Core is None:
            raise RuntimeError(f"OpenVINO 不可用: {OPENVINO_RUNTIME_ERROR}")

        model_path = resolve_openvino_model_path(self.checkpoint_dir)
        config_path = model_path.parent / CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(f"OpenVINO 模型目录缺少配置文件: {config_path}")

        self.ov_device = str(device).upper()
        self.model_source = model_path
        self.input_dim = int(load_swipe_config(config_path)["input_dim"])

        core = Core()
        self.compiled_model = core.compile_model(model=str(model_path), device_name=self.ov_device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        input_shape = _partial_shape_to_dims(self.input_layer.partial_shape)
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"OpenVINO 输入形状异常: {input_shape}")
        self.window_size = None if input_shape[1] is None else int(input_shape[1])

        output_shape = _partial_shape_to_dims(self.output_layer.partial_shape)
        if output_shape is None or len(output_shape) < 2 or output_shape[-1] is None:
            raise ValueError(f"OpenVINO 输出形状异常: {output_shape}")
        output_dim = int(output_shape[-1])
        self.class_ids = PINCH_CLASS_ID_SETS.get(output_dim, tuple(range(output_dim)))

    def prepare(self, imu_values: np.ndarray) -> np.ndarray:
        values = np.asarray(imu_values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"IMU 数据形状必须为 (N, {self.input_dim})，收到: {values.shape}")
        if values.shape[0] <= 0:
            raise ValueError("IMU 数据为空，无法推理")
        if self.window_size is not None and values.shape[0] != self.window_size:
            raise ValueError(f"OpenVINO 模型要求窗口长度为 {self.window_size}，收到: {values.shape[0]}")

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = (values - PINCH_ZSCORE_MEAN) / PINCH_ZSCORE_STD
        return values.astype(np.float32, copy=False)

    @torch.inference_mode()
    def predict(self, imu_values: np.ndarray) -> PinchPrediction:
        prepared = self.prepare(imu_values)
        if self.backend == "pytorch":
            inputs = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
            logits = self.model(inputs)
            raw_probs = torch.softmax(logits[0], dim=0).cpu().numpy().astype(np.float32, copy=False)
        else:
            logits = self.compiled_model({self.input_layer: prepared[None, ...]})[self.output_layer]
            raw_probs = _softmax(np.asarray(logits, dtype=np.float32).reshape(1, -1))[0].astype(np.float32, copy=False)
        probs = np.zeros(len(PINCH_VISIBLE_CLASS_IDS), dtype=np.float32)
        for index, value in enumerate(raw_probs.tolist()):
            raw_class_id = int(self.class_ids[index])
            class_id = raw_class_id if raw_class_id in PINCH_VISIBLE_CLASS_IDS else 0
            probs[class_id] += float(value)

        top_index = int(np.argmax(probs))
        class_id = int(PINCH_VISIBLE_CLASS_IDS[top_index])
        return PinchPrediction(
            class_id=class_id,
            label=PINCH_CLASS_LABELS.get(class_id, str(class_id)),
            confidence=float(probs[top_index]),
            probabilities=tuple(float(value) for value in probs.tolist()),
        )
