from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from models.cnn import SimpleCNN

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from openvino import Core

    HAS_OPENVINO_RUNTIME = True
    OPENVINO_RUNTIME_ERROR = ""
except Exception as exc:
    Core = None
    HAS_OPENVINO_RUNTIME = False
    OPENVINO_RUNTIME_ERROR = str(exc)


CONFIG_FILENAME = "config.yaml"
CHECKPOINT_FILENAME = "swipe_cnn_classification_best.pt"
CONFIG_KEYS = ("input_dim", "conv_dim", "input_dropout1d")
SWIPE_CLASS_LABELS = {
    0: "空",
    1: "上",
    2: "下",
    3: "左",
    4: "右",
    5: "点击",
}
SWIPE_CLASS_ID_SETS = {
    5: (1, 2, 3, 4, 5),
    6: (0, 1, 2, 3, 4, 5),
}

SWIPE_ZSCORE_MEAN = np.array([-1.126, -4.515, -2.414, 0.032, -0.016, -0.010], dtype=np.float32)
SWIPE_ZSCORE_STD = np.array([4.050, 4.676, 5.885, 1.158, 1.655, 1.128], dtype=np.float32)
# SWIPE_ZSCORE_MEAN = np.array([-1.112, -4.195, -2.152, 0.030, -0.011, -0.010], dtype=np.float32)
# SWIPE_ZSCORE_STD = np.array([4.122, 5.141, 6.103, 1.173, 1.619, 1.140], dtype=np.float32)


@dataclass(frozen=True)
class SwipePrediction:
    class_id: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]


def resolve_openvino_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).resolve()
    if path.is_file():
        if path.suffix.lower() != ".xml":
            raise ValueError(f"OpenVINO 模型文件必须是 .xml，收到: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"未找到 OpenVINO 模型路径: {path}")

    preferred = path / f"{path.name}.xml"
    if preferred.exists():
        return preferred

    candidates = sorted(path.glob("*.xml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"目录中未找到 OpenVINO xml 模型: {path}")
    raise RuntimeError(f"目录中存在多个 OpenVINO xml 模型，请显式指定文件路径: {path}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _partial_shape_to_dims(partial_shape) -> list[int | None] | None:
    rank = getattr(partial_shape, "rank", None)
    if rank is not None and getattr(rank, "is_dynamic", False):
        return None

    dims: list[int | None] = []
    for dim in partial_shape:
        if getattr(dim, "is_dynamic", False):
            dims.append(None)
            continue
        if hasattr(dim, "get_length"):
            dims.append(int(dim.get_length()))
            continue
        dims.append(int(dim))
    return dims


def load_swipe_config(config_path: str | Path) -> dict[str, float]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")

    if yaml is not None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        task_cfg = dict(payload.get("task", {}))
        return {
            "input_dim": int(task_cfg["input_dim"]),
            "conv_dim": int(task_cfg["conv_dim"]),
            "input_dropout1d": float(task_cfg["input_dropout1d"]),
        }

    values: dict[str, float] = {}
    in_task_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
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
        values[key] = float(value.strip())

    missing = [key for key in CONFIG_KEYS if key not in values]
    if missing:
        raise KeyError(f"配置缺少字段: {', '.join(missing)}")
    return values


class SwipeCNNRecognizer:
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
        self.class_ids = SWIPE_CLASS_ID_SETS.get(int(final_weight.shape[0]), tuple(range(int(final_weight.shape[0]))))
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
        self.class_ids = SWIPE_CLASS_ID_SETS.get(output_dim, tuple(range(output_dim)))

    def prepare(self, imu_values: np.ndarray) -> np.ndarray:
        values = np.asarray(imu_values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"IMU 数据形状必须为 (N, {self.input_dim})，收到: {values.shape}")
        if values.shape[0] <= 0:
            raise ValueError("IMU 数据为空，无法推理")
        if self.window_size is not None and values.shape[0] != self.window_size:
            raise ValueError(f"OpenVINO 模型要求窗口长度为 {self.window_size}，收到: {values.shape[0]}")

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = (values - SWIPE_ZSCORE_MEAN) / SWIPE_ZSCORE_STD
        return values.astype(np.float32, copy=False)

    @torch.inference_mode()
    def predict(self, imu_values: np.ndarray) -> SwipePrediction:
        prepared = self.prepare(imu_values)
        if self.backend == "pytorch":
            inputs = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
            logits = self.model(inputs)
            probs = torch.softmax(logits[0], dim=0).cpu().numpy().astype(np.float32, copy=False)
        else:
            logits = self.compiled_model({self.input_layer: prepared[None, ...]})[self.output_layer]
            probs = _softmax(np.asarray(logits, dtype=np.float32).reshape(1, -1))[0].astype(np.float32, copy=False)
        top_index = int(np.argmax(probs))
        class_id = int(self.class_ids[top_index])
        return SwipePrediction(
            class_id=class_id,
            label=SWIPE_CLASS_LABELS.get(class_id, str(class_id)),
            confidence=float(probs[top_index]),
            probabilities=tuple(float(value) for value in probs.tolist()),
        )
