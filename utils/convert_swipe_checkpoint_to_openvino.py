from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    import openvino as ov
except ModuleNotFoundError as exc:
    raise SystemExit(
        "openvino 未安装。请先在项目环境中安装依赖后再运行该脚本。"
    ) from exc

from models.cnn import SimpleCNN
from models.swipe_cnn_recognizer import CHECKPOINT_FILENAME, CONFIG_FILENAME, load_swipe_config


def discover_swipe_checkpoint_dirs() -> list[Path]:
    base_dir = APP_ROOT / "checkpoint" / "swipe"
    if not base_dir.exists():
        return []
    result: list[Path] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / CONFIG_FILENAME).exists() and (path / CHECKPOINT_FILENAME).exists():
            result.append(path)
    return result


def load_swipe_model(checkpoint_dir: Path) -> tuple[SimpleCNN, int]:
    config = load_swipe_config(checkpoint_dir / CONFIG_FILENAME)
    state_dict = torch.load(checkpoint_dir / CHECKPOINT_FILENAME, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint 结构异常: {checkpoint_dir / CHECKPOINT_FILENAME}")

    final_weight = state_dict.get("net.14.weight")
    if final_weight is None:
        raise KeyError(f"{checkpoint_dir} 缺少 net.14.weight，无法推断输出维度")

    input_dim = int(config["input_dim"])
    output_dim = int(final_weight.shape[0])
    model = SimpleCNN(
        input_dim=input_dim,
        output_dim=output_dim,
        conv_dim=int(config["conv_dim"]),
        input_dropout1d=float(config["input_dropout1d"]),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, input_dim


def default_output_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"{checkpoint_dir.name}.xml"


def save_openvino_model(ov_model, output_path: Path, *, compress_to_fp16: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ov.save_model(ov_model, str(output_path), compress_to_fp16=compress_to_fp16)
    except TypeError:
        if compress_to_fp16:
            raise
        ov.save_model(ov_model, str(output_path))


def convert_checkpoint(
    checkpoint_dir: Path,
    *,
    output_path: Path,
    window_size: int,
    compress_to_fp16: bool,
) -> Path:
    model, input_dim = load_swipe_model(checkpoint_dir)
    example_input = torch.randn(1, window_size, input_dim, dtype=torch.float32)
    ov_model = ov.convert_model(model, example_input=example_input)
    try:
        ov_model.reshape([1, window_size, input_dim])
    except Exception:
        pass
    save_openvino_model(ov_model, output_path, compress_to_fp16=compress_to_fp16)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 checkpoint/swipe 下的 PyTorch CNN checkpoint 转成 OpenVINO IR。"
    )
    parser.add_argument(
        "checkpoint_dirs",
        nargs="*",
        help="要转换的 checkpoint 目录；不传时默认扫描 checkpoint/swipe/*",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 xml 路径；仅在转换单个 checkpoint 时可用",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="导出时使用的固定时间窗长度，默认 60",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="导出为 FP16 IR（若当前 OpenVINO 版本支持）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dirs = [Path(path).expanduser().resolve() for path in args.checkpoint_dirs]
    if not checkpoint_dirs:
        checkpoint_dirs = discover_swipe_checkpoint_dirs()

    if not checkpoint_dirs:
        print("未找到可转换的 swipe checkpoint 目录", file=sys.stderr)
        return 1

    if args.output is not None and len(checkpoint_dirs) != 1:
        print("--output 仅支持单个 checkpoint 转换", file=sys.stderr)
        return 1

    if args.window_size <= 0:
        print("--window-size 必须大于 0", file=sys.stderr)
        return 1

    failed = False
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            print(f"[skip] 目录不存在: {checkpoint_dir}", file=sys.stderr)
            failed = True
            continue
        if not checkpoint_dir.is_dir():
            print(f"[skip] 不是目录: {checkpoint_dir}", file=sys.stderr)
            failed = True
            continue

        output_path = args.output.resolve() if args.output is not None else default_output_path(checkpoint_dir)
        try:
            written = convert_checkpoint(
                checkpoint_dir,
                output_path=output_path,
                window_size=int(args.window_size),
                compress_to_fp16=bool(args.fp16),
            )
        except Exception as exc:
            print(f"[fail] {checkpoint_dir}: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"[ok] {checkpoint_dir} -> {written}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
