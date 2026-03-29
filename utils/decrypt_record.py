import argparse
from pathlib import Path

from apps.common.file_saver import fallback_seed, materialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="解密录制导出的 .enc 文件，恢复原始 .npy 或 .wav 文件。"
    )
    parser.add_argument("inputs", nargs="+", help="一个或多个 .enc 文件路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="解密后输出目录，默认写回每个源文件所在目录",
    )
    parser.add_argument(
        "-p",
        "--password",
        default=fallback_seed(),
        help="解包口令，默认使用内置值",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code = 0

    for raw_path in args.inputs:
        source = Path(raw_path)
        try:
            target = materialize(
                source,
                output_dir=args.output_dir,
                seed=args.password,
            )
            print(f"{source} -> {target}")
        except Exception as exc:
            exit_code = 1
            print(f"{source} 解密失败: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
