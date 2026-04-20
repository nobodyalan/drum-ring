import argparse
from pathlib import Path

from apps.features.whisper.touch import WhisperTouchAppSpec, WhisperTouchWindowBase
from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.ui_style import APP_STYLE_SHEET

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from whisper.whisper_number_recognizer import WhisperNumberRecognizer

    HAS_WHISPER_RUNTIME = True
    WHISPER_RUNTIME_ERROR = ""
except Exception as exc:
    WhisperNumberRecognizer = None
    HAS_WHISPER_RUNTIME = False
    WHISPER_RUNTIME_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET
APP_SPEC = WhisperTouchAppSpec(
    window_title="触摸数字书写识别",
    title="触摸数字书写识别",
    description="按下触摸按键开始缓存 IMU，松开时触发一次 CPU 推理，模型来自 checkpoint/whisper-number。",
    model_label="checkpoint/whisper-number",
    checkpoint_dir=PROJECT_ROOT / "checkpoint" / "whisper-number",
)


class WhisperNumberTouchWindow(WhisperTouchWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(
            host=host,
            port=port,
            spec=APP_SPEC,
            recognizer_cls=WhisperNumberRecognizer,
            runtime_available=HAS_WHISPER_RUNTIME,
            runtime_error=WHISPER_RUNTIME_ERROR,
        )


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: WhisperNumberTouchWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
