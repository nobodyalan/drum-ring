import argparse
from pathlib import Path

from apps.features.whisper.touch import WhisperTouchAppSpec, WhisperTouchWindowBase
from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.ui_style import APP_STYLE_SHEET

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from whisper.whisper_word_recognizer import WhisperWordRecognizer

    HAS_WHISPER_RUNTIME = True
    WHISPER_RUNTIME_ERROR = ""
except Exception as exc:
    WhisperWordRecognizer = None
    HAS_WHISPER_RUNTIME = False
    WHISPER_RUNTIME_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET
APP_SPEC = WhisperTouchAppSpec(
    window_title="触摸英文小写字符识别",
    title="触摸英文小写字符识别",
    description="按下触摸按键开始缓存 IMU，松开时触发一次 CPU 推理，模型来自 checkpoint/whisper-word。",
    model_label="checkpoint/whisper-word",
    checkpoint_dir=PROJECT_ROOT / "checkpoint" / "whisper-word",
)


class WhisperWordTouchWindow(WhisperTouchWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(
            host=host,
            port=port,
            spec=APP_SPEC,
            recognizer_cls=WhisperWordRecognizer,
            runtime_available=HAS_WHISPER_RUNTIME,
            runtime_error=WHISPER_RUNTIME_ERROR,
        )


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: WhisperWordTouchWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
