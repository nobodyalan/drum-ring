import argparse
from pathlib import Path

from apps.features.whisper.touch import tr, WhisperTouchAppSpec, WhisperTouchModeSpec, WhisperTouchWindowBase
from apps.framework.bootstrap import create_base_parser, run_window
from apps.framework.ui_style import APP_STYLE_SHEET

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from whisper.whisper_with_pretrain_recognizer import WhisperWithPretrainRecognizer

    HAS_WHISPER_RUNTIME = True
    WHISPER_RUNTIME_ERROR = ""
except Exception as exc:
    WhisperWithPretrainRecognizer = None
    HAS_WHISPER_RUNTIME = False
    WHISPER_RUNTIME_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET
APP_SPEC = WhisperTouchAppSpec(
    window_title=tr("app_window_title"),
    title=tr("app_title"),
    description=tr("app_description"),
    modes=(
        WhisperTouchModeSpec(
            key="number",
            label=tr("mode_number"),
            model_label="checkpoint/whisper-number-exp",
            checkpoint_dir=PROJECT_ROOT / "checkpoint" / "whisper-number-exp",
        ),
        WhisperTouchModeSpec(
            key="lowercase",
            label=tr("mode_lowercase"),
            model_label="checkpoint/whisper-word-exp",
            checkpoint_dir=PROJECT_ROOT / "checkpoint" / "whisper-word-exp",
        ),
    ),
    default_mode_key="number",
)


class WhisperWritingTouchWindow(WhisperTouchWindowBase):
    def __init__(self, host: str, port: int):
        super().__init__(
            host=host,
            port=port,
            spec=APP_SPEC,
            recognizer_cls=WhisperWithPretrainRecognizer,
            runtime_available=HAS_WHISPER_RUNTIME,
            runtime_error=WHISPER_RUNTIME_ERROR,
        )


WhisperNumberTouchWindow = WhisperWritingTouchWindow


def parse_args() -> argparse.Namespace:
    parser = create_base_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_window(
        lambda: WhisperWritingTouchWindow(host=args.host, port=args.port),
        style_sheet=STYLE_SHEET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
