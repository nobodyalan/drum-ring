from apps.whisper_number_touch_app import (
    APP_SPEC,
    HAS_WHISPER_RUNTIME,
    STYLE_SHEET,
    WHISPER_RUNTIME_ERROR,
    WhisperWritingTouchWindow,
    main,
    parse_args,
)

WhisperWordTouchWindow = WhisperWritingTouchWindow


if __name__ == "__main__":
    raise SystemExit(main())
