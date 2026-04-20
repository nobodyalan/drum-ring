from __future__ import annotations

"""Standalone swipe classifier app composed from the shared classifier window."""

from apps.features.window_classifier import SWIPE_PRESET, run_window_classifier_app


def main() -> int:
    return run_window_classifier_app(SWIPE_PRESET)


if __name__ == "__main__":
    raise SystemExit(main())
