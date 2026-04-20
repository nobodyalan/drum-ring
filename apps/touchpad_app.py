from __future__ import annotations

"""Standalone touchpad app composed from the shared touchpad feature package."""

from apps.features.touchpad import run_touchpad_app


def main() -> int:
    return run_touchpad_app()


if __name__ == "__main__":
    raise SystemExit(main())
