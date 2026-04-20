from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from PyQt6.QtWidgets import QApplication, QWidget


WindowFactory = Callable[[], QWidget]


def create_base_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17888)
    return parser


def run_window(
    window_factory: WindowFactory,
    *,
    style_sheet: str,
    style: str | None = None,
) -> int:
    app = QApplication(sys.argv)
    if style:
        app.setStyle(style)
    app.setStyleSheet(style_sheet)
    window = window_factory()
    window.show()
    return app.exec()
