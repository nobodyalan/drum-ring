from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path


def _clear_hidden_flag(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        subprocess.run(
            ["chflags", "nohidden", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def configure_qt_plugin_path() -> None:
    if "QT_PLUGIN_PATH" in os.environ:
        return

    spec = importlib.util.find_spec("PyQt6")
    if spec is None or not spec.submodule_search_locations:
        return

    pyqt_dir = Path(next(iter(spec.submodule_search_locations)))
    source_plugins = pyqt_dir / "Qt6" / "plugins"
    if not (source_plugins / "platforms").is_dir():
        return

    target_plugins = Path(__file__).resolve().parent / "qt_plugins"
    _clear_hidden_flag(target_plugins)

    for source in source_plugins.rglob("*"):
        target = target_plugins / source.relative_to(source_plugins)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _clear_hidden_flag(target)
            continue
        if not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _clear_hidden_flag(target.parent)
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, target)
            shutil.copymode(source, target)
        _clear_hidden_flag(target)

    os.environ["QT_PLUGIN_PATH"] = str(target_plugins)
