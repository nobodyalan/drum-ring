"""Thin wrapper around subprocess launching for desktop tools/apps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from apps.registry import AppLaunchSpec


class AppLauncher:
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.processes: list[subprocess.Popen] = []

    def launch(self, spec: AppLaunchSpec, *, host: str, port: int) -> subprocess.Popen:
        script = self.workspace_root / spec.script_path
        if not script.exists():
            raise FileNotFoundError(spec.script_path)

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                spec.module_path,
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=str(self.workspace_root),
            start_new_session=True,
        )
        self.processes.append(process)
        return process
