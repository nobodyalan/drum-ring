from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ENV_FILE = Path(__file__).resolve().parent / ".env"

DEFAULT_STREAM_RESOURCE_ID = "volc.bigasr.sauc.duration"
DEFAULT_FILE_RESOURCE_ID = "volc.bigasr.auc"
DEFAULT_MODEL_NAME = "bigmodel"
DEFAULT_WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"


@dataclass(frozen=True)
class DoubaoASRConfig:
    app_key: str
    access_key: str
    stream_resource_id: str
    file_resource_id: str
    model_name: str
    ws_url: str


def load_doubao_asr_config() -> DoubaoASRConfig:
    values = _parse_env_file(ENV_FILE)
    app_key = _required_value(values, "DOUBAO_APP_KEY")
    access_key = _required_value(values, "DOUBAO_ACCESS_KEY")
    return DoubaoASRConfig(
        app_key=app_key,
        access_key=access_key,
        stream_resource_id=_optional_value(values, "DOUBAO_RESOURCE_ID", DEFAULT_STREAM_RESOURCE_ID),
        file_resource_id=_optional_value(values, "DOUBAO_FILE_RESOURCE_ID", DEFAULT_FILE_RESOURCE_ID),
        model_name=_optional_value(values, "DOUBAO_MODEL_NAME", DEFAULT_MODEL_NAME),
        ws_url=_optional_value(values, "DOUBAO_ASR_WS_URL", DEFAULT_WS_URL),
    )


def _required_value(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing required key in {ENV_FILE}: {key}")
    return value


def _optional_value(values: dict[str, str], key: str, default: str) -> str:
    value = values.get(key, "").strip()
    return value or default


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise ValueError(f"Missing ASR config file: {path}")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        values[key] = value
    return values
