import base64
import json
import os
from pathlib import Path
from typing import Optional, Union

from cryptography.fernet import Fernet as _Frame
from cryptography.fernet import InvalidToken as _FrameError
from cryptography.hazmat.primitives import hashes as _hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC as _Mixer

_ROOT = Path(__file__).resolve().parent
_INDEX_FILE = _ROOT / ".ui_index.json"


def _safe_output_name(name: str) -> str:
    return Path(str(name).replace(":", "_")).name


def _index() -> dict:
    return json.loads(_INDEX_FILE.read_text(encoding="utf-8"))


def _phrase() -> str:
    payload = _index()
    return "".join(
        chr(a ^ b) for a, b in zip(payload["left"], payload["right"])
    )


def _stamp() -> bytes:
    return base64.b64decode(_index()["stamp"])


def _rounds() -> int:
    return int(_index()["rounds"])


def _salt_width() -> int:
    return int(_index()["salt_width"])


def _shape(seed: str, salt: bytes, rounds: Optional[int] = None) -> bytes:
    lane = _Mixer(
        algorithm=_hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_rounds() if rounds is None else int(rounds),
    )
    return base64.urlsafe_b64encode(lane.derive(seed.encode("utf-8")))


def _wrap(data: bytes, *, original_name: str, seed: Optional[str] = None) -> bytes:
    salt = os.urandom(_salt_width())
    token = _Frame(_shape(_phrase() if seed is None else seed, salt)).encrypt(data)
    payload = {
        "salt": base64.b64encode(salt).decode("ascii"),
        "rounds": _rounds(),
        "name": original_name,
        "blob": token.decode("ascii"),
    }
    return _stamp() + json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _unwrap(blob: bytes, *, seed: Optional[str] = None) -> tuple[bytes, str]:
    mark = _stamp()
    if not blob.startswith(mark):
        raise ValueError("文件格式不正确")

    payload = json.loads(blob[len(mark) :].decode("utf-8"))
    salt = base64.b64decode(payload["salt"])
    token = str(payload["blob"]).encode("ascii")
    lane = _shape(_phrase() if seed is None else seed, salt, payload["rounds"])
    try:
        data = _Frame(lane).decrypt(token)
    except _FrameError as exc:
        raise ValueError("口令错误或文件内容已损坏") from exc
    return data, str(payload["name"])


def stash(
    output_path: Union[str, Path],
    data: bytes,
    *,
    original_name: str,
    seed: Optional[str] = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_wrap(data, original_name=original_name, seed=seed))
    return target


def materialize(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    seed: Optional[str] = None,
) -> Path:
    source = Path(input_path)
    data, original_name = _unwrap(source.read_bytes(), seed=seed)
    target_dir = Path(output_dir) if output_dir is not None else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _safe_output_name(original_name)
    target.write_bytes(data)
    return target


def fallback_seed() -> str:
    return _phrase()
