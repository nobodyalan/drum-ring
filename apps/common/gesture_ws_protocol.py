from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


PROTOCOL_VERSION = 1

MESSAGE_TYPE_EVENT = "recognizer.event"
MESSAGE_TYPE_STATUS = "runtime.status"
MESSAGE_TYPE_COMMAND = "control.command"
MESSAGE_TYPE_ACK = "control.ack"
MESSAGE_TYPE_ERROR = "control.error"

RECOGNIZER_SWIPE = "swipe"
RECOGNIZER_PINCH = "pinch"
RECOGNIZER_SNAP = "snap"
RECOGNIZER_TOUCHPAD = "touchpad"
RECOGNIZER_AIRMOUSE = "airmouse"
RECOGNIZER_TYPE = "type"
RECOGNIZER_AUDIO = "audio"

ALL_RECOGNIZERS = (
    RECOGNIZER_SWIPE,
    RECOGNIZER_PINCH,
    RECOGNIZER_SNAP,
    RECOGNIZER_TOUCHPAD,
    RECOGNIZER_AIRMOUSE,
    RECOGNIZER_TYPE,
    RECOGNIZER_AUDIO,
)

CONTROL_ACTION_SET_RECOGNIZERS = "set_recognizers"
CONTROL_ACTION_REQUEST_STATUS = "request_status"

ALL_CONTROL_ACTIONS = (
    CONTROL_ACTION_SET_RECOGNIZERS,
    CONTROL_ACTION_REQUEST_STATUS,
)

EVENT_SWIPE_UP = "swipe-up"
EVENT_SWIPE_DOWN = "swipe-down"
EVENT_SWIPE_LEFT = "swipe-left"
EVENT_SWIPE_RIGHT = "swipe-right"
EVENT_TAP = "tap"
EVENT_PINCH_DOWN = "pinch-down"
EVENT_PINCH_UP = "pinch-up"
EVENT_SNAP = "snap"
EVENT_TOUCH_DOWN = "touch-down"
EVENT_TOUCH_UP = "touch-up"
EVENT_TEXT = "text"

WEBSOCKET_RECOGNIZER_EVENTS: dict[str, tuple[str, ...]] = {
    RECOGNIZER_SWIPE: (
        EVENT_SWIPE_UP,
        EVENT_SWIPE_DOWN,
        EVENT_SWIPE_LEFT,
        EVENT_SWIPE_RIGHT,
        EVENT_TAP,
    ),
    RECOGNIZER_PINCH: (
        EVENT_PINCH_DOWN,
        EVENT_PINCH_UP,
    ),
    RECOGNIZER_SNAP: (EVENT_SNAP,),
    RECOGNIZER_TOUCHPAD: (
        EVENT_TOUCH_DOWN,
        EVENT_TOUCH_UP,
    ),
    RECOGNIZER_AIRMOUSE: (),
    RECOGNIZER_TYPE: (EVENT_TEXT,),
    RECOGNIZER_AUDIO: (EVENT_TEXT,),
}


@dataclass(frozen=True, slots=True)
class RingTarget:
    role: str
    address: str
    name: str = ""


@dataclass(frozen=True, slots=True)
class RingStatus:
    connected: bool = False
    imu_fps: float = 0.0
    audio_fps: float = 0.0


@dataclass(frozen=True, slots=True)
class RuntimeStatusMessage:
    left: RingStatus = field(default_factory=RingStatus)
    right: RingStatus = field(default_factory=RingStatus)


@dataclass(frozen=True, slots=True)
class RecognizerEventMessage:
    recognizer: str
    event: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class ControlCommandMessage:
    action: str
    recognizers: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)


def build_recognizer_event(message: RecognizerEventMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": MESSAGE_TYPE_EVENT,
        "version": PROTOCOL_VERSION,
        "recognizer": message.recognizer,
        "event": message.event,
        "timestamp": float(message.timestamp),
        "data": dict(message.data),
    }
    if message.sequence is not None:
        payload["sequence"] = int(message.sequence)
    return payload


def build_runtime_status(message: RuntimeStatusMessage) -> dict[str, Any]:
    return {
        "type": MESSAGE_TYPE_STATUS,
        "version": PROTOCOL_VERSION,
        "left": _build_ring_status_payload(message.left),
        "right": _build_ring_status_payload(message.right),
    }


def build_control_command(
    action: str,
    *,
    recognizers: list[str] | tuple[str, ...] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": MESSAGE_TYPE_COMMAND,
        "version": PROTOCOL_VERSION,
        "action": action,
        "recognizers": list(recognizers or []),
        "payload": dict(payload or {}),
    }


def parse_control_command(payload: Mapping[str, Any]) -> ControlCommandMessage | None:
    if str(payload.get("type", "")) != MESSAGE_TYPE_COMMAND:
        return None

    action = str(payload.get("action", "")).strip()
    if not action:
        return None

    recognizers_raw = payload.get("recognizers", [])
    recognizers = tuple(str(item) for item in recognizers_raw if str(item))
    return ControlCommandMessage(
        action=action,
        recognizers=recognizers,
        payload=dict(payload.get("payload", {}) or {}),
    )


def build_ack_message(
    action: str,
    *,
    ok: bool,
    detail: str = "",
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": MESSAGE_TYPE_ACK,
        "version": PROTOCOL_VERSION,
        "action": action,
        "ok": bool(ok),
        "detail": detail,
        "payload": dict(payload or {}),
    }


def build_error_message(code: str, detail: str, *, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": MESSAGE_TYPE_ERROR,
        "version": PROTOCOL_VERSION,
        "code": code,
        "detail": detail,
        "payload": dict(payload or {}),
    }


def _build_ring_status_payload(status: RingStatus) -> dict[str, Any]:
    return {
        "connected": bool(status.connected),
        "imu_fps": float(status.imu_fps),
        "audio_fps": float(status.audio_fps),
    }
