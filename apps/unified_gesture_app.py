from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import queue
import struct
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tsai.models.InceptionTime import InceptionTime

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from common.stream_window import NONE_SUBSCRIPTION, RingStreamWindowBase
    from common.ui_style import APP_STYLE_SHEET
    from common.gesture_ws_protocol import (
        ALL_RECOGNIZERS,
        CONTROL_ACTION_REQUEST_STATUS,
        CONTROL_ACTION_SET_RECOGNIZERS,
        RECOGNIZER_AIRMOUSE,
        RECOGNIZER_AUDIO,
        RECOGNIZER_PINCH,
        RECOGNIZER_SNAP,
        RECOGNIZER_SWIPE,
        RECOGNIZER_TOUCHPAD,
        RECOGNIZER_TYPE,
        ControlCommandMessage,
        RecognizerEventMessage,
        RingTarget,
        RingStatus,
        RuntimeStatusMessage,
        VOICE_STATE_FINISHED,
        VOICE_STATE_LISTENING,
        VoiceStatusMessage,
        build_ack_message,
        build_error_message,
        build_recognizer_event,
        build_runtime_status,
        build_voice_status,
        parse_control_command,
    )
except ModuleNotFoundError:
    from apps.common.stream_window import NONE_SUBSCRIPTION, RingStreamWindowBase
    from apps.common.ui_style import APP_STYLE_SHEET
    from apps.common.gesture_ws_protocol import (
        ALL_RECOGNIZERS,
        CONTROL_ACTION_REQUEST_STATUS,
        CONTROL_ACTION_SET_RECOGNIZERS,
        RECOGNIZER_AIRMOUSE,
        RECOGNIZER_AUDIO,
        RECOGNIZER_PINCH,
        RECOGNIZER_SNAP,
        RECOGNIZER_SWIPE,
        RECOGNIZER_TOUCHPAD,
        RECOGNIZER_TYPE,
        ControlCommandMessage,
        RecognizerEventMessage,
        RingTarget,
        RingStatus,
        RuntimeStatusMessage,
        VOICE_STATE_FINISHED,
        VOICE_STATE_LISTENING,
        VoiceStatusMessage,
        build_ack_message,
        build_error_message,
        build_recognizer_event,
        build_runtime_status,
        build_voice_status,
        parse_control_command,
    )

from apps.gesture_recognition_app import GestureConfig, GestureInferenceWorker
from models.pinch_cnn_recognizer import PinchCNNRecognizer
from models.swipe_cnn_recognizer import SwipeCNNRecognizer
from models.tcn_lstm import TCNLSTMStreaming
from utils.cd_ratio import CDRatioEstimator, OneEuroFilter
from utils.imu_processor import IMUProcessor
from utils.window import Window

try:
    from pyquaternion import Quaternion as PyQuaternion

    HAS_AIRMOUSE_RUNTIME = True
    AIRMOUSE_RUNTIME_ERROR = ""
except Exception as exc:
    PyQuaternion = None
    HAS_AIRMOUSE_RUNTIME = False
    AIRMOUSE_RUNTIME_ERROR = str(exc)

try:
    from pynput import mouse

    HAS_TOUCHPAD_CURSOR_CONTROL = True
    TOUCHPAD_CURSOR_CONTROL_ERROR = ""
except Exception as exc:
    mouse = None
    HAS_TOUCHPAD_CURSOR_CONTROL = False
    TOUCHPAD_CURSOR_CONTROL_ERROR = str(exc)

try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

try:
    from PyQt6.QtNetwork import QHostAddress
    from PyQt6.QtWebSockets import QWebSocketServer

    HAS_QT_WEBSOCKETS = True
    QT_WEBSOCKETS_ERROR = ""
except Exception as exc:
    QHostAddress = None
    QWebSocketServer = None
    HAS_QT_WEBSOCKETS = False
    QT_WEBSOCKETS_ERROR = str(exc)

if sys.platform == "darwin":
    try:
        from Foundation import (
            NSActivityAutomaticTerminationDisabled,
            NSActivitySuddenTerminationDisabled,
            NSActivityUserInitiated,
            NSProcessInfo,
        )

        HAS_MACOS_KEEPALIVE = True
        MACOS_KEEPALIVE_ERROR = ""
        MACOS_KEEPALIVE_OPTIONS = (
            int(NSActivityUserInitiated)
            | int(NSActivitySuddenTerminationDisabled)
            | int(NSActivityAutomaticTerminationDisabled)
        )
    except Exception as exc:
        NSProcessInfo = None
        HAS_MACOS_KEEPALIVE = False
        MACOS_KEEPALIVE_ERROR = str(exc)
        MACOS_KEEPALIVE_OPTIONS = 0
else:
    NSProcessInfo = None
    HAS_MACOS_KEEPALIVE = False
    MACOS_KEEPALIVE_ERROR = "仅支持 macOS"
    MACOS_KEEPALIVE_OPTIONS = 0

try:
    from core.asr.model.doubao_asr_stream import DoubaoASRStream

    HAS_DOUBAO_ASR_RUNTIME = True
    DOUBAO_ASR_RUNTIME_ERROR = ""
except Exception as exc:
    DoubaoASRStream = None
    HAS_DOUBAO_ASR_RUNTIME = False
    DOUBAO_ASR_RUNTIME_ERROR = str(exc)


STYLE_SHEET = APP_STYLE_SHEET
IMU_SAMPLE_RATE_HZ = 200.0
STREAM_SEQUENCE_KEY = "__stream_seq__"


@dataclass(frozen=True)
class WindowClassifierConfig:
    checkpoint_dir: Path
    backend: str
    window_size: int
    step_frames: int
    stable_window_seconds: float
    positive_ratio: float
    reset_delay_frames: int


@dataclass(frozen=True)
class SnapRecognizerConfig:
    allowed_gestures: tuple[int, ...]
    skipped_gestures: tuple[int, ...]
    positive_ratio: float
    result_window_length: int
    window_size: int
    models: tuple[str, ...]
    models_allowed_gestures: tuple[tuple[int, ...], ...]
    device: str
    merge_window_seconds: float
    target_fps: float = 50.0


@dataclass(frozen=True)
class TouchpadRecognizerConfig:
    gesture_ckpt: Path
    motion_ckpt: Path
    gesture_window_size: int = 20
    gesture_calculate_interval: int = 2
    gesture_history_size: int = 5
    threshold_down: int = 1
    threshold_up: int = 2
    click_interval_seconds: float = 0.1
    idle_gyro_threshold: float = 0.03
    min_move_abs: float = 0.05
    move_emit_interval_seconds: float = 1 / 30.0


@dataclass(frozen=True)
class AirmouseRecognizerConfig:
    pointer_scale: float = 4000.0
    min_move_abs: float = 0.05
    move_emit_interval_seconds: float = 1 / 60.0
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.2
    filter_d_cutoff: float = 1.0
    cursor_home_x_ratio: float = 0.5
    cursor_home_y_ratio: float = 0.5


@dataclass(frozen=True)
class UnifiedAppConfig:
    websocket_enabled: bool
    websocket_host: str
    websocket_port: int
    message_log_limit: int
    status_broadcast_interval_seconds: float
    swipe: WindowClassifierConfig
    pinch: WindowClassifierConfig
    snap: SnapRecognizerConfig
    touchpad: TouchpadRecognizerConfig
    airmouse: AirmouseRecognizerConfig


APP_CONFIG = UnifiedAppConfig(
    websocket_enabled=True,
    websocket_host="127.0.0.1",
    websocket_port=18765,
    message_log_limit=300,
    status_broadcast_interval_seconds=1.0,
    swipe=WindowClassifierConfig(
        checkpoint_dir=APP_ROOT / "checkpoint" / "swipe" / "cls",
        backend="openvino",
        window_size=60,
        step_frames=2,
        stable_window_seconds=0.05,
        positive_ratio=1,
        reset_delay_frames=10,
    ),
    pinch=WindowClassifierConfig(
        checkpoint_dir=APP_ROOT / "checkpoint" / "swipe" / "cls-pinch",
        backend="openvino",
        window_size=60,
        step_frames=2,
        stable_window_seconds=0.05,
        positive_ratio=1,
        reset_delay_frames=2,
    ),
    snap=SnapRecognizerConfig(
        allowed_gestures=(16,),
        skipped_gestures=(),
        positive_ratio=0.80,
        result_window_length=20,
        window_size=200,
        models=("checkpoint/it-snap/it-snap.xml",),
        models_allowed_gestures=((16,),),
        device="CPU",
        merge_window_seconds=0.5,
        target_fps=50.0,
    ),
    touchpad=TouchpadRecognizerConfig(
        gesture_ckpt=APP_ROOT / "checkpoint" / "tap-94.pth",
        motion_ckpt=APP_ROOT / "checkpoint" / "tcnlstm.pt",
    ),
    airmouse=AirmouseRecognizerConfig(
        pointer_scale=4000.0,
        cursor_home_x_ratio=0.5,
        cursor_home_y_ratio=0.5,
    ),
)


@dataclass(frozen=True)
class InferenceJob:
    address: str
    values: np.ndarray
    generation: int


@dataclass(frozen=True)
class InferenceOutput:
    address: str
    prediction: object
    elapsed_ms: float
    sample_count: int
    generation: int


@dataclass
class ClassifierState:
    samples: deque[list[float]] = field(default_factory=lambda: deque(maxlen=1024))
    frames_since_infer: int = 0
    prediction_history: deque[int] = field(default_factory=deque)
    reset_after_frames_remaining: int = 0


@dataclass
class PendingImuReading:
    data: np.ndarray
    event_ts: float
    generation: int


@dataclass
class TouchpadModelState:
    gesture_detector: "GestureDetector"
    motion_model: TCNLSTMStreaming
    imu_processor: IMUProcessor
    cd_ratio_estimator: CDRatioEstimator
    filter_x: OneEuroFilter
    filter_y: OneEuroFilter
    imu_data: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=1))
    last_gesture: int = 3
    last_down_time: float = 0.0
    last_move_emit_time: float = 0.0


@dataclass
class AirmouseState:
    imu_processor: IMUProcessor
    filter_x: OneEuroFilter
    filter_y: OneEuroFilter
    orientation_queue: deque = field(default_factory=lambda: deque(maxlen=2))
    last_move_emit_time: float = 0.0


@dataclass
class RecognizerRowWidgets:
    checkbox: QCheckBox
    target_label: QLabel
    fps_label: QLabel
    status_label: QLabel


@dataclass
class VoiceCaptureState:
    active: bool = False
    capture_id: str = ""
    started_at: float = 0.0
    audio_message_count: int = 0


@dataclass(frozen=True)
class VoiceCaptureResult:
    capture_id: str
    started_at: float
    finished_at: float
    audio_message_count: int


@dataclass(frozen=True)
class ASRStreamPacket:
    capture_id: str
    text: str
    is_final: bool
    segment_id: int
    start_time: int
    end_time: int


class AdpcmDecoder:
    index_table = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]

    step_size_table = [
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
        19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
        50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
        130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
        337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
        876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
        2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
        5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
        15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
    ]

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.predsample = 0
        self.pcm_index = 0

    def decode(self, code: int) -> int:
        step = self.step_size_table[self.pcm_index]
        diff = step >> 3
        if code & 4:
            diff += step
        if code & 2:
            diff += step >> 1
        if code & 1:
            diff += step >> 2

        if code & 8:
            diff = -diff
        self.predsample += diff

        if self.predsample > 32767:
            self.predsample = 32767
        elif self.predsample < -32768:
            self.predsample = -32768

        self.pcm_index += self.index_table[code]
        if self.pcm_index < 0:
            self.pcm_index = 0
        if self.pcm_index > 88:
            self.pcm_index = 88
        return self.predsample

    def decode_packet(self, data: bytes) -> bytes:
        pcm = bytearray()
        for value in data:
            pcm.extend(struct.pack("<h", self.decode(value & 0xF)))
            pcm.extend(struct.pack("<h", self.decode((value >> 4) & 0xF)))
        return bytes(pcm)


class RateTracker:
    def __init__(self, window_seconds: float = 1.0):
        self.window_seconds = max(0.1, float(window_seconds))
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()

    def mark(self, timestamp: float | None = None) -> None:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        with self._lock:
            self._timestamps.append(now)
            self._trim_locked(now)

    def fps(self) -> float:
        now = time.perf_counter()
        with self._lock:
            self._trim_locked(now)
            return len(self._timestamps) / self.window_seconds

    def _trim_locked(self, now: float) -> None:
        threshold = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < threshold:
            self._timestamps.popleft()


class RecognizerBus(QObject):
    recognizer_event = pyqtSignal(str, str, object, object, float)
    recognizer_status = pyqtSignal(str, str)


class FrontendWebSocketBridge(QObject):
    command_received = pyqtSignal(object, object)
    state_changed = pyqtSignal()

    def __init__(self, host: str, port: int):
        super().__init__()
        self.host = host
        self.port = port
        self.server = None
        self.clients: list[object] = []
        self.enabled = HAS_QT_WEBSOCKETS
        self.listening = False
        self.last_error = "" if HAS_QT_WEBSOCKETS else QT_WEBSOCKETS_ERROR
        self.url = f"ws://{host}:{port}"

    def start(self) -> None:
        if not self.enabled or not HAS_QT_WEBSOCKETS:
            self.last_error = QT_WEBSOCKETS_ERROR or "Qt WebSocket 不可用"
            self.state_changed.emit()
            return
        if self.server is not None:
            return

        self.server = QWebSocketServer("gesture-bridge", QWebSocketServer.SslMode.NonSecureMode, self)
        host_address = self._host_address()
        if not self.server.listen(host_address, self.port):
            self.last_error = self.server.errorString()
            self.server.deleteLater()
            self.server = None
            self.listening = False
            self.state_changed.emit()
            return

        self.server.newConnection.connect(self._on_new_connection)
        self.listening = True
        self.last_error = ""
        self.state_changed.emit()

    def stop(self) -> None:
        for client in list(self.clients):
            try:
                client.close()
            except Exception:
                pass
        self.clients.clear()

        if self.server is not None:
            self.server.close()
            self.server.deleteLater()
            self.server = None

        self.listening = False
        self.state_changed.emit()

    def send_payload(self, payload: dict[str, Any], client: object | None = None) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        targets = [client] if client is not None else list(self.clients)
        for target in targets:
            try:
                target.sendTextMessage(text)
            except Exception:
                pass

    def client_count(self) -> int:
        return len(self.clients)

    def _host_address(self):
        if self.host in {"127.0.0.1", "localhost"}:
            return QHostAddress(QHostAddress.SpecialAddress.LocalHost)
        return QHostAddress(QHostAddress.SpecialAddress.Any)

    def _on_new_connection(self) -> None:
        if self.server is None:
            return
        while self.server.hasPendingConnections():
            client = self.server.nextPendingConnection()
            if client is None:
                return
            client.textMessageReceived.connect(lambda text, sock=client: self._on_text_message(sock, text))
            client.disconnected.connect(lambda sock=client: self._on_client_disconnected(sock))
            self.clients.append(client)
        self.state_changed.emit()

    def _on_client_disconnected(self, client: object) -> None:
        if client in self.clients:
            self.clients.remove(client)
        try:
            client.deleteLater()
        except Exception:
            pass
        self.state_changed.emit()

    def _on_text_message(self, client: object, text: str) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.send_payload(build_error_message("invalid_json", "消息不是合法 JSON"), client=client)
            return

        command = parse_control_command(payload)
        if command is None:
            self.send_payload(build_error_message("invalid_command", "不支持的控制命令"), client=client)
            return

        self.command_received.emit(command, client)


class MacOSKeepAliveManager:
    def __init__(self, reason: str):
        self.reason = str(reason)
        self.available = HAS_MACOS_KEEPALIVE
        self.error = MACOS_KEEPALIVE_ERROR
        self._token = None

    def set_enabled(self, enabled: bool) -> bool:
        if not enabled:
            self.disable()
            return False
        return self.enable()

    def enable(self) -> bool:
        if not self.available or NSProcessInfo is None:
            return False
        if self._token is not None:
            return True
        try:
            process_info = NSProcessInfo.processInfo()
            self._token = process_info.beginActivityWithOptions_reason_(
                MACOS_KEEPALIVE_OPTIONS,
                self.reason,
            )
            self.error = ""
            return self._token is not None
        except Exception as exc:
            self._token = None
            self.error = str(exc)
            return False

    def disable(self) -> None:
        if self._token is None or NSProcessInfo is None:
            self._token = None
            return
        try:
            NSProcessInfo.processInfo().endActivity_(self._token)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self._token = None

    def is_active(self) -> bool:
        return self._token is not None


class DoubaoASRStreamWorker(QThread):
    packet_ready = pyqtSignal(object)
    fatal_error = pyqtSignal(str, str)

    def __init__(self, capture_id: str, sample_rate: int):
        super().__init__()
        self.capture_id = capture_id
        self.sample_rate = max(1, int(sample_rate))
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self.segment_index = 0
        self._first_packet = True
        self._last_packet_signature: tuple[Any, ...] | None = None
        self._last_final_content_signature: tuple[Any, ...] | None = None

    def submit_pcm(self, pcm_bytes: bytes) -> None:
        if pcm_bytes:
            self.audio_queue.put_nowait(bytes(pcm_bytes))

    def finish_stream(self) -> None:
        self.audio_queue.put_nowait(None)

    def run(self) -> None:
        if DoubaoASRStream is None:
            self.fatal_error.emit(self.capture_id, DOUBAO_ASR_RUNTIME_ERROR or "Doubao ASR 不可用")
            return
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.fatal_error.emit(self.capture_id, str(exc))

    async def _run_async(self) -> None:
        async with DoubaoASRStream() as doubao:
            sender_task = asyncio.create_task(self._forward_audio(doubao))
            try:
                async for response in doubao.execute():
                    packet = self._prepare_packet(response)
                    if packet is not None:
                        self.packet_ready.emit(packet)
            finally:
                if sender_task.done():
                    await sender_task
                else:
                    sender_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender_task

    async def _forward_audio(self, doubao: Any) -> None:
        while True:
            chunk = await asyncio.to_thread(self.audio_queue.get)
            if chunk is None:
                doubao.data_queue.put_nowait(None)
                return
            if self._first_packet:
                chunk = self._wav_header() + chunk
                self._first_packet = False
            doubao.data_queue.put_nowait(chunk)

    def _prepare_packet(self, response: Any) -> ASRStreamPacket | None:
        response_dict = self._response_to_dict(response)
        if response_dict is None:
            return None

        utterances = list(response_dict.get("payload_msg", {}).get("result", {}).get("utterances", []) or [])
        if not utterances:
            return None

        utterance = utterances[0] or {}
        text = str(utterance.get("text", "") or "").strip()
        if not text:
            return None

        is_final = bool(utterance.get("definite", False))
        start_time = 0
        end_time = 0
        words = list(utterance.get("words", []) or [])
        if words:
            start_time = int(words[0].get("start_time", 0) or 0)
            end_time = int(words[-1].get("end_time", 0) or 0)

        final_content_signature = (text, start_time, end_time)
        if is_final and final_content_signature == self._last_final_content_signature:
            return None
        segment_id = self.segment_index
        packet = ASRStreamPacket(
            capture_id=self.capture_id,
            text=text,
            is_final=is_final,
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
        )
        signature = (
            packet.capture_id,
            packet.segment_id,
            packet.is_final,
            packet.text,
            packet.start_time,
            packet.end_time,
        )
        if signature == self._last_packet_signature:
            return None
        self._last_packet_signature = signature
        if is_final:
            self._last_final_content_signature = final_content_signature
            self.segment_index += 1
        else:
            self._last_final_content_signature = None
        return packet

    def _response_to_dict(self, response: Any) -> dict[str, Any] | None:
        if hasattr(response, "to_dict"):
            try:
                response = response.to_dict()
            except Exception:
                return None
        if isinstance(response, dict):
            return response
        return None

    def _wav_header(self) -> bytes:
        channel_num = 1
        bits_per_sample = 16
        block_align = channel_num * bits_per_sample // 8
        byte_rate = self.sample_rate * block_align
        data_size = 0x22551000
        riff_size = data_size + 36
        return (
            b"RIFF"
            + struct.pack("<I", riff_size)
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, channel_num, self.sample_rate, byte_rate, block_align, bits_per_sample)
            + b"data"
            + struct.pack("<I", data_size)
        )


class InferenceWorker(QThread):
    ready = pyqtSignal()
    result_ready = pyqtSignal(object)
    job_failed = pyqtSignal(int, str)
    fatal_error = pyqtSignal(str)

    def __init__(self, factory: Callable[[], object]):
        super().__init__()
        self.factory = factory
        self._jobs: queue.Queue[InferenceJob | None] = queue.Queue(maxsize=1)
        self._stopping = False

    def submit(self, job: InferenceJob) -> None:
        if self._jobs.full():
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
        self._jobs.put_nowait(job)

    def stop(self) -> None:
        self._stopping = True
        try:
            if self._jobs.full():
                self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._jobs.put(None)
        self.wait(1500)

    def run(self) -> None:
        try:
            recognizer = self.factory()
        except Exception as exc:
            self.fatal_error.emit(str(exc))
            return

        self.ready.emit()

        while not self._stopping:
            job = self._jobs.get()
            if job is None:
                return

            started = time.perf_counter()
            try:
                prediction = recognizer.predict(job.values)
            except Exception as exc:
                self.job_failed.emit(job.generation, str(exc))
                continue

            self.result_ready.emit(
                InferenceOutput(
                    address=job.address,
                    prediction=prediction,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    sample_count=int(job.values.shape[0]),
                    generation=job.generation,
                )
            )


class GestureDetector:
    def __init__(self, model: InceptionTime, config: TouchpadRecognizerConfig, device: torch.device):
        self.device = device
        self.model = model
        self.config = config
        self.data_buffer: deque[np.ndarray] = deque()
        self.imu_window: Window[np.ndarray] = Window(config.gesture_window_size)
        self.cur_time_state = 0
        self.history_queue = deque(maxlen=config.gesture_history_size)

    def detect(self, data: np.ndarray) -> int:
        outputs = self.model(torch.from_numpy(data.T).float().view(1, 6, 20).to(self.device))
        pred = int(torch.argmax(outputs, dim=1).detach().cpu().item())
        self.history_queue.append(pred)

        if self.history_queue.count(0) >= self.config.threshold_down:
            self.history_queue.clear()
            return 4
        if self.history_queue.count(1) >= self.config.threshold_up:
            self.history_queue.clear()
            return 3
        return 0

    def calculate(self, data: np.ndarray) -> int:
        self.data_buffer.append(data)
        while self.data_buffer:
            self.imu_window.push(self.data_buffer.popleft())
        data_window = self.imu_window.map(lambda item: item).to_numpy().reshape(-1, 6)
        if data_window.shape[0] != self.config.gesture_window_size:
            return 0
        if self.cur_time_state < self.config.gesture_calculate_interval:
            self.cur_time_state += 1
            return 0
        self.cur_time_state = 0
        return self.detect(data_window)


class TouchpadRecognizerEngine:
    def __init__(self, config: TouchpadRecognizerConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gesture_state_dict = self._load_gesture_state_dict(config.gesture_ckpt)
        self.motion_state_dict = self._load_motion_state_dict(config.motion_ckpt)
        self.state: TouchpadModelState | None = None
        self.cursor_control_enabled = False
        self.mouse_controller = mouse.Controller() if HAS_TOUCHPAD_CURSOR_CONTROL else None

        monitors = []
        if HAS_TOUCHPAD_CURSOR_CONTROL and get_monitors is not None:
            try:
                monitors = get_monitors()
            except Exception:
                monitors = []
        if monitors:
            self.left = min(m.x for m in monitors)
            self.top = min(m.y for m in monitors)
            self.right = max(m.x + m.width for m in monitors)
            self.bottom = max(m.y + m.height for m in monitors)
        else:
            self.left = -10**9
            self.top = -10**9
            self.right = 10**9
            self.bottom = 10**9

    def clear(self) -> None:
        self.state = None

    def set_cursor_control_enabled(self, enabled: bool) -> None:
        self.cursor_control_enabled = bool(enabled) and self.mouse_controller is not None

    def process_imu(self, data: np.ndarray, event_ts: float) -> list[tuple[str, dict[str, Any], float]]:
        state = self._state()
        events: list[tuple[str, dict[str, Any], float]] = []

        linear_data = data.copy()
        if state.imu_processor.pre_timestamp == 0:
            state.imu_processor.init_quaternion_and_timestamp(data[:3], timestamp=1)
        else:
            state.imu_processor.update_attitude_by_data(data[3:6], data[:3])
        linear_data[:3] = state.imu_processor.calculate_linear_acceleration(data[:3])
        state.imu_data.append(linear_data)

        current_gesture = state.gesture_detector.calculate(linear_data)
        if current_gesture != 0:
            events.extend(self._handle_gesture_transition(state, current_gesture, event_ts))

        if len(state.imu_data) < state.imu_data.maxlen:
            return events

        merged_data = torch.tensor(np.array(state.imu_data), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = state.motion_model(merged_data.unsqueeze(0))
        outputs = output.detach().cpu().numpy()
        if outputs.shape != (1, 1, 2):
            raise RuntimeError(f"unexpected touchpad output shape: {outputs.shape}")

        result = outputs[0]
        if np.mean(np.abs(data[3:6])) < self.config.idle_gyro_threshold:
            result = np.array([[0.0, 0.0]], dtype=np.float32)

        if state.last_gesture == 4:
            if self.cursor_control_enabled:
                self._move_mouse(state, result)
            events.extend(self._build_move_events(state, result, event_ts))
        return events

    def _handle_gesture_transition(
        self,
        state: TouchpadModelState,
        current_gesture: int,
        event_ts: float,
    ) -> list[tuple[str, dict[str, Any], float]]:
        events: list[tuple[str, dict[str, Any], float]] = []
        now = time.time()
        if current_gesture == 4:
            state.last_down_time = now
            if state.last_gesture != 4:
                events.append(("touch-down", {}, event_ts))
        elif current_gesture == 3:
            if state.last_gesture != 3:
                events.append(("touch-up", {}, event_ts))
            if (
                self.cursor_control_enabled
                and self.mouse_controller is not None
                and state.last_gesture == 4
                and now - state.last_down_time < self.config.click_interval_seconds
            ):
                try:
                    self.mouse_controller.click(mouse.Button.left)
                except Exception:
                    pass
        state.last_gesture = current_gesture
        return events

    def _build_move_events(
        self,
        state: TouchpadModelState,
        outputs: np.ndarray,
        event_ts: float,
    ) -> list[tuple[str, dict[str, Any], float]]:
        events: list[tuple[str, dict[str, Any], float]] = []
        for output in outputs:
            vx = float(output[0]) / 10.0 * 9.0
            vy = float(output[1]) / 10.0 * 9.0
            velocity_norm = float(np.linalg.norm([vx, vy]))
            cd_ratio = state.cd_ratio_estimator.update(velocity_norm)
            vx = float(state.filter_x.filter(vx))
            vy = float(state.filter_y.filter(vy))
            vx *= cd_ratio
            vy *= cd_ratio

            now = time.perf_counter()
            if now - state.last_move_emit_time < self.config.move_emit_interval_seconds:
                continue
            if abs(vx) < self.config.min_move_abs and abs(vy) < self.config.min_move_abs:
                continue

            state.last_move_emit_time = now
            events.append(
                (
                    "move",
                    {
                        "move_x": vx,
                        "move_y": vy,
                    },
                    event_ts,
                )
            )
        return events

    def _move_mouse(self, state: TouchpadModelState, outputs: np.ndarray) -> None:
        if self.mouse_controller is None:
            return

        for output in outputs:
            vx = float(output[0]) / 10.0 * 9.0
            vy = float(output[1]) / 10.0 * 9.0
            velocity_norm = float(np.linalg.norm([vx, vy]))
            cd_ratio = state.cd_ratio_estimator.update(velocity_norm)
            vx = float(state.filter_x.filter(vx))
            vy = float(state.filter_y.filter(vy))
            vx *= cd_ratio
            vy *= cd_ratio

            cx, cy = self.mouse_controller.position
            if cx + vx < self.left:
                vx = self.left - cx
            elif cx + vx > self.right:
                vx = self.right - cx
            if cy + vy < self.top:
                vy = self.top - cy
            elif cy + vy > self.bottom:
                vy = self.bottom - cy

            try:
                self.mouse_controller.move(vx, vy)
            except Exception:
                return

    def _state(self) -> TouchpadModelState:
        if self.state is not None:
            return self.state

        gesture_model = InceptionTime(6, 4, nf=16).to(self.device)
        gesture_model.load_state_dict(self.gesture_state_dict)
        gesture_model.eval()

        motion_model = TCNLSTMStreaming(6, 128, 2, [64, 128], 3, 0.2).to(self.device)
        motion_model.load_state_dict(self.motion_state_dict)
        motion_model.eval()

        self.state = TouchpadModelState(
            gesture_detector=GestureDetector(gesture_model, self.config, self.device),
            motion_model=motion_model,
            imu_processor=IMUProcessor(frequency=200, core="VQF"),
            cd_ratio_estimator=CDRatioEstimator(min_cd=0.3, max_cd=2.0, c=0.2),
            filter_x=OneEuroFilter(freq=200, min_cutoff=0.0004, beta=0.08),
            filter_y=OneEuroFilter(freq=200, min_cutoff=0.0004, beta=0.08),
        )
        return self.state

    def _load_gesture_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(f"invalid gesture checkpoint: {checkpoint_path}")
        return checkpoint["model"]

    def _load_motion_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"invalid motion checkpoint: {checkpoint_path}")
        return checkpoint


class AirmouseRecognizerEngine:
    def __init__(self, config: AirmouseRecognizerConfig):
        if not HAS_AIRMOUSE_RUNTIME:
            raise RuntimeError(AIRMOUSE_RUNTIME_ERROR)
        self.config = config
        self.state = AirmouseState(
            imu_processor=IMUProcessor(frequency=200, core="VQF"),
            filter_x=OneEuroFilter(
                freq=200,
                min_cutoff=config.filter_min_cutoff,
                beta=config.filter_beta,
                d_cutoff=config.filter_d_cutoff,
            ),
            filter_y=OneEuroFilter(
                freq=200,
                min_cutoff=config.filter_min_cutoff,
                beta=config.filter_beta,
                d_cutoff=config.filter_d_cutoff,
            ),
        )

    def clear(self) -> None:
        self.state = AirmouseState(
            imu_processor=IMUProcessor(frequency=200, core="VQF"),
            filter_x=OneEuroFilter(
                freq=200,
                min_cutoff=self.config.filter_min_cutoff,
                beta=self.config.filter_beta,
                d_cutoff=self.config.filter_d_cutoff,
            ),
            filter_y=OneEuroFilter(
                freq=200,
                min_cutoff=self.config.filter_min_cutoff,
                beta=self.config.filter_beta,
                d_cutoff=self.config.filter_d_cutoff,
            ),
        )

    def process_imu(self, data: np.ndarray, event_ts: float) -> list[tuple[str, dict[str, Any], float]]:
        state = self.state
        timestamp_ms = event_ts * 1000.0
        if state.imu_processor.pre_timestamp == 0:
            state.imu_processor.init_quaternion_and_timestamp(data[:3], timestamp_ms)
            return []

        state.imu_processor.update_attitude_by_data(data[3:], data[:3], timestamp=timestamp_ms)
        quaternion = np.asarray(state.imu_processor.quaternion, dtype=np.float64)
        current_orientation = PyQuaternion(quaternion)
        state.orientation_queue.append(current_orientation)
        if len(state.orientation_queue) < state.orientation_queue.maxlen:
            return []

        delta_orientation = state.orientation_queue[0].inverse * current_orientation
        delta_orientation = np.array(delta_orientation.yaw_pitch_roll, dtype=np.float64)
        x_move, y_move = self._ring_to_world_motion(delta_orientation, current_orientation)
        x_move *= self.config.pointer_scale
        y_move *= self.config.pointer_scale
        vx = float(state.filter_x.filter(x_move))
        vy = float(state.filter_y.filter(y_move))
        state.orientation_queue.clear()
        state.orientation_queue.append(current_orientation)

        now = time.perf_counter()
        if now - state.last_move_emit_time < self.config.move_emit_interval_seconds:
            return []
        if abs(vx) < self.config.min_move_abs and abs(vy) < self.config.min_move_abs:
            return []

        state.last_move_emit_time = now
        return [("move", {"move_x": vx, "move_y": vy}, event_ts)]

    def _ring_to_world_motion(
        self,
        delta_orientation: np.ndarray,
        current_orientation: object,
    ) -> tuple[float, float]:
        ring_yaw_change, ring_pitch_change, _ring_roll_change = delta_orientation
        rotated_y = current_orientation.rotate(np.array([0.0, 1.0, 0.0]))
        rotated_z = current_orientation.rotate(np.array([0.0, 0.0, 1.0]))
        x_move = ring_yaw_change * rotated_z[2] + ring_pitch_change * rotated_y[2]
        y_move = ring_yaw_change * rotated_y[2] - ring_pitch_change * rotated_z[2]
        return -float(x_move), -float(y_move)


class RecognizerRuntimeBase:
    def __init__(self, name: str, bus: RecognizerBus, *, ring_role: str):
        self.name = name
        self.bus = bus
        self.ring_role = ring_role
        self.requested = False
        self.target: RingTarget | None = None
        self.status_text = "未启动"
        self._generation = 0
        self._activation_min_stream_seq = 0

    def activate(
        self,
        left_ring: RingTarget | None,
        right_ring: RingTarget | None,
        *,
        min_stream_seq: int = 0,
    ) -> None:
        self._generation += 1
        self._activation_min_stream_seq = max(0, int(min_stream_seq))
        self.requested = True
        self.target = self.resolve_target(left_ring, right_ring)
        if self.target is None:
            self._set_status(self.missing_target_text())
            return
        self.on_activate()

    def deactivate(self) -> None:
        self._generation += 1
        self._activation_min_stream_seq = 0
        self.requested = False
        self.target = None
        self.on_deactivate()
        self._set_status("未启动")

    def resolve_target(self, left_ring: RingTarget | None, right_ring: RingTarget | None) -> RingTarget | None:
        if self.ring_role == "left":
            return left_ring
        if self.ring_role == "both":
            return right_ring or left_ring
        return right_ring

    def missing_target_text(self) -> str:
        if self.ring_role == "left":
            return "缺少左手戒指"
        if self.ring_role == "both":
            return "缺少左右手戒指"
        return "缺少右手戒指"

    def on_activate(self) -> None:
        self._set_status("已启动")

    def on_deactivate(self) -> None:
        pass

    def handle_imu(self, address: str, data: np.ndarray, event_ts: float) -> None:
        _ = address
        _ = data
        _ = event_ts

    def handle_audio(self, message: dict[str, Any]) -> None:
        _ = message

    def current_fps(self) -> float:
        return 0.0

    def is_running(self) -> bool:
        return self.requested and self.target is not None

    def accepts_stream_seq(self, stream_seq: int) -> bool:
        if stream_seq <= 0:
            return True
        return stream_seq >= self._activation_min_stream_seq

    def close(self) -> None:
        pass

    def _set_status(self, text: str) -> None:
        if self.status_text == text:
            return
        self.status_text = text
        self.bus.recognizer_status.emit(self.name, text)


class WindowClassifierRuntime(RecognizerRuntimeBase):
    def __init__(
        self,
        name: str,
        bus: RecognizerBus,
        config: WindowClassifierConfig,
        factory: Callable[[], object],
        event_map: dict[int, str],
    ):
        super().__init__(name, bus, ring_role="right")
        self.config = config
        self.event_map = event_map
        self.state = ClassifierState()
        self.rate_tracker = RateTracker()
        self.worker_busy = False
        self.inflight_generation: int | None = None
        self.runtime_ready = False
        self.runtime_failed = False
        self.worker = InferenceWorker(factory)
        self.worker.ready.connect(self._on_worker_ready)
        self.worker.result_ready.connect(self._on_inference_result)
        self.worker.job_failed.connect(self._on_inference_failed)
        self.worker.fatal_error.connect(self._on_worker_fatal_error)
        self.worker.start()

    def on_activate(self) -> None:
        self._reset_state()
        if self.runtime_failed:
            self._set_status("模型不可用")
        elif self.runtime_ready:
            self._set_status("等待窗口积累")
        else:
            self._set_status("模型加载中")

    def on_deactivate(self) -> None:
        self._reset_state()
        self.rate_tracker.reset()
        self.worker_busy = False
        self.inflight_generation = None

    def handle_imu(self, address: str, data: np.ndarray, event_ts: float) -> None:
        _ = event_ts
        if not self.requested or self.target is None or address != self.target.address:
            return
        if self.runtime_failed:
            return

        current_sample = data.astype(np.float32, copy=False).tolist()
        self.state.samples.append(current_sample)
        self.state.frames_since_infer += 1
        self._advance_pending_reset(current_sample)
        self._maybe_submit_inference(address)

    def current_fps(self) -> float:
        if not self.requested:
            return 0.0
        return self.rate_tracker.fps()

    def is_running(self) -> bool:
        return self.requested and self.target is not None and self.runtime_ready and not self.runtime_failed

    def close(self) -> None:
        self.worker.stop()

    def _on_worker_ready(self) -> None:
        self.runtime_ready = True
        if self.requested:
            self._set_status("模型已就绪")

    def _on_worker_fatal_error(self, message: str) -> None:
        self.runtime_failed = True
        self.runtime_ready = False
        self.worker_busy = False
        self._set_status(f"模型失败: {message}")

    def _on_inference_failed(self, generation: int, message: str) -> None:
        if self.inflight_generation == generation:
            self.worker_busy = False
            self.inflight_generation = None
        if generation == self._generation and self.requested:
            self._set_status(f"推理失败: {message}")

    def _on_inference_result(self, result: InferenceOutput) -> None:
        if self.inflight_generation == result.generation:
            self.worker_busy = False
            self.inflight_generation = None
        if result.generation != self._generation:
            return
        if not self.requested or self.target is None or result.address != self.target.address:
            return

        self.rate_tracker.mark()
        class_id = int(result.prediction.class_id)
        triggered = self._update_gesture_state(class_id)
        if triggered is None:
            self._set_status(
                f"最近类别 {class_id} | 置信度 {float(result.prediction.confidence):.3f}"
            )
            self._maybe_submit_inference(result.address)
            return

        event_name = self.event_map.get(triggered)
        if event_name is None:
            self._maybe_submit_inference(result.address)
            return

        self.bus.recognizer_event.emit(
            self.name,
            event_name,
            self.target,
            {
                "class_id": triggered,
                "label": str(result.prediction.label),
                "confidence": float(result.prediction.confidence),
                "probabilities": list(result.prediction.probabilities),
            },
            time.time(),
        )
        self._set_status(f"触发 {event_name}")
        self._maybe_submit_inference(result.address)

    def _reset_state(self) -> None:
        self.state.samples.clear()
        self.state.frames_since_infer = 0
        self.state.prediction_history.clear()
        self.state.reset_after_frames_remaining = 0
        self.inflight_generation = None

    def _queue_length(self) -> int:
        return max(1, int(round(self.config.stable_window_seconds * IMU_SAMPLE_RATE_HZ / self.config.step_frames)))

    def _min_trigger_count(self, queue_length: int) -> int:
        return max(1, int(math.ceil(self.config.positive_ratio * queue_length)))

    def _advance_pending_reset(self, current_sample: list[float]) -> None:
        if self.state.reset_after_frames_remaining <= 0:
            return

        self.state.reset_after_frames_remaining -= 1
        if self.state.reset_after_frames_remaining > 0:
            return

        self.state.samples.clear()
        for _ in range(self.config.window_size // 2 - 1):
            self.state.samples.append(current_sample)
        self.state.frames_since_infer = 0
        self.state.prediction_history.clear()

    def _maybe_submit_inference(self, address: str) -> None:
        if self.worker_busy or self.runtime_failed or not self.runtime_ready:
            return
        if self.state.reset_after_frames_remaining > 0:
            return
        if len(self.state.samples) < self.config.window_size:
            return
        if self.state.frames_since_infer < self.config.step_frames:
            return

        values = np.asarray(list(self.state.samples)[-self.config.window_size :], dtype=np.float32)
        self.state.frames_since_infer = 0
        self.worker_busy = True
        self.inflight_generation = self._generation
        self.worker.submit(InferenceJob(address=address, values=values, generation=self._generation))

    def _update_gesture_state(self, class_id: int) -> int | None:
        history = self.state.prediction_history
        queue_length = self._queue_length()
        history.append(class_id)
        while len(history) > queue_length:
            history.popleft()
        if len(history) < queue_length:
            return None

        threshold = self._min_trigger_count(queue_length)
        for candidate in sorted(self.event_map.keys()):
            if history.count(candidate) < threshold:
                continue
            self.state.prediction_history.clear()
            self.state.frames_since_infer = 0
            self.state.reset_after_frames_remaining = self.config.reset_delay_frames
            return candidate
        return None


class SnapRuntime(RecognizerRuntimeBase):
    def __init__(self, bus: RecognizerBus, config: SnapRecognizerConfig):
        super().__init__(RECOGNIZER_SNAP, bus, ring_role="right")
        self.config = config
        self.current_infer_fps = 0.0
        self.worker = GestureInferenceWorker(
            GestureConfig(
                allowed_gestures=list(config.allowed_gestures),
                skipped_gestures=list(config.skipped_gestures),
                positive_ratio=float(config.positive_ratio),
                result_window_length=int(config.result_window_length),
                window_size=int(config.window_size),
                models=list(config.models),
                models_allowed_gestures=[list(item) for item in config.models_allowed_gestures],
                device=config.device,
                merge_window_seconds=float(config.merge_window_seconds),
            )
        )
        self.worker_ready = False
        self.worker_error = ""
        self.worker.sig_runtime.connect(self._on_runtime)
        self.worker.sig_result.connect(self._on_result)
        self.worker.sig_stats.connect(self._on_stats)
        self.worker.start()

    def on_activate(self) -> None:
        self.current_infer_fps = 0.0
        if self.worker_error:
            self._set_status(f"模型失败: {self.worker_error}")
            return
        self.worker.set_processing_mode(
            GestureInferenceWorker.MODE_TARGET,
            self.config.target_fps,
        )
        self.worker.clear_stream_state(None)
        if self.worker_ready:
            self._set_status(f"等待窗口积累（限速 {self.config.target_fps:.0f} FPS）")
        else:
            self._set_status("模型加载中")

    def on_deactivate(self) -> None:
        self.current_infer_fps = 0.0
        if self.worker_ready:
            self.worker.clear_stream_state(None)

    def handle_imu(self, address: str, data: np.ndarray, event_ts: float) -> None:
        if not self.requested or self.target is None or address != self.target.address or not self.worker_ready:
            return
        self.worker.enqueue_reading(address, data, event_ts)

    def current_fps(self) -> float:
        if not self.requested:
            return 0.0
        return self.current_infer_fps

    def is_running(self) -> bool:
        return self.requested and self.target is not None and self.worker_ready and not self.worker_error

    def close(self) -> None:
        self.worker.stop()
        self.worker.wait(1500)

    def _on_runtime(self, ready: bool, error_message: str) -> None:
        self.worker_ready = ready
        self.worker_error = error_message
        if self.requested:
            if error_message:
                self._set_status(f"模型失败: {error_message}")
            elif ready:
                self.worker.set_processing_mode(
                    GestureInferenceWorker.MODE_TARGET,
                    self.config.target_fps,
                )
                self._set_status(f"模型已就绪（限速 {self.config.target_fps:.0f} FPS）")

    def _on_result(self, address: str, _state_payload: dict[str, Any], code: int) -> None:
        if not self.requested or self.target is None or address != self.target.address:
            return
        if code != 16:
            return
        self.bus.recognizer_event.emit(self.name, "snap", self.target, {}, time.time())
        self._set_status("触发 snap")

    def _on_stats(self, payload: dict[str, Any]) -> None:
        self.current_infer_fps = float(payload.get("infer_fps", 0.0) or 0.0)


class ThreadedMotionRuntime(RecognizerRuntimeBase):
    def __init__(self, name: str, bus: RecognizerBus, *, ring_role: str, engine_factory: Callable[[], object]):
        super().__init__(name, bus, ring_role=ring_role)
        self.engine = None
        self.init_error = ""
        self.rate_tracker = RateTracker()
        self.pending: deque[PendingImuReading] = deque(maxlen=4096)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._alive = True

        try:
            self.engine = engine_factory()
        except Exception as exc:
            self.init_error = str(exc)

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def on_activate(self) -> None:
        self.rate_tracker.reset()
        self._clear_pending()
        if self.engine is not None:
            self.engine.clear()
        if self.init_error:
            self._set_status(f"初始化失败: {self.init_error}")
            return
        self._set_status("等待数据")

    def on_deactivate(self) -> None:
        self.rate_tracker.reset()
        self._clear_pending()
        if self.engine is not None:
            self.engine.clear()

    def handle_imu(self, address: str, data: np.ndarray, event_ts: float) -> None:
        if not self.requested or self.target is None or address != self.target.address or self.engine is None:
            return
        with self._condition:
            self.pending.append(
                PendingImuReading(
                    data=np.asarray(data, dtype=np.float32),
                    event_ts=event_ts,
                    generation=self._generation,
                )
            )
            self._condition.notify()

    def current_fps(self) -> float:
        if not self.requested:
            return 0.0
        return self.rate_tracker.fps()

    def is_running(self) -> bool:
        return self.requested and self.target is not None and self.engine is not None and not self.init_error

    def close(self) -> None:
        with self._condition:
            self._alive = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def set_cursor_control_enabled(self, enabled: bool) -> None:
        if self.engine is None:
            return
        setter = getattr(self.engine, "set_cursor_control_enabled", None)
        if callable(setter):
            setter(enabled)

    def _clear_pending(self) -> None:
        with self._condition:
            self.pending.clear()

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while self._alive and not self.pending:
                    self._condition.wait(timeout=0.2)
                if not self._alive:
                    return
                reading = self.pending.popleft()

            if self.engine is None:
                continue

            try:
                events = self.engine.process_imu(reading.data, reading.event_ts)
            except Exception as exc:
                self._set_status(f"运行失败: {exc}")
                continue

            with self._condition:
                if (
                    reading.generation != self._generation
                    or not self.requested
                    or self.target is None
                ):
                    continue
                target = self.target

            self.rate_tracker.mark()
            for event_name, data, event_ts in events:
                self.bus.recognizer_event.emit(self.name, event_name, target, data, event_ts)
                if event_name == "move":
                    self._set_status("持续输出位移")
                else:
                    self._set_status(f"触发 {event_name}")


class TypePlaceholderRuntime(RecognizerRuntimeBase):
    def __init__(self, bus: RecognizerBus):
        super().__init__(RECOGNIZER_TYPE, bus, ring_role="both")

    def on_activate(self) -> None:
        self._set_status("预留中，暂未接入")

    def is_running(self) -> bool:
        return False


class AudioPlaceholderRuntime(RecognizerRuntimeBase):
    def __init__(self, bus: RecognizerBus):
        super().__init__(RECOGNIZER_AUDIO, bus, ring_role="right")
        self.rate_tracker = RateTracker()
        self.decoder = AdpcmDecoder()
        self.capture_state = VoiceCaptureState()
        self.asr_workers: dict[str, DoubaoASRStreamWorker] = {}
        self.capture_targets: dict[str, RingTarget] = {}

    def on_activate(self) -> None:
        self.rate_tracker.reset()
        self._stop_all_asr_workers(wait_ms=1500)
        self.reset_audio_stream_state()
        self._reset_voice_capture(clear_capture_id=True)
        if self.target is None:
            self._set_status(self.missing_target_text())
            return
        self._set_status(self._idle_status_text())

    def on_deactivate(self) -> None:
        self.rate_tracker.reset()
        self._reset_voice_capture(clear_capture_id=True)
        self.reset_audio_stream_state()
        self._stop_all_asr_workers(wait_ms=1500)

    def handle_audio(self, message: dict[str, Any]) -> None:
        if not self.requested or self.target is None:
            return
        if str(message.get("address", "")) != self.target.address:
            return
        self.rate_tracker.mark()
        pcm_bytes = self._decode_audio_message(message)
        if pcm_bytes is None:
            return
        if not self.capture_state.active:
            return
        self.capture_state.audio_message_count += 1
        self._ensure_asr_worker(int(message.get("sample_rate", 16000) or 16000))
        worker = self.asr_workers.get(self.capture_state.capture_id)
        if worker is not None:
            worker.submit_pcm(pcm_bytes)

    def current_fps(self) -> float:
        if not self.requested:
            return 0.0
        return self.rate_tracker.fps()

    def is_voice_input_active(self) -> bool:
        return self.capture_state.active

    def begin_voice_input(self, *, started_at: float) -> bool:
        if not self.requested or self.target is None or self.capture_state.active:
            return False
        if not HAS_DOUBAO_ASR_RUNTIME:
            self._set_status(f"Doubao ASR 不可用: {DOUBAO_ASR_RUNTIME_ERROR}")
            return False
        capture_id = uuid.uuid4().hex
        self.capture_state.active = True
        self.capture_state.capture_id = capture_id
        self.capture_state.started_at = float(started_at)
        self.capture_state.audio_message_count = 0
        self.capture_targets[capture_id] = self.target
        self._set_status("语音输入中，等待实时转写结果")
        return True

    def finish_voice_input(self, *, finished_at: float) -> VoiceCaptureResult | None:
        if not self.capture_state.active:
            return None
        capture_id = self.capture_state.capture_id
        started_at = float(self.capture_state.started_at)
        result = VoiceCaptureResult(
            capture_id=capture_id,
            started_at=started_at,
            finished_at=float(finished_at),
            audio_message_count=int(self.capture_state.audio_message_count),
        )
        worker = self.asr_workers.get(capture_id)
        if worker is not None:
            worker.finish_stream()
        self._reset_voice_capture(clear_capture_id=True)
        if self.requested and self.target is not None:
            self._set_status(self._idle_status_text())
        return result

    def cancel_voice_input(self, *, finished_at: float) -> VoiceCaptureResult | None:
        if not self.capture_state.active:
            return None
        capture_id = self.capture_state.capture_id
        result = VoiceCaptureResult(
            capture_id=capture_id,
            started_at=float(self.capture_state.started_at),
            finished_at=float(finished_at),
            audio_message_count=int(self.capture_state.audio_message_count),
        )
        worker = self.asr_workers.get(capture_id)
        if worker is not None:
            worker.finish_stream()
        self._reset_voice_capture(clear_capture_id=True)
        if self.requested and self.target is not None:
            self._set_status(self._idle_status_text())
        return result

    def reset_audio_stream_state(self) -> None:
        self.decoder.reset()

    def close(self) -> None:
        self._stop_all_asr_workers(wait_ms=1500)

    def _reset_voice_capture(self, *, clear_capture_id: bool) -> None:
        self.capture_state.active = False
        if clear_capture_id:
            self.capture_state.capture_id = ""
        self.capture_state.started_at = 0.0
        self.capture_state.audio_message_count = 0

    def _idle_status_text(self) -> str:
        if not HAS_DOUBAO_ASR_RUNTIME:
            return f"Doubao ASR 不可用: {DOUBAO_ASR_RUNTIME_ERROR}"
        return "已接入音频流，等待 swipe-tap 开始语音输入"

    def _decode_audio_message(self, message: dict[str, Any]) -> bytes | None:
        audio_type = int(message.get("audio_type", -1) or -1)
        if audio_type != 1:
            return None
        data_b64 = message.get("data_b64", "")
        if not isinstance(data_b64, str) or not data_b64:
            return None
        try:
            raw_audio = base64.b64decode(data_b64)
        except Exception:
            return None
        if not raw_audio:
            return None
        return self.decoder.decode_packet(raw_audio)

    def _ensure_asr_worker(self, sample_rate: int) -> None:
        capture_id = self.capture_state.capture_id
        if not capture_id or capture_id in self.asr_workers:
            return
        worker = DoubaoASRStreamWorker(capture_id, sample_rate)
        worker.packet_ready.connect(self._on_asr_packet_ready)
        worker.fatal_error.connect(self._on_asr_fatal_error)
        worker.finished.connect(lambda cid=capture_id: self._on_asr_worker_finished(cid))
        self.asr_workers[capture_id] = worker
        worker.start()

    def _on_asr_packet_ready(self, packet: object) -> None:
        if not isinstance(packet, ASRStreamPacket):
            return
        target = self.capture_targets.get(packet.capture_id)
        if target is None:
            return
        self.bus.recognizer_event.emit(
            self.name,
            "text",
            target,
            {
                "capture_id": packet.capture_id,
                "text": packet.text,
                "result_type": "final" if packet.is_final else "partial",
                "is_final": packet.is_final,
                "segment_id": packet.segment_id,
                "start_time": packet.start_time,
                "end_time": packet.end_time,
                "engine": "doubao-asr-stream",
            },
            time.time(),
        )

    def _on_asr_fatal_error(self, capture_id: str, message: str) -> None:
        if capture_id == self.capture_state.capture_id:
            self._reset_voice_capture(clear_capture_id=True)
        self._set_status(f"Doubao ASR 失败: {message}")

    def _on_asr_worker_finished(self, capture_id: str) -> None:
        worker = self.asr_workers.pop(capture_id, None)
        if worker is not None:
            worker.deleteLater()
        if capture_id != self.capture_state.capture_id:
            self.capture_targets.pop(capture_id, None)

    def _stop_all_asr_workers(self, *, wait_ms: int) -> None:
        for worker in list(self.asr_workers.values()):
            worker.finish_stream()
        for capture_id, worker in list(self.asr_workers.items()):
            worker.wait(wait_ms)
            self.asr_workers.pop(capture_id, None)
            self.capture_targets.pop(capture_id, None)
            worker.deleteLater()


class LocalCursorController:
    def __init__(self):
        self.error = ""
        self.mouse_controller = None
        self.left = -10**9
        self.top = -10**9
        self.right = 10**9
        self.bottom = 10**9
        self.home_left = self.left
        self.home_top = self.top
        self.home_right = self.right
        self.home_bottom = self.bottom

        if not HAS_TOUCHPAD_CURSOR_CONTROL:
            self.error = TOUCHPAD_CURSOR_CONTROL_ERROR or "pynput 不可用"
            return

        try:
            self.mouse_controller = mouse.Controller()
            if get_monitors is not None:
                monitors = get_monitors()
                if monitors:
                    self.left = min(item.x for item in monitors)
                    self.top = min(item.y for item in monitors)
                    self.right = max(item.x + item.width for item in monitors)
                    self.bottom = max(item.y + item.height for item in monitors)
                    home_monitor = next(
                        (item for item in monitors if bool(getattr(item, "is_primary", False))),
                        monitors[0],
                    )
                    self.home_left = home_monitor.x
                    self.home_top = home_monitor.y
                    self.home_right = home_monitor.x + home_monitor.width
                    self.home_bottom = home_monitor.y + home_monitor.height
        except Exception as exc:
            self.error = str(exc)
            self.mouse_controller = None

    def available(self) -> bool:
        return self.mouse_controller is not None and not self.error

    def move_to(self, target_x: float, target_y: float) -> bool:
        if self.mouse_controller is None:
            return False

        target_x = min(max(float(target_x), self.left), self.right)
        target_y = min(max(float(target_y), self.top), self.bottom)
        try:
            self.mouse_controller.position = (target_x, target_y)
        except Exception:
            return False
        return True

    def move_to_home_ratio(self, x_ratio: float, y_ratio: float) -> bool:
        width = max(1.0, float(self.home_right - self.home_left))
        height = max(1.0, float(self.home_bottom - self.home_top))
        target_x = self.home_left + width * float(np.clip(x_ratio, 0.0, 1.0))
        target_y = self.home_top + height * float(np.clip(y_ratio, 0.0, 1.0))
        return self.move_to(target_x, target_y)

    def move_by(self, vx: float, vy: float) -> bool:
        if self.mouse_controller is None:
            return False
        if vx == 0.0 and vy == 0.0:
            return False

        try:
            cx, cy = self.mouse_controller.position
            target_x = min(max(cx + vx, self.left), self.right)
            target_y = min(max(cy + vy, self.top), self.bottom)
            self.mouse_controller.position = (target_x, target_y)
        except Exception:
            return False
        return True


class UnifiedGestureWindow(RingStreamWindowBase):
    RECOGNIZER_TITLES = {
        RECOGNIZER_SWIPE: "Swipe",
        RECOGNIZER_PINCH: "Pinch",
        RECOGNIZER_SNAP: "Snap",
        RECOGNIZER_TOUCHPAD: "Touchpad",
        RECOGNIZER_AIRMOUSE: "Airmouse",
        RECOGNIZER_TYPE: "Type",
        RECOGNIZER_AUDIO: "Audio",
    }
    DEFAULT_CHECKED = {
        RECOGNIZER_SWIPE: True,
        RECOGNIZER_PINCH: True,
        RECOGNIZER_SNAP: True,
        RECOGNIZER_TOUCHPAD: False,
        RECOGNIZER_AIRMOUSE: False,
        RECOGNIZER_TYPE: False,
        RECOGNIZER_AUDIO: False,
    }

    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port)
        self.setWindowTitle("综合手势识别")
        self.resize(1180, 860)

        self.bus = RecognizerBus()
        self.bus.recognizer_event.connect(self.on_recognizer_event)
        self.bus.recognizer_status.connect(self.on_recognizer_status)

        self.sequence = 0
        self.stream_connected = False
        self.subscription_applied = False
        self.last_status_broadcast_at = 0.0
        self._stream_seq = 0
        self._stream_seq_lock = threading.Lock()

        self.left_ring_rate = RateTracker()
        self.right_ring_rate = RateTracker()
        self.left_audio_rate = RateTracker()
        self.right_audio_rate = RateTracker()

        self.stream_status_label = QLabel("数据流状态：未连接")
        self.stream_status_label.setObjectName("statusTag")
        self.subscription_status_label = QLabel("未订阅")
        self.subscription_status_label.setObjectName("statusTag")
        self.websocket_status_label = QLabel("WebSocket：初始化中")
        self.websocket_status_label.setObjectName("statusTag")

        self.left_combo = QComboBox()
        self.right_combo = QComboBox()
        self.left_fps_label = QLabel("左手帧率：0.0 Hz")
        self.right_fps_label = QLabel("右手帧率：0.0 Hz")
        self.touchpad_cursor_checkbox = QCheckBox("Touchpad 控制本地光标")
        self.touchpad_cursor_checkbox.setChecked(False)
        self.pinch_airmouse_cursor_checkbox = QCheckBox("Pinch/Airmouse 控制本地光标")
        self.pinch_airmouse_cursor_checkbox.setChecked(False)
        self.macos_keepalive_checkbox = QCheckBox("macOS 后台保活")
        self.macos_keepalive_checkbox.setChecked(True)

        self.btn_subscribe = QPushButton("订阅")
        self.btn_subscribe.setProperty("class", "btn-primary")
        self.btn_start = QPushButton("启动")
        self.btn_start.setProperty("class", "btn-success")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setProperty("class", "btn-danger")

        self.message_log = QPlainTextEdit()
        self.message_log.setReadOnly(True)
        self.message_log.document().setMaximumBlockCount(APP_CONFIG.message_log_limit)

        self.recognizer_rows: dict[str, RecognizerRowWidgets] = {}
        self.pinch_airmouse_move_active = False
        self.local_cursor_controller = LocalCursorController()
        self.macos_keepalive = MacOSKeepAliveManager("Unified Gesture 后台识别保活")
        self.runtimes = self._build_runtimes()

        self._build_ui()
        self.set_stream_status_label(self.stream_status_label)
        self.client.add_message_hook(self._annotate_stream_message)
        self.bind_stream_client_signals()

        self.ws_bridge = FrontendWebSocketBridge(APP_CONFIG.websocket_host, APP_CONFIG.websocket_port)
        self.ws_bridge.command_received.connect(self.on_ws_command)
        self.ws_bridge.state_changed.connect(self.refresh_websocket_status)
        if APP_CONFIG.websocket_enabled:
            self.ws_bridge.start()
        else:
            self.refresh_websocket_status()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_metrics)
        self.refresh_timer.start(250)

        self._configure_macos_keepalive_ui()
        self.client.start()

    def _build_runtimes(self) -> dict[str, RecognizerRuntimeBase]:
        return {
            RECOGNIZER_SWIPE: WindowClassifierRuntime(
                RECOGNIZER_SWIPE,
                self.bus,
                APP_CONFIG.swipe,
                lambda: SwipeCNNRecognizer(
                    APP_CONFIG.swipe.checkpoint_dir,
                    device="cpu",
                    backend=APP_CONFIG.swipe.backend,
                ),
                {
                    1: "swipe-up",
                    2: "swipe-down",
                    3: "swipe-left",
                    4: "swipe-right",
                    5: "tap",
                },
            ),
            RECOGNIZER_PINCH: WindowClassifierRuntime(
                RECOGNIZER_PINCH,
                self.bus,
                APP_CONFIG.pinch,
                lambda: PinchCNNRecognizer(
                    APP_CONFIG.pinch.checkpoint_dir,
                    device="cpu",
                    backend=APP_CONFIG.pinch.backend,
                ),
                {
                    1: "pinch-down",
                    2: "pinch-up",
                },
            ),
            RECOGNIZER_SNAP: SnapRuntime(self.bus, APP_CONFIG.snap),
            RECOGNIZER_TOUCHPAD: ThreadedMotionRuntime(
                RECOGNIZER_TOUCHPAD,
                self.bus,
                ring_role="right",
                engine_factory=lambda: TouchpadRecognizerEngine(APP_CONFIG.touchpad),
            ),
            RECOGNIZER_AIRMOUSE: ThreadedMotionRuntime(
                RECOGNIZER_AIRMOUSE,
                self.bus,
                ring_role="right",
                engine_factory=lambda: AirmouseRecognizerEngine(APP_CONFIG.airmouse),
            ),
            RECOGNIZER_TYPE: TypePlaceholderRuntime(self.bus),
            RECOGNIZER_AUDIO: AudioPlaceholderRuntime(self.bus),
        }

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        hidden_host = QFrame()
        hidden_host.setVisible(False)
        hidden_layout = QVBoxLayout(hidden_host)
        hidden_layout.setContentsMargins(0, 0, 0, 0)
        self.add_ring_selection_panel(hidden_layout, title_text="hidden", max_height=1)
        root.addWidget(hidden_host)

        header = QFrame()
        header.setProperty("class", "panel")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(8)

        title = QLabel("综合识别应用")
        title.setProperty("class", "title")
        desc = QLabel(
            "统一订阅左右手戒指，按需启动 Swipe / Pinch / Snap / Touchpad / Airmouse / Type / Audio。"
        )
        desc.setProperty("class", "sub-text")
        desc.setWordWrap(True)

        header_layout.addWidget(title)
        header_layout.addWidget(desc)
        header_layout.addWidget(self.stream_status_label)
        header_layout.addWidget(self.subscription_status_label)
        header_layout.addWidget(self.websocket_status_label)
        root.addWidget(header)

        ring_panel = QFrame()
        ring_panel.setProperty("class", "panel")
        ring_layout = QGridLayout(ring_panel)
        ring_layout.setContentsMargins(16, 14, 16, 14)
        ring_layout.setHorizontalSpacing(12)
        ring_layout.setVerticalSpacing(10)

        ring_layout.addWidget(QLabel("左手戒指"), 0, 0)
        ring_layout.addWidget(self.left_combo, 0, 1)
        ring_layout.addWidget(self.left_fps_label, 0, 2)

        ring_layout.addWidget(QLabel("右手戒指"), 1, 0)
        ring_layout.addWidget(self.right_combo, 1, 1)
        ring_layout.addWidget(self.right_fps_label, 1, 2)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.btn_subscribe)
        action_row.addWidget(self.btn_start)
        action_row.addWidget(self.btn_stop)
        action_row.addStretch()
        ring_layout.addLayout(action_row, 2, 0, 1, 3)

        if self.local_cursor_controller.available():
            self.touchpad_cursor_checkbox.setToolTip("启用后，Touchpad 手势会驱动本地鼠标光标")
            self.pinch_airmouse_cursor_checkbox.setToolTip(
                "启用后，Pinch 启动或 pinch-up 时会将光标移动到配置点；仅在 pinch-down 到 pinch-up 期间，Airmouse 输出 move 会驱动本地鼠标光标"
            )
        else:
            cursor_error = self.local_cursor_controller.error or "unknown error"
            self.touchpad_cursor_checkbox.setEnabled(False)
            self.touchpad_cursor_checkbox.setToolTip(f"本地光标控制不可用：{cursor_error}")
            self.pinch_airmouse_cursor_checkbox.setEnabled(False)
            self.pinch_airmouse_cursor_checkbox.setToolTip(f"本地光标控制不可用：{cursor_error}")
        ring_layout.addWidget(self.touchpad_cursor_checkbox, 3, 0, 1, 3)
        ring_layout.addWidget(self.pinch_airmouse_cursor_checkbox, 4, 0, 1, 3)
        ring_layout.addWidget(self.macos_keepalive_checkbox, 5, 0, 1, 3)
        root.addWidget(ring_panel)

        recognizer_panel = QFrame()
        recognizer_panel.setProperty("class", "panel")
        recognizer_layout = QVBoxLayout(recognizer_panel)
        recognizer_layout.setContentsMargins(16, 14, 16, 14)
        recognizer_layout.setSpacing(10)
        recognizer_layout.addWidget(QLabel("识别器"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.addWidget(QLabel("启用"), 0, 0)
        grid.addWidget(QLabel("数据来源"), 0, 1)
        grid.addWidget(QLabel("运行帧率"), 0, 2)
        grid.addWidget(QLabel("状态"), 0, 3)

        for row_index, recognizer in enumerate(ALL_RECOGNIZERS, start=1):
            checkbox = QCheckBox(self.RECOGNIZER_TITLES[recognizer])
            checkbox.setChecked(self.DEFAULT_CHECKED.get(recognizer, False))
            target_label = QLabel(self._recognizer_target_text(recognizer))
            fps_label = QLabel("0.0 Hz")
            status_label = QLabel("未启动")
            status_label.setWordWrap(True)
            self.recognizer_rows[recognizer] = RecognizerRowWidgets(
                checkbox=checkbox,
                target_label=target_label,
                fps_label=fps_label,
                status_label=status_label,
            )
            grid.addWidget(checkbox, row_index, 0)
            grid.addWidget(target_label, row_index, 1)
            grid.addWidget(fps_label, row_index, 2)
            grid.addWidget(status_label, row_index, 3)

        recognizer_layout.addLayout(grid)
        root.addWidget(recognizer_panel)

        message_panel = QFrame()
        message_panel.setProperty("class", "panel")
        message_layout = QVBoxLayout(message_panel)
        message_layout.setContentsMargins(16, 14, 16, 14)
        message_layout.setSpacing(8)
        message_title = QLabel("Socket 发送日志")
        message_title.setProperty("class", "title")
        message_desc = QLabel("只显示实际经 WebSocket 发出的 payload，并标注广播目标数量。")
        message_desc.setProperty("class", "sub-text")
        message_layout.addWidget(message_title)
        message_layout.addWidget(message_desc)
        message_layout.addWidget(self.message_log, 1)
        root.addWidget(message_panel, 1)

        self.btn_subscribe.clicked.connect(self.apply_subscription)
        self.btn_start.clicked.connect(self.start_selected_recognizers)
        self.btn_stop.clicked.connect(self.stop_all_recognizers)
        self.left_combo.currentIndexChanged.connect(self.refresh_target_labels)
        self.right_combo.currentIndexChanged.connect(self.refresh_target_labels)
        self.touchpad_cursor_checkbox.toggled.connect(self.on_touchpad_cursor_toggled)
        self.pinch_airmouse_cursor_checkbox.toggled.connect(self.on_pinch_airmouse_cursor_toggled)
        self.macos_keepalive_checkbox.toggled.connect(self.on_macos_keepalive_toggled)

    def on_connected_changed(self, connected: bool) -> None:
        self.stream_connected = connected
        super().on_connected_changed(connected)
        if not connected:
            self._cancel_voice_input_if_needed(timestamp=time.time())
            audio_runtime = self._audio_runtime()
            if audio_runtime is not None:
                audio_runtime.reset_audio_stream_state()
            self.left_ring_rate.reset()
            self.right_ring_rate.reset()
            self.left_audio_rate.reset()
            self.right_audio_rate.reset()
        self.publish_runtime_status()

    def on_stream_connected(self) -> None:
        self.refresh_target_labels()

    def handle_stream_message(self, message: dict) -> None:
        message_type = str(message.get("type", ""))
        if message_type == "subscribed":
            rings = list(message.get("rings", []))
            streams = list(message.get("streams", []))
            self.subscription_applied = True
            self.subscription_status_label.setText(
                f"已订阅 | rings={rings or ['all']} | streams={streams or ['imu']}"
            )
            return

        if message_type == "audio":
            address = str(message.get("address", ""))
            left_ring = self.selected_ring("left")
            right_ring = self.selected_ring("right")
            if left_ring is not None and address == left_ring.address:
                self.left_audio_rate.mark()
            if right_ring is not None and address == right_ring.address:
                self.right_audio_rate.mark()
            self.runtimes[RECOGNIZER_AUDIO].handle_audio(message)
            return

        if message_type != "imu":
            return

        address = str(message.get("address", ""))
        acc_raw = message.get("acc")
        gyr_raw = message.get("gyr")
        if not address or not isinstance(acc_raw, list) or not isinstance(gyr_raw, list):
            return
        if len(acc_raw) != 3 or len(gyr_raw) != 3:
            return

        event_ts = self._message_timestamp(message)
        stream_seq = int(message.get(STREAM_SEQUENCE_KEY, 0) or 0)
        data = np.asarray(
            [
                float(acc_raw[0]),
                float(acc_raw[1]),
                float(acc_raw[2]),
                float(gyr_raw[0]),
                float(gyr_raw[1]),
                float(gyr_raw[2]),
            ],
            dtype=np.float32,
        )

        left_ring = self.selected_ring("left")
        right_ring = self.selected_ring("right")
        if left_ring is not None and address == left_ring.address:
            self.left_ring_rate.mark()
        if right_ring is not None and address == right_ring.address:
            self.right_ring_rate.mark()

        for runtime in self.runtimes.values():
            if not runtime.accepts_stream_seq(stream_seq):
                continue
            runtime.handle_imu(address, data, event_ts)

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> None:
        _ = rings
        self._refresh_ring_combos()
        if removed_addresses:
            left_ring = self.selected_ring("left")
            right_ring = self.selected_ring("right")
            if left_ring is not None and left_ring.address in removed_addresses:
                self.left_ring_rate.reset()
                self.left_audio_rate.reset()
            if right_ring is not None and right_ring.address in removed_addresses:
                self.right_ring_rate.reset()
                self.right_audio_rate.reset()
        if self._voice_input_invalidated(removed_addresses, changed_connection_addresses):
            self._cancel_voice_input_if_needed(timestamp=time.time())
            audio_runtime = self._audio_runtime()
            if audio_runtime is not None:
                audio_runtime.reset_audio_stream_state()
        self.refresh_target_labels()
        self.publish_runtime_status()

    def apply_subscription(self) -> None:
        validation_error = self.validate_ring_selection()
        if validation_error is not None:
            self.subscription_status_label.setText(validation_error)
            self.publish_runtime_status()
            return

        rings = self.subscription_ring_addresses()
        streams = self.current_streams()
        self.client.send_command(
            {
                "cmd": "subscribe",
                "rings": rings if rings else [NONE_SUBSCRIPTION],
                "streams": streams,
            }
        )
        self.subscription_status_label.setText("订阅请求已发送")
        self.left_ring_rate.reset()
        self.right_ring_rate.reset()
        self.left_audio_rate.reset()
        self.right_audio_rate.reset()
        self.publish_runtime_status()

    def start_selected_recognizers(self) -> None:
        self.apply_recognizer_selection()

    def apply_recognizer_selection(self) -> None:
        validation_error = self.validate_ring_selection(require_right_for_active=True)
        if validation_error is not None:
            self.subscription_status_label.setText(validation_error)
            self.publish_runtime_status()
            return

        self._cancel_voice_input_if_needed(timestamp=time.time())
        self._sync_main_audio_stream_mode()
        self.apply_subscription()
        left_ring = self.selected_ring("left")
        right_ring = self.selected_ring("right")
        min_stream_seq = self._current_stream_seq() + 1

        for recognizer, runtime in self.runtimes.items():
            if self.recognizer_rows[recognizer].checkbox.isChecked():
                runtime.activate(left_ring, right_ring, min_stream_seq=min_stream_seq)
            else:
                runtime.deactivate()

        self._sync_pinch_airmouse_cursor_control(reset_home=True)
        self.refresh_target_labels()

    def stop_all_recognizers(self) -> None:
        self._cancel_voice_input_if_needed(timestamp=time.time())
        self._request_main_audio_stream(False)
        self.pinch_airmouse_move_active = False
        for runtime in self.runtimes.values():
            runtime.deactivate()
        self.publish_runtime_status()

    def on_recognizer_event(
        self,
        recognizer: str,
        event_name: str,
        ring: RingTarget,
        data: dict[str, Any],
        event_ts: float,
    ) -> None:
        _ = ring
        if recognizer == RECOGNIZER_PINCH:
            self._handle_pinch_airmouse_event(event_name)
        if recognizer == RECOGNIZER_AIRMOUSE and event_name == "move":
            self._apply_airmouse_cursor_move(data)
        if event_name == "move":
            return
        payload = build_recognizer_event(
            RecognizerEventMessage(
                recognizer=recognizer,
                event=event_name,
                timestamp=event_ts,
                data=dict(data),
                sequence=self._next_sequence(),
            )
        )
        self._publish_logged_payload(payload)
        if recognizer == RECOGNIZER_SWIPE and event_name == "tap":
            self._toggle_voice_input_from_swipe_tap(event_ts)

    def on_recognizer_status(self, recognizer: str, status: str) -> None:
        row = self.recognizer_rows.get(recognizer)
        if row is not None:
            row.status_label.setText(status)

    def refresh_metrics(self) -> None:
        self.left_fps_label.setText(f"左手帧率：{self.left_ring_rate.fps():.1f} Hz")
        self.right_fps_label.setText(f"右手帧率：{self.right_ring_rate.fps():.1f} Hz")
        for recognizer, runtime in self.runtimes.items():
            row = self.recognizer_rows[recognizer]
            row.fps_label.setText(f"{runtime.current_fps():.1f} Hz")
            row.status_label.setText(runtime.status_text)
        self.refresh_websocket_status()

    def refresh_websocket_status(self) -> None:
        if not APP_CONFIG.websocket_enabled:
            self.websocket_status_label.setText("WebSocket：已关闭")
            return
        if self.ws_bridge.last_error:
            self.websocket_status_label.setText(f"WebSocket：异常（{self.ws_bridge.last_error}）")
            return
        if self.ws_bridge.listening:
            self.websocket_status_label.setText(
                f"WebSocket：监听 {self.ws_bridge.url} | clients={self.ws_bridge.client_count()}"
            )
            return
        self.websocket_status_label.setText("WebSocket：未监听")

    def publish_runtime_status(self, client: object | None = None, *, broadcast: bool = False) -> None:
        payload = build_runtime_status(
            RuntimeStatusMessage(
                left=RingStatus(
                    connected=self.selected_ring_connected("left"),
                    imu_fps=self.left_ring_rate.fps(),
                    audio_fps=self.left_audio_rate.fps(),
                ),
                right=RingStatus(
                    connected=self.selected_ring_connected("right"),
                    imu_fps=self.right_ring_rate.fps(),
                    audio_fps=self.right_audio_rate.fps(),
                ),
            )
        )
        if client is not None:
            self._send_payload(payload, client=client, log=True)
            return
        if broadcast:
            self._send_payload(payload, log=True)

    def on_ws_command(self, command: ControlCommandMessage, client: object) -> None:
        if command.action == CONTROL_ACTION_REQUEST_STATUS:
            self.publish_runtime_status(client=client)
            self._publish_current_voice_status(client=client)
            return

        try:
            self.apply_control_command(command)
        except Exception as exc:
            self._send_payload(
                build_error_message("command_failed", str(exc), payload={"action": command.action}),
                client=client,
                log=True,
            )
            return

        if command.action == CONTROL_ACTION_SET_RECOGNIZERS:
            self._send_payload(
                build_ack_message(
                    command.action,
                    ok=True,
                    detail="ok",
                    payload={
                        "recognizers": list(self.checked_recognizers()),
                    },
                ),
                client=client,
                log=True,
            )
            return

        self._send_payload(
            build_error_message("unsupported_action", f"unsupported action: {command.action}"),
            client=client,
            log=True,
        )

    def apply_control_command(self, command: ControlCommandMessage) -> None:
        if command.action == CONTROL_ACTION_REQUEST_STATUS:
            return
        if command.action == CONTROL_ACTION_SET_RECOGNIZERS:
            requested = set(command.recognizers)
            for recognizer, row in self.recognizer_rows.items():
                row.checkbox.setChecked(recognizer in requested)
            self.apply_recognizer_selection()
            return
        raise ValueError(f"unsupported action: {command.action}")

    def validate_ring_selection(self, *, require_right_for_active: bool = False) -> str | None:
        left_address = self.selected_ring_address("left")
        right_address = self.selected_ring_address("right")
        if left_address and right_address and left_address == right_address:
            return "左右手不能选择同一枚戒指"
        if require_right_for_active:
            active = set(self.checked_recognizers())
            needs_right = active - {RECOGNIZER_TYPE}
            if needs_right and not right_address:
                return "除 Type 外，其余识别器都需要选择右手戒指"
        return None

    def on_touchpad_cursor_toggled(self, checked: bool) -> None:
        touchpad_runtime = self.runtimes.get(RECOGNIZER_TOUCHPAD)
        if isinstance(touchpad_runtime, ThreadedMotionRuntime):
            touchpad_runtime.set_cursor_control_enabled(checked)
        if checked and not self.local_cursor_controller.available():
            self.subscription_status_label.setText(
                f"Touchpad 光标控制不可用：{self.local_cursor_controller.error or 'unknown error'}"
            )
        self.publish_runtime_status()

    def on_pinch_airmouse_cursor_toggled(self, checked: bool) -> None:
        if checked and not self.local_cursor_controller.available():
            self.subscription_status_label.setText(
                f"Pinch/Airmouse 光标控制不可用：{self.local_cursor_controller.error or 'unknown error'}"
            )
        elif checked:
            self._sync_pinch_airmouse_cursor_control(reset_home=True)
        else:
            self.pinch_airmouse_move_active = False
        self.publish_runtime_status()

    def _configure_macos_keepalive_ui(self) -> None:
        if not self.macos_keepalive.available:
            self.macos_keepalive_checkbox.setChecked(False)
            self.macos_keepalive_checkbox.setEnabled(False)
            self.macos_keepalive_checkbox.setToolTip(
                f"后台保活不可用：{self.macos_keepalive.error or 'unknown error'}"
            )
            return
        self.macos_keepalive_checkbox.setToolTip("启用后会在 macOS 上申请前台活动，尽量避免后台 App Nap 节流。")
        self._apply_macos_keepalive(True)

    def on_macos_keepalive_toggled(self, checked: bool) -> None:
        self._apply_macos_keepalive(checked)

    def _apply_macos_keepalive(self, enabled: bool) -> None:
        active = self.macos_keepalive.set_enabled(enabled)
        if enabled and not active:
            self.subscription_status_label.setText(
                f"macOS 后台保活开启失败：{self.macos_keepalive.error or 'unknown error'}"
            )
            if self.macos_keepalive_checkbox.isChecked():
                self.macos_keepalive_checkbox.blockSignals(True)
                self.macos_keepalive_checkbox.setChecked(False)
                self.macos_keepalive_checkbox.blockSignals(False)
            return
        state = "已开启" if active else "已关闭"
        self.macos_keepalive_checkbox.setToolTip(
            "启用后会在 macOS 上申请前台活动，尽量避免后台 App Nap 节流。"
            if self.macos_keepalive.available
            else f"后台保活不可用：{self.macos_keepalive.error or 'unknown error'}"
        )
        self.subscription_status_label.setText(f"macOS 后台保活{state}")

    def _pinch_airmouse_cursor_enabled(self) -> bool:
        return self.pinch_airmouse_cursor_checkbox.isChecked() and self.local_cursor_controller.available()

    def _recognizer_requested(self, recognizer: str) -> bool:
        runtime = self.runtimes.get(recognizer)
        return runtime is not None and runtime.requested and runtime.target is not None

    def _reset_pinch_airmouse_cursor(self) -> None:
        if not self._pinch_airmouse_cursor_enabled():
            return
        self.local_cursor_controller.move_to_home_ratio(
            APP_CONFIG.airmouse.cursor_home_x_ratio,
            APP_CONFIG.airmouse.cursor_home_y_ratio,
        )

    def _sync_pinch_airmouse_cursor_control(self, *, reset_home: bool = False) -> None:
        self.pinch_airmouse_move_active = False
        if not self._pinch_airmouse_cursor_enabled():
            return
        if reset_home and self._recognizer_requested(RECOGNIZER_PINCH):
            self._reset_pinch_airmouse_cursor()

    def _handle_pinch_airmouse_event(self, event_name: str) -> None:
        if event_name == "pinch-down":
            self.pinch_airmouse_move_active = (
                self._pinch_airmouse_cursor_enabled()
                and self._recognizer_requested(RECOGNIZER_PINCH)
                and self._recognizer_requested(RECOGNIZER_AIRMOUSE)
            )
            return
        if event_name == "pinch-up":
            self.pinch_airmouse_move_active = False
            self._reset_pinch_airmouse_cursor()

    def _apply_airmouse_cursor_move(self, data: dict[str, Any]) -> None:
        if not self._pinch_airmouse_cursor_enabled():
            return
        if not self._recognizer_requested(RECOGNIZER_AIRMOUSE):
            return
        if not self.pinch_airmouse_move_active:
            return

        vx = float(data.get("move_x", 0.0) or 0.0)
        vy = float(data.get("move_y", 0.0) or 0.0)
        self.local_cursor_controller.move_by(vx, vy)

    def _audio_runtime(self) -> AudioPlaceholderRuntime | None:
        runtime = self.runtimes.get(RECOGNIZER_AUDIO)
        if isinstance(runtime, AudioPlaceholderRuntime):
            return runtime
        return None

    def _swipe_audio_toggle_enabled(self) -> bool:
        return self._recognizer_requested(RECOGNIZER_SWIPE) and self._recognizer_requested(RECOGNIZER_AUDIO)

    def _toggle_voice_input_from_swipe_tap(self, event_ts: float) -> None:
        if not self._swipe_audio_toggle_enabled():
            return
        audio_runtime = self._audio_runtime()
        if audio_runtime is None:
            return
        if audio_runtime.is_voice_input_active():
            result = audio_runtime.finish_voice_input(finished_at=event_ts)
            if result is None:
                return
            self._publish_voice_status(
                VOICE_STATE_FINISHED,
                event_ts,
                data={
                    "capture_id": result.capture_id,
                    "audio_message_count": result.audio_message_count,
                },
            )
            return
        if audio_runtime.begin_voice_input(started_at=event_ts):
            self._publish_voice_status(
                VOICE_STATE_LISTENING,
                event_ts,
                data={
                    "capture_id": audio_runtime.capture_state.capture_id,
                },
            )

    def _cancel_voice_input_if_needed(self, *, timestamp: float) -> None:
        audio_runtime = self._audio_runtime()
        if audio_runtime is None:
            return
        result = audio_runtime.cancel_voice_input(finished_at=timestamp)
        if result is not None:
            self._publish_voice_status(
                VOICE_STATE_FINISHED,
                timestamp,
                data={
                    "capture_id": result.capture_id,
                    "audio_message_count": result.audio_message_count,
                },
            )

    def _voice_input_invalidated(
        self,
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> bool:
        audio_runtime = self._audio_runtime()
        if audio_runtime is None or not audio_runtime.is_voice_input_active():
            return False

        watched_addresses: set[str] = set()
        for recognizer in (RECOGNIZER_AUDIO, RECOGNIZER_SWIPE):
            runtime = self.runtimes.get(recognizer)
            if runtime is not None and runtime.target is not None:
                watched_addresses.add(runtime.target.address)

        removed_set = set(removed_addresses)
        if watched_addresses & removed_set:
            return True

        changed_set = set(changed_connection_addresses)
        for address in watched_addresses & changed_set:
            ring = self.ring_info.get(address, {})
            if not bool(ring.get("connected", False)):
                return True
        return False

    def _publish_current_voice_status(self, *, client: object) -> None:
        audio_runtime = self._audio_runtime()
        if audio_runtime is None or not audio_runtime.is_voice_input_active():
            return
        started_at = float(audio_runtime.capture_state.started_at or time.time())
        self._publish_voice_status(
            VOICE_STATE_LISTENING,
            started_at,
            data={
                "capture_id": audio_runtime.capture_state.capture_id,
            },
            client=client,
            log=True,
        )

    def _publish_voice_status(
        self,
        state: str,
        timestamp: float,
        *,
        data: dict[str, Any] | None = None,
        client: object | None = None,
        log: bool = True,
    ) -> None:
        payload = build_voice_status(
            VoiceStatusMessage(
                state=state,
                timestamp=timestamp,
                data=dict(data or {}),
                sequence=self._next_sequence(),
            )
        )
        self._send_payload(payload, client=client, log=log)

    def _next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def _publish_logged_payload(self, payload: dict[str, Any]) -> None:
        self._send_payload(payload, log=True)

    def _sync_main_audio_stream_mode(self) -> None:
        enabled = self.recognizer_rows[RECOGNIZER_AUDIO].checkbox.isChecked()
        self._request_main_audio_stream(enabled)

    def _request_main_audio_stream(self, enabled: bool) -> None:
        self.client.send_command(
            {
                "cmd": "set_main_audio",
                "enabled": bool(enabled),
                "source": "unified_gesture_app",
            }
        )

    def _send_payload(
        self,
        payload: dict[str, Any],
        *,
        client: object | None = None,
        log: bool = False,
    ) -> None:
        if log:
            scope = "unicast" if client is not None else "broadcast"
            target_count = 1 if client is not None else self.ws_bridge.client_count()
            self.message_log.appendPlainText(
                f"[ws {scope} targets={target_count}] {json.dumps(payload, ensure_ascii=False)}"
            )
        self.ws_bridge.send_payload(payload, client=client)

    def current_streams(self) -> list[str]:
        streams = ["imu"]
        if self.recognizer_rows[RECOGNIZER_AUDIO].checkbox.isChecked():
            streams.append("audio")
        return streams

    def checked_recognizers(self) -> list[str]:
        return [
            recognizer
            for recognizer, row in self.recognizer_rows.items()
            if row.checkbox.isChecked()
        ]

    def subscription_ring_addresses(self) -> list[str]:
        result: list[str] = []
        for address in (self.selected_ring_address("left"), self.selected_ring_address("right")):
            if address and address not in result:
                result.append(address)
        return result

    def selected_ring(self, role: str) -> RingTarget | None:
        combo = self.left_combo if role == "left" else self.right_combo
        address = str(combo.currentData() or "")
        if not address:
            return None
        ring = self.ring_info.get(address, {})
        name = str(ring.get("name", "")).strip()
        return RingTarget(role=role, address=address, name=name)

    def selected_ring_address(self, role: str) -> str:
        ring = self.selected_ring(role)
        return "" if ring is None else ring.address

    def selected_ring_connected(self, role: str) -> bool:
        address = self.selected_ring_address(role)
        if not address:
            return False
        ring = self.ring_info.get(address, {})
        return bool(ring.get("connected", False))

    def set_ring_combo(self, combo: QComboBox, address: str) -> None:
        for index in range(combo.count()):
            if str(combo.itemData(index) or "") == address:
                combo.setCurrentIndex(index)
                return

    def refresh_target_labels(self) -> None:
        left_name = self._ring_label_text(self.selected_ring("left"))
        right_name = self._ring_label_text(self.selected_ring("right"))
        for recognizer, row in self.recognizer_rows.items():
            if recognizer == RECOGNIZER_TYPE:
                row.target_label.setText(f"左手={left_name} | 右手={right_name}")
            else:
                row.target_label.setText(f"右手={right_name}")

    def _refresh_ring_combos(self) -> None:
        previous_left = self.selected_ring_address("left")
        previous_right = self.selected_ring_address("right")
        options = [("", "未选择")]
        for address, ring in sorted(self.ring_info.items(), key=lambda item: (str(item[1].get("name", "")), item[0])):
            label = self.format_ring_text(ring)
            options.append((address, label))
        self._populate_ring_combo(self.left_combo, options, previous_left)
        self._populate_ring_combo(self.right_combo, options, previous_right)

    def _populate_ring_combo(self, combo: QComboBox, options: list[tuple[str, str]], current_address: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        selected_index = 0
        for index, (address, label) in enumerate(options):
            combo.addItem(label, address)
            if address == current_address:
                selected_index = index
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _recognizer_target_text(self, recognizer: str) -> str:
        if recognizer == RECOGNIZER_TYPE:
            return "左手 + 右手"
        return "默认右手"

    def _ring_label_text(self, ring: RingTarget | None) -> str:
        if ring is None:
            return "未选择"
        return ring.name or ring.address[-8:]

    def _message_timestamp(self, message: dict[str, Any]) -> float:
        event_ts = float(message.get("local_timestamp", 0.0) or 0.0)
        if event_ts <= 0:
            event_ts = float(message.get("host_timestamp", 0.0) or 0.0)
        if event_ts <= 0:
            event_ts = time.time()
        return event_ts

    def _annotate_stream_message(self, message: dict[str, Any]) -> None:
        with self._stream_seq_lock:
            self._stream_seq += 1
            message[STREAM_SEQUENCE_KEY] = self._stream_seq

    def _current_stream_seq(self) -> int:
        with self._stream_seq_lock:
            return self._stream_seq

    def closeEvent(self, event) -> None:
        self.refresh_timer.stop()
        self.macos_keepalive.disable()
        self._request_main_audio_stream(False)
        self.client.stop()
        self.client.wait(1500)
        for runtime in self.runtimes.values():
            runtime.close()
        self.ws_bridge.stop()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17888)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = UnifiedGestureWindow(host=args.host, port=args.port)
    window.show()
    raise SystemExit(app.exec())
