import asyncio
import base64
import contextlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class RingInfo:
    address: str
    name: str
    connected: bool = False


@dataclass(eq=False)
class ClientSession:
    writer: asyncio.StreamWriter
    subscriptions: set[str] | None = None
    stream_types: set[str] = field(default_factory=lambda: {"imu"})
    queue: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=4096))
    writer_task: asyncio.Task | None = None


class ImuStreamHub:
    """Simple JSONL pub/sub server for cross-process IMU streaming."""

    def __init__(self, host: str = "127.0.0.1", port: int = 17888):
        self.host = host
        self.port = port
        self._server: asyncio.base_events.Server | None = None
        self._clients: set[ClientSession] = set()
        self._rings: dict[str, RingInfo] = {}
        self.command_handler: Callable[[ClientSession, dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any] | None] | None] | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(self._on_client, self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        clients = list(self._clients)
        self._clients.clear()
        for client in clients:
            if client.writer_task is not None:
                client.writer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await client.writer_task
            client.writer.close()
            with contextlib.suppress(Exception):
                await client.writer.wait_closed()

    def upsert_ring(self, address: str, name: str, connected: bool) -> None:
        self._rings[address] = RingInfo(address=address, name=name, connected=connected)
        self._broadcast_status()

    def remove_ring(self, address: str) -> None:
        self._rings.pop(address, None)
        self._broadcast_status()

    def publish_imu(
        self,
        *,
        address: str,
        name: str,
        local_timestamp: float,
        host_timestamp: float,
        acc_x: float,
        acc_y: float,
        acc_z: float,
        gyr_x: float,
        gyr_y: float,
        gyr_z: float,
        packet_end_counter: int = 0,
        arrival_perf: float = 0.0,
    ) -> None:
        payload = {
            "type": "imu",
            "address": address,
            "name": name,
            "timestamp": local_timestamp,
            "local_timestamp": local_timestamp,
            "host_timestamp": host_timestamp,
            "packet_end_counter": packet_end_counter,
            "arrival_perf": arrival_perf,
            "acc": [acc_x, acc_y, acc_z],
            "gyr": [gyr_x, gyr_y, gyr_z],
        }
        message = json.dumps(payload, ensure_ascii=False)
        for client in tuple(self._clients):
            if "imu" not in client.stream_types:
                continue
            if client.subscriptions is not None and address not in client.subscriptions:
                continue
            self._enqueue_for_client(client, message)

    def publish_audio(
        self,
        *,
        address: str,
        name: str,
        local_timestamp: float,
        host_timestamp: float,
        audio_type: int,
        audio_bytes: bytes,
    ) -> None:
        payload = {
            "type": "audio",
            "address": address,
            "name": name,
            "timestamp": local_timestamp,
            "local_timestamp": local_timestamp,
            "host_timestamp": host_timestamp,
            "audio_type": audio_type,
            "sample_rate": 16000,
            "data_b64": base64.b64encode(audio_bytes).decode("ascii"),
        }
        message = json.dumps(payload, ensure_ascii=False)
        for client in tuple(self._clients):
            if "audio" not in client.stream_types:
                continue
            if client.subscriptions is not None and address not in client.subscriptions:
                continue
            self._enqueue_for_client(client, message)

    def publish_touch(
        self,
        *,
        address: str,
        name: str,
        host_timestamp: float,
        action_code: int,
        tap_timestamp: float | None = None,
        release_timestamp: float | None = None,
    ) -> None:
        payload = {
            "type": "touch",
            "address": address,
            "name": name,
            "host_timestamp": host_timestamp,
            "action_code": action_code,
            "tap_timestamp": tap_timestamp,
            "release_timestamp": release_timestamp,
        }
        message = json.dumps(payload, ensure_ascii=False)
        for client in tuple(self._clients):
            if "touch" not in client.stream_types:
                continue
            if client.subscriptions is not None and address not in client.subscriptions:
                continue
            self._enqueue_for_client(client, message)

    def publish_swipe(
        self,
        *,
        address: str,
        name: str,
        host_timestamp: float,
        event: str,
        class_id: int,
        confidence: float,
        probabilities: list[float] | tuple[float, ...] | None = None,
        total_count: int | None = None,
        source: str | None = None,
    ) -> None:
        payload = {
            "type": "swipe",
            "address": address,
            "name": name,
            "timestamp": host_timestamp,
            "host_timestamp": host_timestamp,
            "event": event,
            "class_id": int(class_id),
            "confidence": float(confidence),
        }
        if probabilities is not None:
            payload["probabilities"] = [float(value) for value in probabilities]
        if total_count is not None:
            payload["total_count"] = int(total_count)
        if source:
            payload["source"] = str(source)
        message = json.dumps(payload, ensure_ascii=False)
        for client in tuple(self._clients):
            if "swipe" not in client.stream_types:
                continue
            if client.subscriptions is not None and address not in client.subscriptions:
                continue
            self._enqueue_for_client(client, message)

    async def _on_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        session = ClientSession(writer=writer)
        session.writer_task = asyncio.create_task(self._writer_loop(session))
        self._clients.add(session)
        self._enqueue_for_client(session, json.dumps({"type": "hello", "version": 1}))
        self._enqueue_for_client(session, self._status_message())

        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                text = raw.decode("utf-8").strip()
                if not text:
                    continue
                try:
                    command = json.loads(text)
                except json.JSONDecodeError:
                    self._enqueue_for_client(
                        session,
                        json.dumps({"type": "error", "message": "invalid_json"}),
                    )
                    continue
                await self._handle_command(session, command)
        finally:
            self._clients.discard(session)
            if session.writer_task is not None:
                session.writer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await session.writer_task
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _handle_command(self, session: ClientSession, command: dict[str, Any]) -> None:
        cmd = command.get("cmd")
        if cmd == "subscribe":
            rings = command.get("rings")
            if isinstance(rings, list):
                subscriptions = {str(x) for x in rings if str(x)}
                session.subscriptions = subscriptions or None
            else:
                session.subscriptions = None
            streams = command.get("streams")
            if isinstance(streams, list):
                stream_types = {str(x) for x in streams if str(x)}
                session.stream_types = stream_types or {"imu"}
            else:
                session.stream_types = {"imu"}
            self._enqueue_for_client(
                session,
                json.dumps(
                    {
                        "type": "subscribed",
                        "rings": [] if session.subscriptions is None else sorted(session.subscriptions),
                        "streams": sorted(session.stream_types),
                    }
                ),
            )
            return

        if cmd == "status":
            self._enqueue_for_client(session, self._status_message())
            return

        if self.command_handler is not None:
            try:
                response = self.command_handler(session, command)
                if inspect.isawaitable(response):
                    response = await response
            except Exception as exc:
                self._enqueue_for_client(
                    session,
                    json.dumps({"type": "error", "message": f"command_handler_failed: {exc}"}),
                )
                return
            if response is not None:
                self._enqueue_for_client(session, json.dumps(response, ensure_ascii=False))
                return

        self._enqueue_for_client(
            session,
            json.dumps({"type": "error", "message": "unsupported_command"}),
        )

    async def _writer_loop(self, session: ClientSession) -> None:
        while True:
            message = await session.queue.get()
            session.writer.write((message + "\n").encode("utf-8"))
            await session.writer.drain()

    def _enqueue_for_client(self, session: ClientSession, message: str) -> None:
        queue_obj = session.queue
        if queue_obj.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue_obj.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue_obj.put_nowait(message)

    def _status_message(self) -> str:
        payload = {
            "type": "status",
            "rings": [
                {
                    "address": ring.address,
                    "name": ring.name,
                    "connected": ring.connected,
                }
                for ring in self._rings.values()
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _broadcast_status(self) -> None:
        message = self._status_message()
        for client in tuple(self._clients):
            self._enqueue_for_client(client, message)
