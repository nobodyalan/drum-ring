"""Owns the stream hub lifecycle so the main window only coordinates UI events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from core.stream_hub import ImuStreamHub


@dataclass(frozen=True, slots=True)
class StreamMode:
    enabled_imu: bool
    enabled_audio: bool


class StreamRuntime:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        command_handler: Callable[[Any, dict[str, Any]], Awaitable[dict | None] | None]
        | None = None,
    ):
        self.host = host
        self.port = int(port)
        self._command_handler = command_handler
        self.hub = self._create_hub()
        self.started = False

    def _create_hub(self) -> ImuStreamHub:
        hub = ImuStreamHub(host=self.host, port=self.port)
        hub.command_handler = self._command_handler
        return hub

    def set_command_handler(
        self,
        handler: Callable[[Any, dict[str, Any]], Awaitable[dict | None] | None] | None,
    ) -> None:
        self._command_handler = handler
        self.hub.command_handler = handler

    async def start(self) -> None:
        await self.hub.start()
        self.started = True

    async def stop(self) -> None:
        await self.hub.stop()
        self.started = False

    async def restart(self, *, port: int, rings: list[tuple[str, str, bool]]) -> None:
        await self.stop()
        self.port = int(port)
        self.hub = self._create_hub()
        for address, name, connected in rings:
            self.upsert_ring(address, name, connected=connected)
        await self.start()

    def upsert_ring(self, address: str, name: str, *, connected: bool) -> None:
        self.hub.upsert_ring(address, name, connected=connected)

    def remove_ring(self, address: str) -> None:
        self.hub.remove_ring(address)

    def publish_imu(self, **payload: Any) -> None:
        self.hub.publish_imu(**payload)

    def publish_audio(self, **payload: Any) -> None:
        self.hub.publish_audio(**payload)

    def publish_touch(self, **payload: Any) -> None:
        self.hub.publish_touch(**payload)

    def publish_swipe(self, **payload: Any) -> None:
        self.hub.publish_swipe(**payload)
