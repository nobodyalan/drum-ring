import os
import traceback
import asyncio
from typing import Optional

import httpx
from httpx import URL
from loguru import logger

DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024

class AudioUploader:
    def __init__(
        self,
        token: str,
        base_url: str,
        *,
        file_name: str = "audio.wav",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout: float = 30.0,
        buffer_id_callback: callable = None,
    ):
        self.token = token
        self.base_url = base_url
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.buffer_id_callback = buffer_id_callback

        self.data_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._buffer_id: str | None = None
        self._chunk_index = 0
        self._buf = bytearray()
        self._upload_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _upload_chunk(self, client: httpx.AsyncClient, chunk_data: bytes, is_end: bool) -> None:
        # url = f"{self.base_url}/file-chunk"
        url = URL(f"{self.base_url}").join("/api/v1/file-chunk")
        logger.info(f"Upload chunk {self._chunk_index}, size={len(chunk_data)}, is_end={is_end}, url={url}")
        data = {
            "index": str(self._chunk_index),
            "file_name": self.file_name,
            "is_end": "true" if is_end else "false",
        }
        if self._buffer_id:
            data["buffer_id"] = self._buffer_id

        files = {"file": (self.file_name, chunk_data)}
        headers = {}
        if self.token is not None and self.token != "":
            headers["Authorization"] = f"Bearer {self.token}"

        resp = await client.post(url, headers=headers, data=data, files=files, timeout=self.timeout)
        resp.raise_for_status()
        out = resp.json()

        new_buffer_id = out.get("buffer_id")
        if new_buffer_id and self._buffer_id != new_buffer_id:
            if self.buffer_id_callback:
                logger.info(f"set buffer id: {new_buffer_id}")
                await self.buffer_id_callback(new_buffer_id)
            self._buffer_id = new_buffer_id
        self._chunk_index += 1
        logger.debug(f"AudioUploader chunk {self._chunk_index - 1} uploaded, buffer_id={self._buffer_id}")

    async def _upload_loop(self, client: httpx.AsyncClient) -> None:
        try:
            while True:
                item = await self.data_queue.get()

                if item is None:
                    if self._buf:
                        await self._upload_chunk(client, bytes(self._buf), is_end=True)
                        self._buf.clear()
                    else:
                        await self._upload_chunk(client, b"", is_end=True)
                    break

                self._buf.extend(item)
                while len(self._buf) >= self.chunk_size:
                    chunk = bytes(self._buf[: self.chunk_size])
                    del self._buf[: self.chunk_size]
                    await self._upload_chunk(client, chunk, is_end=False)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"AudioUploader upload loop error: {e}")
            raise

    async def __aenter__(self) -> "AudioUploader":
        self._client = httpx.AsyncClient()
        self._upload_task = asyncio.create_task(self._upload_loop(self._client))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.data_queue.put_nowait(None)
        if self._upload_task:
            await self._upload_task
            self._upload_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
