import json
import queue
import socket
import time
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal


class JsonLineStreamClient(QThread):
    sig_connected = pyqtSignal(bool)
    sig_message = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(self, host: str, port: int, reconnect_interval: float = 2.0):
        super().__init__()
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval
        self._running = True
        self._send_queue: queue.Queue[str] = queue.Queue(maxsize=1024)
        self._message_hooks: list = []

    def send_command(self, command: dict[str, Any]) -> None:
        try:
            payload = json.dumps(command, ensure_ascii=False)
            self._send_queue.put_nowait(payload)
        except queue.Full:
            self.sig_error.emit("发送队列已满，指令被丢弃")

    def stop(self) -> None:
        self._running = False

    def add_message_hook(self, hook) -> None:
        self._message_hooks.append(hook)

    def run(self) -> None:
        while self._running:
            sock = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=3.0)
                sock.settimeout(0.2)
                self.sig_connected.emit(True)
                self._connection_loop(sock)
            except Exception as exc:
                self.sig_error.emit(f"连接异常: {exc}")
            finally:
                self.sig_connected.emit(False)
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

            self._sleep_with_break(self.reconnect_interval)

    def _connection_loop(self, sock: socket.socket) -> None:
        buffer = b""
        while self._running:
            self._flush_send_queue(sock)
            try:
                chunk = sock.recv(8192)
                if not chunk:
                    return
                buffer += chunk
            except socket.timeout:
                continue

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    self.sig_error.emit("收到无法解析的 JSON 消息")
                    continue
                for hook in tuple(self._message_hooks):
                    try:
                        hook(payload)
                    except Exception:
                        pass
                self.sig_message.emit(payload)

    def _flush_send_queue(self, sock: socket.socket) -> None:
        while not self._send_queue.empty():
            try:
                payload = self._send_queue.get_nowait()
            except queue.Empty:
                return
            try:
                sock.sendall((payload + "\n").encode("utf-8"))
            except Exception as exc:
                self.sig_error.emit(f"发送异常: {exc}")
                raise

    def _sleep_with_break(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while self._running and time.time() < deadline:
            time.sleep(0.1)
