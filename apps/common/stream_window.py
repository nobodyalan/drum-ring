from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QMainWindow, QScrollArea, QVBoxLayout, QWidget

from common.stream_client import JsonLineStreamClient

NONE_SUBSCRIPTION = "__none__"


class RingStreamWindowBase(QMainWindow):
    """Shared flow for stream apps: connect, fetch ring status, select rings, subscribe."""

    def __init__(self, host: str, port: int):
        super().__init__()
        self.client = JsonLineStreamClient(host=host, port=port)
        self.ring_boxes: dict[str, QCheckBox] = {}
        self.ring_info: dict[str, dict[str, Any]] = {}
        self.status_label: QLabel | None = None
        self.ring_layout: QVBoxLayout | None = None

    def set_stream_status_label(self, label: QLabel) -> None:
        self.status_label = label
        self.set_connection_status(False)

    def add_ring_selection_panel(
        self,
        parent_layout: QVBoxLayout,
        *,
        title_text: str,
        max_height: int = 180,
    ) -> QScrollArea:
        title = QLabel(title_text)
        title.setProperty("class", "sub-text")
        parent_layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        content = QWidget()
        self.ring_layout = QVBoxLayout(content)
        self.ring_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        scroll.setMaximumHeight(max_height)
        parent_layout.addWidget(scroll)
        return scroll

    def bind_stream_client_signals(self) -> None:
        self.client.sig_connected.connect(self.on_connected_changed)
        self.client.sig_message.connect(self.on_stream_message)
        self.client.sig_error.connect(self.on_error)

    def set_connection_status(self, connected: bool) -> None:
        if self.status_label is not None:
            self.status_label.setText(f"数据流状态：{'已连接' if connected else '未连接'}")

    def set_stream_error(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(f"数据流状态：异常（{message}）")

    def on_connected_changed(self, connected: bool) -> None:
        self.set_connection_status(connected)
        if connected:
            self.client.send_command({"cmd": "status"})
            self.on_stream_connected()

    def on_stream_connected(self) -> None:
        pass

    def on_error(self, message: str) -> None:
        self.set_stream_error(message)

    def on_stream_message(self, message: dict) -> None:
        if message.get("type") == "status":
            self.sync_rings(message.get("rings", []))
            return
        self.handle_stream_message(message)

    def handle_stream_message(self, message: dict) -> None:
        raise NotImplementedError

    def format_ring_text(self, ring: dict[str, Any]) -> str:
        ring_name = str(ring.get("name", "Unknown"))
        address = str(ring.get("address", ""))
        connected = bool(ring.get("connected", False))
        return f"{ring_name} | {address} | {'在线' if connected else '离线'}"

    def sync_rings(self, rings: list[dict]) -> None:
        if self.ring_layout is None:
            raise RuntimeError("ring selection panel has not been initialized")

        previous_info = self.ring_info
        ring_map = {
            str(item.get("address", "")): dict(item)
            for item in rings
            if str(item.get("address", ""))
        }
        self.ring_info = ring_map

        removed_addresses: list[str] = []
        changed_connection_addresses: list[str] = []

        for address in list(self.ring_boxes.keys()):
            if address not in ring_map:
                box = self.ring_boxes.pop(address)
                box.setParent(None)
                removed_addresses.append(address)

        for address, ring in ring_map.items():
            if address in self.ring_boxes:
                box = self.ring_boxes[address]
            else:
                box = QCheckBox()
                box.setChecked(False)
                self.ring_boxes[address] = box
                self.ring_layout.addWidget(box)

            previous_connected = bool(previous_info.get(address, {}).get("connected", False))
            current_connected = bool(ring.get("connected", False))
            if address in previous_info and previous_connected != current_connected:
                changed_connection_addresses.append(address)

            box.setText(self.format_ring_text(ring))

        self.on_rings_updated(
            rings=rings,
            removed_addresses=removed_addresses,
            changed_connection_addresses=changed_connection_addresses,
        )

    def on_rings_updated(
        self,
        *,
        rings: list[dict],
        removed_addresses: list[str],
        changed_connection_addresses: list[str],
    ) -> None:
        _ = rings
        _ = removed_addresses
        _ = changed_connection_addresses

    def selected_ring_addresses(self) -> list[str]:
        return [addr for addr, box in self.ring_boxes.items() if box.isChecked()]

    def build_subscription_rings(self, empty_behavior: str = "all") -> list[str]:
        selected = self.selected_ring_addresses()
        if selected:
            return selected
        if empty_behavior == "none":
            return [NONE_SUBSCRIPTION]
        return []

    def send_subscription(self, streams: list[str], empty_behavior: str = "all") -> None:
        self.client.send_command(
            {
                "cmd": "subscribe",
                "rings": self.build_subscription_rings(empty_behavior=empty_behavior),
                "streams": streams,
            }
        )
