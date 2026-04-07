from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TypeRingEventLogEntry:
    timestamp: float
    source: str
    event: str
    action: str = ""


@dataclass(slots=True)
class TypeRingSessionState:
    status_text: str = "等待开始"
    phrase_text: str = ""
    input_text: str = ""
    committed_text: str = ""
    candidates: list[str] = field(default_factory=list)
    selected_candidate_index: int = 0
    left_ring_text: str = "未绑定"
    right_ring_text: str = "未绑定"
    event_logs: list[TypeRingEventLogEntry] = field(default_factory=list)
    worker_status: dict[str, str] = field(default_factory=dict)
