from __future__ import annotations

from typing import Any, Dict, List, Protocol


class KGReadStore(Protocol):
    """
    Ports：图谱只读仓储端口（用于 Projection/Snapshots）。
    """

    def fetch_entities(self) -> List[Dict[str, Any]]: ...
    def fetch_events(self) -> List[Dict[str, Any]]: ...
    def fetch_participants_with_events(self) -> List[Dict[str, Any]]: ...
    def fetch_relations(self) -> List[Dict[str, Any]]: ...
    def fetch_event_edges(self) -> List[Dict[str, Any]]: ...






