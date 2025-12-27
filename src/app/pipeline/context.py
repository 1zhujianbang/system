from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


LogCallback = Callable[[Dict[str, Any]], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineContext:
    """UI/执行器共享的上下文（仅存 JSON 友好的中小对象；大对象建议落盘后存指针）。"""

    initial_data: Optional[Dict[str, Any]] = None
    log_callback: Optional[LogCallback] = None

    def __post_init__(self) -> None:
        self._store: Dict[str, Any] = dict(self.initial_data or {})

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._store)

    def log(self, message: str, *, level: str = "INFO", source: str = "Pipeline") -> None:
        entry = {"timestamp": _utc_now_iso(), "level": level, "source": source, "message": message}
        if self.log_callback:
            try:
                self.log_callback(entry)
            except Exception:
                pass






