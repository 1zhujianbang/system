from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from .serialization import Serializer

# 类型别名
LogEntry = Dict[str, str]
ExecutionRecord = Dict[str, Union[str, float]]

class PipelineContext:
    """
    流程执行上下文
    用于在 Pipeline 的不同 Task 之间传递数据，并记录执行日志。
    """
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None, log_callback: Optional[Callable[[LogEntry], None]] = None) -> None:
        self._store: Dict[str, Any] = initial_data or {}
        self._logs: List[LogEntry] = []
        self._execution_history: List[ExecutionRecord] = []
        self.log_callback: Optional[Callable[[LogEntry], None]] = log_callback

    def set(self, key: str, value: Any) -> None:
        """设置上下文变量"""
        self._store[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """获取上下文变量"""
        # 支持嵌套键访问，例如 "news.data.0" (简化版暂只支持一级)
        return self._store.get(key, default)

    def log(self, message: str, level: str = "INFO", source: str = "System") -> None:
        """记录日志"""
        timestamp = datetime.now().isoformat()
        entry: LogEntry = {
            "timestamp": timestamp,
            "level": level,
            "source": source,
            "message": message
        }
        self._logs.append(entry)
        # 简单控制台输出，后续可对接更复杂的日志系统
        print(f"[{timestamp}] [{level}] [{source}] {message}")

        if self.log_callback:
            try:
                self.log_callback(entry)
            except Exception:
                pass

    def record_execution(self, tool_name: str, status: str, duration: float, error: Optional[str] = None) -> None:
        """记录工具执行情况"""
        self._execution_history.append({
            "tool": tool_name,
            "status": status,
            "duration": duration,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_all(self) -> Dict[str, Any]:
        """获取所有数据"""
        return self._store

    @property
    def logs(self) -> List[Dict[str, str]]:
        return self._logs

    def to_json(self) -> str:
        """导出上下文状态（使用安全的序列化）"""
        return Serializer.serialize_context(self)

