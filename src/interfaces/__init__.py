"""
入口层（Interfaces Layer）。

包含：
- tools: @register_tool 工具封装（薄封装调用 app 服务）
- web: Web UI 相关接口和协议

设计原则：
- 入口层只做交互与展示，不直接拼业务逻辑
- 工具是"调用 service"的薄封装
"""

# Tools - 延迟导入以避免循环依赖
# 可以直接从子模块导入：
# - from src.interfaces.tools.snapshots import generate_graph_snapshots
# - from src.interfaces.tools.migration import migrate_json_to_sqlite, backfill_mentions
# - from src.interfaces.tools.review import enqueue_entity_merge_candidates_v2, ...

# Web - 快照协议
from .web import (
    GRAPH_TYPE_LABELS,
    NODE_COLORS,
    SnapshotLoader,
    RenderConfig,
    FilterConfig,
    SnapshotTransformer,
)

__all__ = [
    # Web
    "GRAPH_TYPE_LABELS",
    "NODE_COLORS",
    "SnapshotLoader",
    "RenderConfig",
    "FilterConfig",
    "SnapshotTransformer",
]



