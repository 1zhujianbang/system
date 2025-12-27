from __future__ import annotations

from typing import Any, Dict, Optional

from ...app.snapshot_service import SnapshotService
from ...infra.registry import register_tool


@register_tool(
    name="generate_kg_visual_snapshots",
    description="生成轻量图谱快照：kg_visual.json（实体-事件裁剪）与 kg_visual_timeline.json（事件时间线）",
    category="Knowledge Graph"
)
def generate_kg_visual_snapshots(
    kg_path: str = "data/knowledge_graph.json",
    out_path_graph: str = "data/kg_visual.json",
    out_path_timeline: str = "data/kg_visual_timeline.json",
    top_entities: int = 500,
    top_events: int = 500,
    max_title_len: int = 80,
    days_window: Optional[int] = None
) -> Dict[str, str]:
    """生成轻量级可视化快照，委托到 functions/graph_ops 实现"""
    from ...app.business.graph_ops import generate_kg_visual_snapshots as _impl
    return _impl(
        kg_path=kg_path,
        out_path_graph=out_path_graph,
        out_path_timeline=out_path_timeline,
        top_entities=top_entities,
        top_events=top_events,
        max_title_len=max_title_len,
        days_window=days_window
    )


@register_tool(
    name="generate_graph_snapshots",
    description="生成五种图谱快照到 data/snapshots（GE/GET/EE/EE_EVO/EVENT_EVO）",
    category="Knowledge Graph",
)
def generate_graph_snapshots(
    top_entities: int = 500,
    top_events: int = 500,
    max_edges: int = 5000,
    days_window: int = 0,
    gap_days: int = 30,
) -> Dict[str, Any]:
    svc = SnapshotService()
    return svc.generate(
        top_entities=int(top_entities),
        top_events=int(top_events),
        max_edges=int(max_edges),
        days_window=int(days_window),
        gap_days=int(gap_days),
    )






