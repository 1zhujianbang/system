"""
Web 快照协议（Knowledge Graph 页面统一协议）。

设计原则：
- UI 不懂业务，只懂"快照协议"
- 所有图谱类型使用统一的 nodes/edges/meta 格式
- 前端只负责读取快照、渲染、交互
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...ports.snapshot import (
    GraphSnapshotType,
    Snapshot,
    SnapshotMeta,
    SnapshotNode,
    SnapshotEdge,
)


# =============================================================================
# 快照协议常量
# =============================================================================


# 默认快照目录
DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")

# 图谱类型显示名称
GRAPH_TYPE_LABELS: Dict[str, str] = {
    "GE": "实体-事件图 (GE)",
    "GET": "实体-事件时间线 (GET)",
    "EE": "实体-实体关系图 (EE)",
    "EE_EVO": "实体-实体演化图 (EE_EVO)",
    "EVENT_EVO": "事件演化图 (EVENT_EVO)",
    "KG": "原始知识图谱 (KG)",
}

# 节点类型颜色
NODE_COLORS: Dict[str, str] = {
    "entity": "#1f77b4",  # 蓝色
    "event": "#ff7f0e",   # 橙色
    "relation_state": "#999999",  # 灰色
}


# =============================================================================
# 快照加载器（供 UI 使用）
# =============================================================================


class SnapshotLoader:
    """
    快照加载器。
    供 Knowledge Graph 页面使用，统一加载和解析快照。
    """

    def __init__(self, snapshot_dir: Optional[Path] = None):
        self.snapshot_dir = snapshot_dir or DEFAULT_SNAPSHOT_DIR

    def list_available_types(self) -> List[str]:
        """列出可用的图谱类型"""
        available = ["KG"]  # 始终可用
        if self.snapshot_dir.exists():
            for graph_type in GraphSnapshotType:
                if graph_type == GraphSnapshotType.KG:
                    continue
                path = self.snapshot_dir / f"{graph_type.value}.json"
                if path.exists():
                    available.append(graph_type.value)
        return available

    def load_snapshot(
        self,
        graph_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        加载快照（返回原始 dict 供 UI 直接使用）。
        
        Args:
            graph_type: 图谱类型（GE/GET/EE/EE_EVO/EVENT_EVO/KG）
            
        Returns:
            快照数据 dict，或 None
        """
        import json

        if graph_type == "KG":
            # 加载原始 knowledge_graph.json
            kg_path = Path("data/knowledge_graph.json")
            if kg_path.exists():
                try:
                    return json.loads(kg_path.read_text(encoding="utf-8"))
                except Exception:
                    return None
            return None

        # 加载五图谱快照
        path = self.snapshot_dir / f"{graph_type}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_snapshot_meta(self, graph_type: str) -> Optional[Dict[str, Any]]:
        """获取快照元数据"""
        data = self.load_snapshot(graph_type)
        if data:
            return data.get("meta", {})
        return None

    def get_snapshot_stats(self, graph_type: str) -> Dict[str, int]:
        """获取快照统计"""
        data = self.load_snapshot(graph_type)
        if data:
            return {
                "nodes": len(data.get("nodes", [])),
                "edges": len(data.get("edges", [])),
            }
        return {"nodes": 0, "edges": 0}


# =============================================================================
# 渲染配置（供 UI 使用）
# =============================================================================


@dataclass
class RenderConfig:
    """渲染配置"""
    max_nodes: int = 500
    max_edges: int = 2000
    window_hours: int = 0  # 0 = 不限制
    focus_entity: str = ""
    focus_event: str = ""
    show_labels: bool = True
    layout_algorithm: str = "force"  # force / hierarchical / circular


@dataclass
class FilterConfig:
    """过滤配置"""
    node_types: List[str] = field(default_factory=lambda: ["entity", "event"])
    edge_types: List[str] = field(default_factory=list)  # 空 = 全部
    min_degree: int = 0
    time_range_hours: int = 0


# =============================================================================
# 快照转换器（统一协议）
# =============================================================================


class SnapshotTransformer:
    """
    快照转换器。
    将不同格式的数据转换为统一的快照协议格式。
    """

    @staticmethod
    def normalize_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        规范化节点列表。
        确保每个节点都有 id, label, type, color 字段。
        """
        result = []
        for n in nodes:
            if not isinstance(n, dict) or not n.get("id"):
                continue
            result.append({
                "id": str(n.get("id", "")),
                "label": str(n.get("label", n.get("id", ""))),
                "type": str(n.get("type", "entity")),
                "color": str(n.get("color", NODE_COLORS.get(n.get("type", "entity"), "#1f77b4"))),
                **{k: v for k, v in n.items() if k not in ["id", "label", "type", "color"]},
            })
        return result

    @staticmethod
    def normalize_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        规范化边列表。
        确保每条边都有 from, to, type, title, time 字段。
        """
        result = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            from_node = str(e.get("from", ""))
            to_node = str(e.get("to", ""))
            if not from_node or not to_node:
                continue
            result.append({
                "from": from_node,
                "to": to_node,
                "type": str(e.get("type", "")),
                "title": str(e.get("title", "")),
                "time": str(e.get("time", "")),
                **{k: v for k, v in e.items() if k not in ["from", "to", "type", "title", "time"]},
            })
        return result

    @staticmethod
    def from_kg_json(kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 knowledge_graph.json 格式转换为统一协议。
        """
        nodes = []
        edges = []

        # 转换实体
        for name, entity in (kg_data.get("entities") or {}).items():
            if not isinstance(entity, dict):
                continue
            nodes.append({
                "id": name,
                "label": name,
                "type": "entity",
                "color": NODE_COLORS["entity"],
            })

        # 转换事件
        for abstract, event in (kg_data.get("events") or {}).items():
            if not isinstance(event, dict):
                continue
            event_id = f"EVT:{abstract}"
            nodes.append({
                "id": event_id,
                "label": str(event.get("event_summary", abstract))[:80],
                "type": "event",
                "color": NODE_COLORS["event"],
            })

        # 转换边
        for edge in (kg_data.get("edges") or []):
            if not isinstance(edge, dict):
                continue
            edges.append({
                "from": str(edge.get("from", "")),
                "to": str(edge.get("to", "")),
                "type": str(edge.get("type", "")),
                "title": str(edge.get("predicate", "") or edge.get("title", "")),
                "time": str(edge.get("time", "")),
            })

        return {
            "meta": {
                "graph_type": "KG",
                "generated_at": "",
            },
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def filter_by_time(
        snapshot: Dict[str, Any],
        hours: int,
    ) -> Dict[str, Any]:
        """
        按时间过滤快照。
        """
        if hours <= 0:
            return snapshot

        from datetime import timezone, timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        def parse_time(val: str) -> Optional[datetime]:
            if not val:
                return None
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None

        # 过滤边
        filtered_edges = []
        valid_nodes = set()
        for edge in snapshot.get("edges", []):
            time_str = str(edge.get("time", ""))
            dt = parse_time(time_str)
            if dt and dt >= cutoff:
                filtered_edges.append(edge)
                valid_nodes.add(edge.get("from", ""))
                valid_nodes.add(edge.get("to", ""))
            elif not time_str:
                # 无时间的边也保留
                filtered_edges.append(edge)
                valid_nodes.add(edge.get("from", ""))
                valid_nodes.add(edge.get("to", ""))

        # 过滤节点
        filtered_nodes = [
            n for n in snapshot.get("nodes", [])
            if n.get("id") in valid_nodes
        ]

        return {
            "meta": snapshot.get("meta", {}),
            "nodes": filtered_nodes,
            "edges": filtered_edges,
        }

    @staticmethod
    def filter_by_focus(
        snapshot: Dict[str, Any],
        focus_entity: str = "",
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """
        按聚焦实体过滤快照（BFS 扩展）。
        """
        if not focus_entity:
            return snapshot

        from collections import defaultdict, deque

        # 构建邻接表
        adj = defaultdict(set)
        for edge in snapshot.get("edges", []):
            u, v = edge.get("from", ""), edge.get("to", "")
            if u and v:
                adj[u].add(v)
                adj[v].add(u)

        # BFS
        visited = set()
        queue = deque([(focus_entity, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        # 过滤
        filtered_nodes = [n for n in snapshot.get("nodes", []) if n.get("id") in visited]
        filtered_edges = [
            e for e in snapshot.get("edges", [])
            if e.get("from") in visited and e.get("to") in visited
        ]

        return {
            "meta": snapshot.get("meta", {}),
            "nodes": filtered_nodes,
            "edges": filtered_edges,
        }


REQUIRED_NODE_FIELDS = ["id", "label", "type", "color"]
REQUIRED_EDGE_FIELDS = ["from", "to", "type", "title", "time"]

RECOMMENDED_NODE_FIELDS_BY_GRAPH: Dict[str, List[str]] = {
    "GE": ["entity_id", "event_id", "abstract", "time", "event_types", "roles"],
    "GET": ["entity_id", "event_id", "abstract", "time", "event_types"],
    "EE": ["entity_id"],
    "EE_EVO": ["entity_id", "interval_start", "interval_end", "evidence"],
    "EVENT_EVO": ["event_id", "abstract", "time", "event_types"],
    "KG": [],
}

RECOMMENDED_EDGE_FIELDS_BY_GRAPH: Dict[str, List[str]] = {
    "GE": ["roles", "event_id", "abstract"],
    "GET": ["event_id", "abstract"],
    "EE": ["predicate", "evidence", "event_id"],
    "EE_EVO": ["predicate", "evidence"],
    "EVENT_EVO": ["confidence", "evidence", "reported_at"],
    "KG": [],
}


def validate_snapshot_dict(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    meta = snapshot.get("meta") if isinstance(snapshot, dict) else {}
    graph_type = str((meta or {}).get("graph_type") or "")

    nodes = snapshot.get("nodes") if isinstance(snapshot, dict) else None
    edges = snapshot.get("edges") if isinstance(snapshot, dict) else None
    nodes = nodes if isinstance(nodes, list) else []
    edges = edges if isinstance(edges, list) else []

    errors: List[str] = []

    missing_nodes = []
    for i, n in enumerate(nodes[:2000]):
        if not isinstance(n, dict):
            missing_nodes.append({"index": i, "missing": REQUIRED_NODE_FIELDS})
            continue
        missing = [k for k in REQUIRED_NODE_FIELDS if not str(n.get(k, "")).strip()]
        if missing:
            missing_nodes.append({"index": i, "id": str(n.get("id", "")), "missing": missing})

    missing_edges = []
    for i, e in enumerate(edges[:5000]):
        if not isinstance(e, dict):
            missing_edges.append({"index": i, "missing": REQUIRED_EDGE_FIELDS})
            continue
        missing = [k for k in REQUIRED_EDGE_FIELDS if not str(e.get(k, "")).strip()]
        if missing:
            missing_edges.append({"index": i, "from": str(e.get("from", "")), "to": str(e.get("to", "")), "missing": missing})

    meta_node_count = (meta or {}).get("node_count")
    meta_edge_count = (meta or {}).get("edge_count")
    if meta_node_count is not None and int(meta_node_count) != len(nodes):
        errors.append(f"meta.node_count({meta_node_count}) != len(nodes)({len(nodes)})")
    if meta_edge_count is not None and int(meta_edge_count) != len(edges):
        errors.append(f"meta.edge_count({meta_edge_count}) != len(edges)({len(edges)})")

    return {
        "graph_type": graph_type,
        "ok": (not errors) and (not missing_nodes) and (not missing_edges),
        "errors": errors,
        "missing_nodes": missing_nodes[:50],
        "missing_edges": missing_edges[:50],
        "counts": {"nodes": len(nodes), "edges": len(edges)},
        "required": {"node": REQUIRED_NODE_FIELDS, "edge": REQUIRED_EDGE_FIELDS},
        "recommended": {
            "node": RECOMMENDED_NODE_FIELDS_BY_GRAPH.get(graph_type, []),
            "edge": RECOMMENDED_EDGE_FIELDS_BY_GRAPH.get(graph_type, []),
        },
    }
