from typing import List, Dict, Any, Optional
from ..core.registry import register_tool
from ..utils.entity_updater import update_entities, update_abstract_map
from ..agents.agent3 import refresh_graph as _refresh_graph
from ..agents.agent3 import append_only_update_graph as _append_only_update_graph
from pathlib import Path
import json
from datetime import datetime, timedelta

@register_tool(
    name="update_graph_data",
    description="将提取的事件数据写入知识图谱文件 (Entities & Events)",
    category="Knowledge Graph"
)
def update_graph_data(events_list: List[Dict[str, Any]], default_source: str = "auto_pipeline") -> Dict[str, Any]:
    """
    更新知识图谱数据文件。
    
    Args:
        events_list: 事件列表，每项应包含 entities, abstract 等，以及可选的 source, published_at
        default_source: 默认来源标识
        
    Returns:
        更新状态
    """
    count = 0
    for event in events_list:
        entities = event.get("entities", [])
        entities_original = event.get("entities_original", [])
        # 如果没有原始形式，回退到实体名
        if not entities_original:
            entities_original = entities
            
        source = event.get("source", default_source)
        published_at = event.get("published_at")
        
        # 更新实体库
        update_entities(entities, entities_original, source, published_at)
        count += 1
        
    # 更新事件映射 (abstract_map)
    # update_abstract_map 期望的是 events_list，但其中的 item 需要有 source 和 published_at
    # 如果 event 字典里已经有这些字段，update_abstract_map 内部怎么处理？
    # 查看 entity_updater.py:
    #   def update_abstract_map(extracted_list, source, published_at):
    # 它接受一个 source 和 published_at 参数，统一应用于所有 item。
    # 这对于批量处理不同来源的事件不太友好。
    # 我们可以稍微 hack 一下：循环调用 update_abstract_map，或者修改 update_abstract_map。
    # 为了不修改原有 util，我们按 source 分组调用。
    
    # 简单的按个调用 (效率稍低但安全)
    for event in events_list:
        src = event.get("source", default_source)
        ts = event.get("published_at")
        update_abstract_map([event], src, ts)
    
    return {"status": "success", "updated_count": count}

@register_tool(
    name="refresh_knowledge_graph",
    description="重建并压缩知识图谱 (触发 Agent3 逻辑)",
    category="Knowledge Graph"
)
def refresh_knowledge_graph() -> Dict[str, str]:
    """
    刷新知识图谱：构建、压缩、更新
    """
    try:
        _refresh_graph()
        return {"status": "refreshed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
    """
    生成轻量级可视化快照，便于前端直接加载：
    - kg_visual.json: nodes (id/label/type/color), edges (from/to/title)
    - kg_visual_timeline.json: events 按时间排序，包含摘要、时间、实体列表（截断）
    """
    kg_file = Path(kg_path)
    if not kg_file.exists():
        return {"status": "error", "message": f"{kg_file} not found"}

    try:
        kg = json.loads(kg_file.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"load kg failed: {e}"}

    entities = kg.get("entities") or {}
    events = kg.get("events") or {}
    edges = kg.get("edges") or []

    # 时间窗口过滤
    cutoff = None
    if days_window and days_window > 0:
        cutoff = datetime.utcnow() - timedelta(days=days_window)

    # 计算度
    deg = {}
    filtered_edges = []
    for e in edges:
        u, v = e.get("from"), e.get("to")
        if not u or not v:
            continue
        # 事件时间过滤（基于事件节点）
        if cutoff and isinstance(v, str) and v.startswith("EVT:"):
            evt_key = v[4:]
            evt = events.get(evt_key, {})
            ts = evt.get("first_seen") or evt.get("published_at")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
            except Exception:
                dt = None
            if dt and dt < cutoff:
                continue

        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
        filtered_edges.append(e)

    # 取 Top 节点
    top_ids = set()
    if deg:
        top_all = sorted(deg, key=deg.get, reverse=True)
        top_ids = set(top_all[: top_entities + top_events])

    # 节点表
    vis_nodes = []
    node_types = {}
    for nid in top_ids:
        if isinstance(nid, str) and nid.startswith("EVT:"):
            evt = events.get(nid[4:], {})
            label = (evt.get("event_summary") or evt.get("abstract") or nid[4:])[:max_title_len]
            vis_nodes.append({"id": nid, "label": label, "type": "event", "color": "#ff7f0e"})
            node_types[nid] = "event"
        else:
            vis_nodes.append({"id": nid, "label": str(nid), "type": "entity", "color": "#1f77b4"})
            node_types[nid] = "entity"

    # 边表
    vis_edges = []
    for e in filtered_edges:
        u, v = e.get("from"), e.get("to")
        if u in top_ids and v in top_ids:
            title = ""
            if isinstance(v, str) and v.startswith("EVT:"):
                evt_key = v[4:]
                title = (events.get(evt_key, {}) or {}).get("event_summary", "")[:max_title_len]
            vis_edges.append({"from": u, "to": v, "title": title})

    out_graph = Path(out_path_graph)
    out_graph.parent.mkdir(parents=True, exist_ok=True)
    out_graph.write_text(json.dumps({"nodes": vis_nodes, "edges": vis_edges}, ensure_ascii=False), encoding="utf-8")

    # 时间线快照（取 TopN 事件按时间排序）
    tl_rows = []
    for evt_key, evt in events.items():
        ts = evt.get("first_seen") or evt.get("published_at")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
        except Exception:
            dt = None
        if not dt:
            continue
        if cutoff and dt < cutoff:
            continue
        ents = evt.get("entities", [])
        tl_rows.append({
            "time": ts,
            "event_summary": (evt.get("event_summary") or evt_key)[:max_title_len],
            "entities": ents[:20],
        })

    tl_rows = sorted(tl_rows, key=lambda x: x["time"])[: top_events]
    out_tl = Path(out_path_timeline)
    out_tl.parent.mkdir(parents=True, exist_ok=True)
    out_tl.write_text(json.dumps(tl_rows, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "ok",
        "graph_snapshot": str(out_graph),
        "timeline_snapshot": str(out_tl),
        "nodes": len(vis_nodes),
        "edges": len(vis_edges),
        "timeline_events": len(tl_rows),
    }


@register_tool(
    name="append_only_update_graph",
    description="仅追加新事件/实体到现有图谱，不修改旧记录（可选是否追加原始表述）",
    category="Knowledge Graph"
)
def append_only_update_graph_tool(
    events_list: Any,
    default_source: str = "auto_pipeline",
    allow_append_original_forms: bool = True
) -> Dict[str, int]:
    """
    包装 agent3.append_only_update_graph：
    - 仅当实体/事件不存在时新增
    - 不改旧 first_seen / sources / 摘要；可选是否为旧实体追加 original_forms
    - 兼容输入为字符串/嵌套列表，尽可能解析为 Dict 列表
    """
    def normalize(ev_input: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if ev_input is None:
            return out

        # 若传入的是文件路径（jsonl/json），尝试读取
        def load_file(path_str: str):
            p = Path(path_str)
            if not p.exists():
                return
            try:
                if p.suffix.lower() in [".jsonl", ".json"]:
                    if p.suffix.lower() == ".jsonl":
                        with p.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict):
                                        out.append(obj)
                                except Exception:
                                    continue
                    else:
                        parsed = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(parsed, dict):
                            out.append(parsed)
                        elif isinstance(parsed, list):
                            out.extend([p for p in parsed if isinstance(p, dict)])
            except Exception:
                return

        items = ev_input if isinstance(ev_input, list) else [ev_input]
        for it in items:
            if isinstance(it, dict):
                out.append(it)
            elif isinstance(it, list):
                for sub in it:
                    if isinstance(sub, dict):
                        out.append(sub)
                    elif isinstance(sub, str):
                        try:
                            parsed = json.loads(sub)
                            if isinstance(parsed, dict):
                                out.append(parsed)
                            elif isinstance(parsed, list):
                                out.extend([p for p in parsed if isinstance(p, dict)])
                        except Exception:
                            # 如果是文件路径，尝试读取 jsonl/json
                            try:
                                load_file(sub)
                            except Exception:
                                continue
            elif isinstance(it, str):
                try:
                    parsed = json.loads(it)
                    if isinstance(parsed, dict):
                        out.append(parsed)
                    elif isinstance(parsed, list):
                        out.extend([p for p in parsed if isinstance(p, dict)])
                except Exception:
                    # 若不是 JSON，尝试当作文件路径
                    load_file(it)
        return out

    normalized_events = normalize(events_list)
    return _append_only_update_graph(
        events_list=normalized_events,
        default_source=default_source,
        allow_append_original_forms=allow_append_original_forms
    )


@register_tool(
    name="append_tmp_extracted_events",
    description="从 data/tmp/extracted_events_*.jsonl 读取事件并追加到图谱（不改旧记录）",
    category="Knowledge Graph"
)
def append_tmp_extracted_events(
    base_dir: str = "data/tmp",
    pattern: str = "extracted_events_*.jsonl",
    max_files: int = 0,
    default_source: str = "auto_pipeline",
    allow_append_original_forms: bool = True
) -> Dict[str, Any]:
    """
    读取 tmp 中的提取结果 jsonl 文件并调用 append_only_update_graph 追加到实体/事件库。
    max_files=0 表示全部；>0 时按修改时间倒序取前 N 个。
    """
    base = Path(base_dir)
    if not base.exists():
        return {"status": "error", "message": f"{base} not found"}
    files = sorted(base.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if max_files and max_files > 0:
        files = files[:max_files]
    events: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            events.append(obj)
                    except Exception:
                        continue
        except Exception:
            continue
    res = _append_only_update_graph(
        events_list=events,
        default_source=default_source,
        allow_append_original_forms=allow_append_original_forms
    )
    return {
        "status": "ok",
        "files_used": [str(f) for f in files],
        "added_entities": res.get("added_entities", 0),
        "added_events": res.get("added_events", 0)
    }