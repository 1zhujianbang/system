from typing import List, Dict, Any, Optional, Set, Tuple
from ...infra.registry import register_tool
from ...core import DataNormalizer, StandardEventPipeline, ConfigManager, LLMAPIPool, RateLimiter, AsyncExecutor, tools, get_config_manager, get_llm_pool
from ...domain.data_operations import update_entities, update_abstract_map
from ...infra.serialization import extract_json_from_llm_response
from ...infra.async_utils import call_llm_with_retry, create_deduplication_prompt, create_event_deduplication_prompt
from ...infra.file_utils import ensure_dir
from ...domain.data_operations import write_json_file, read_json_file
from pathlib import Path
import json
from datetime import datetime, timedelta, timezone
import time
import threading
from difflib import SequenceMatcher
from collections import defaultdict

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
    
    # 简单的按个调用 (效率稍低但安全)
    for event in events_list:
        src = event.get("source", default_source)
        ts = event.get("published_at")
        update_abstract_map([event], src, ts)

    # SQLite 为主存储：批量写入后统一导出兼容 JSON
    try:
        from ...adapters.sqlite.store import get_store
        get_store().export_compat_json_files()
    except Exception as e:
        tools.log(f"⚠️ 导出兼容JSON失败（不影响主存储SQLite）: {e}")
    
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
        # 优先应用已确认的实体合并决策（避免每次 refresh 都让 LLM 重新“漂移”）
        try:
            from .review_ops import apply_entity_merge_decisions
            apply_entity_merge_decisions(max_actions=200)
        except Exception:
            pass
        # 优先应用已确认的事件 merge/evolve 决策
        try:
            from .review_ops import apply_event_merge_or_evolve_decisions
            apply_event_merge_or_evolve_decisions(max_actions=200)
        except Exception:
            pass
        kg = KnowledgeGraph()
        kg.refresh_graph()
        return {"status": "refreshed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@register_tool(
    name="generate_kg_visual_snapshots",
    description="生成轻量图谱快照：kg_visual（实体-事件裁剪）与 kg_visual_timeline（事件时间线）",
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
    - kg_visual: nodes (id/label/type/color), edges (from/to/title/time)
    - kg_visual_timeline: events 按时间排序，包含摘要、时间、实体列表（截断）
    """
    # SQLite 为真相源：优先直接从 SQLite 构建快照（不依赖 knowledge_graph.json）
    try:
        from ...adapters.sqlite.store import get_store
        import sqlite3

        store = get_store()
        db_path = str(getattr(tools, "SQLITE_DB_FILE", None) or "")
        if not db_path:
            raise RuntimeError("SQLITE_DB_FILE not configured")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            # entity_id -> name
            ent_name = {
                str(r["entity_id"]): str(r["name"])
                for r in conn.execute("SELECT entity_id, name FROM entities").fetchall()
            }
            # event_id -> (abstract, summary, time)
            ev_rows = conn.execute(
                "SELECT event_id, abstract, event_summary, event_start_time, reported_at, first_seen FROM events"
            ).fetchall()
            evt_by_id = {}
            for r in ev_rows:
                abstract = str(r["abstract"] or "")
                if not abstract:
                    continue
                evt_by_id[str(r["event_id"])] = {
                    "abstract": abstract,
                    "event_summary": str(r["event_summary"] or "") or abstract,
                    "time": str(r["event_start_time"] or "").strip()
                    or str(r["reported_at"] or "").strip()
                    or str(r["first_seen"] or "").strip(),
                }

            # edges（entity-event / relation / event_edge），全部带 time（缺失兜底）
            edges = []

            # participants -> entity-event
            for r in conn.execute(
                "SELECT event_id, entity_id, time FROM participants"
            ).fetchall():
                eid = str(r["event_id"])
                ent = ent_name.get(str(r["entity_id"]), "")
                evt = evt_by_id.get(eid)
                if not ent or not evt:
                    continue
                t = str(r["time"] or "").strip() or str(evt.get("time") or "").strip() or datetime.now(timezone.utc).isoformat()
                edges.append({"from": ent, "to": f"EVT:{evt['abstract']}", "type": "involved_in", "title": evt["event_summary"], "time": t})

            # relations -> entity-entity
            for r in conn.execute(
                "SELECT subject_entity_id, predicate, object_entity_id, time FROM relations"
            ).fetchall():
                s = ent_name.get(str(r["subject_entity_id"]), "")
                o = ent_name.get(str(r["object_entity_id"]), "")
                p = str(r["predicate"] or "").strip()
                if not s or not o or not p:
                    continue
                t = str(r["time"] or "").strip() or datetime.now(timezone.utc).isoformat()
                edges.append({"from": s, "to": o, "type": "relation", "title": p, "time": t, "predicate": p})

            # event_edges -> event-event
            for r in conn.execute(
                "SELECT from_event_id, to_event_id, edge_type, time FROM event_edges"
            ).fetchall():
                a = evt_by_id.get(str(r["from_event_id"]))
                b = evt_by_id.get(str(r["to_event_id"]))
                if not a or not b:
                    continue
                edge_type = str(r["edge_type"] or "related")
                t = str(r["time"] or "").strip() or str(a.get("time") or "").strip() or datetime.now(timezone.utc).isoformat()
                edges.append({"from": f"EVT:{a['abstract']}", "to": f"EVT:{b['abstract']}", "type": "event_edge", "edge_type": edge_type, "title": edge_type, "time": t})

            # nodes：由边推导
            deg = {}
            for e in edges:
                u, v = e.get("from"), e.get("to")
                if not u or not v:
                    continue
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1

            # 时间窗过滤（按 edge.time）
            cutoff = None
            if days_window and days_window > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days_window)

            def _parse_dt(val: str):
                if not val:
                    return None
                try:
                    dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return None

            filtered_edges = []
            for e in edges:
                if cutoff:
                    dt = _parse_dt(str(e.get("time") or ""))
                    if not dt or dt < cutoff:
                        continue
                filtered_edges.append(e)

            # 分别选 Top 实体/事件，确保两类都能进入可视化
            event_nodes = [n for n in deg.keys() if isinstance(n, str) and n.startswith("EVT:")]
            entity_nodes = [n for n in deg.keys() if not (isinstance(n, str) and n.startswith("EVT:"))]
            top_evt = set(sorted(event_nodes, key=deg.get, reverse=True)[: int(top_events)])
            top_ent = set(sorted(entity_nodes, key=deg.get, reverse=True)[: int(top_entities)])
            top_ids = top_evt | top_ent

            # 构建节点
            vis_nodes = []
            for nid in top_ids:
                if isinstance(nid, str) and nid.startswith("EVT:"):
                    abs_key = nid[4:]
                    # 反向查 summary
                    summary = abs_key
                    for ev in evt_by_id.values():
                        if ev.get("abstract") == abs_key:
                            summary = ev.get("event_summary") or abs_key
                            break
                    vis_nodes.append({"id": nid, "label": str(summary)[:max_title_len], "type": "event", "color": "#ff7f0e"})
                else:
                    vis_nodes.append({"id": nid, "label": str(nid), "type": "entity", "color": "#1f77b4"})

            # 构建边
            vis_edges = []
            for e in filtered_edges:
                u, v = e.get("from"), e.get("to")
                if u in top_ids and v in top_ids:
                    title = str(e.get("title") or "")[:max_title_len]
                    edge_out = {"from": u, "to": v, "title": title}
                    if e.get("time"):
                        edge_out["time"] = e.get("time")
                    if e.get("type") == "event_edge" and e.get("edge_type"):
                        edge_out["edge_type"] = e.get("edge_type")
                    vis_edges.append(edge_out)

            out_graph = Path(out_path_graph)
            ensure_dir(out_graph.parent)
            graph_data = {"nodes": vis_nodes, "edges": vis_edges}
            write_json_file(out_graph, graph_data, ensure_ascii=False, indent=None)

            # 时间线快照（包含 abstract，修复 KG 页依赖）
            parts = conn.execute(
                "SELECT event_id, entity_id, time FROM participants"
            ).fetchall()
            ents_by_evt = {}
            for p in parts:
                ents_by_evt.setdefault(str(p["event_id"]), set()).add(ent_name.get(str(p["entity_id"]), ""))

            tl_rows = []
            for eid, ev in evt_by_id.items():
                ts = str(ev.get("time") or "").strip()
                dt = _parse_dt(ts)
                if not dt:
                    continue
                if cutoff and dt < cutoff:
                    continue
                tl_rows.append(
                    {
                        "time": ts,
                        "abstract": ev.get("abstract", ""),
                        "event_summary": str(ev.get("event_summary") or ev.get("abstract") or "")[:max_title_len],
                        "entities": [x for x in list(ents_by_evt.get(eid, set())) if x][:20],
                    }
                )

            tl_rows = sorted(tl_rows, key=lambda x: x["time"])[: int(top_events)]
            out_tl = Path(out_path_timeline)
            ensure_dir(out_tl.parent)
            write_json_file(out_tl, tl_rows, ensure_ascii=False, indent=None)

            return {"status": "success", "graph": str(out_graph), "timeline": str(out_tl)}
        finally:
            conn.close()
    except Exception:
        # 若 SQLite 不可用，回退旧逻辑：从 knowledge_graph.json 读取
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_window)

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
            ts = evt.get("event_start_time") or evt.get("first_seen") or evt.get("published_at")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt = None
            if dt and cutoff and dt < cutoff:
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
            if e.get("type") == "event_edge":
                title = (str(e.get("edge_type") or "related"))[:max_title_len]
            elif isinstance(v, str) and v.startswith("EVT:"):
                evt_key = v[4:]
                title = (events.get(evt_key, {}) or {}).get("event_summary", "")[:max_title_len]
            else:
                # 对实体-实体关系边显示谓词
                title = (e.get("predicate") or e.get("title") or "")[:max_title_len]
            edge_out = {"from": u, "to": v, "title": title}
            # 关键：把“每条元组的时间”传递给前端快照（如果有）
            if e.get("time"):
                edge_out["time"] = e.get("time")
            vis_edges.append(edge_out)

    out_graph = Path(out_path_graph)
    ensure_dir(out_graph.parent)
    graph_data = {"nodes": vis_nodes, "edges": vis_edges}
    write_json_file(out_graph, graph_data, ensure_ascii=False, indent=None)

    # 时间线快照（取 TopN 事件按时间排序）
    tl_rows = []
    for evt_key, evt in events.items():
        ts = evt.get("event_start_time") or evt.get("first_seen") or evt.get("published_at")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
            if dt and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
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
    ensure_dir(out_tl.parent)
    write_json_file(out_tl, tl_rows, ensure_ascii=False, indent=None)

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
async def append_only_update_graph_tool(
    events_list: Any,
    default_source: str = "auto_pipeline",
    allow_append_original_forms: bool = True
) -> Dict[str, int]:
    """
    包装 agent3.append_only_update_graph：
    - 仅当实体/事件不存在时新增
    - 不改旧 first_seen / sources / 摘要；可选是否为旧实体追加 original_forms
    - 使用标准事件数据管道进行处理
    """
    # 使用标准事件数据管道
    pipeline = StandardEventPipeline()

    # 执行数据管道处理
    pipeline_result = await pipeline.execute(events_list)

    # 获取序列化后的数据
    processed_data = pipeline_result.get("serialization", [])
    if isinstance(processed_data, str):
        # 如果是序列化的JSON字符串，需要解析回来
        import json
        processed_data = json.loads(processed_data)

    kg = KnowledgeGraph()
    return kg.append_only_update(
        events_list=processed_data,
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
    读取 tmp 中的提取结果 jsonl 文件并调用 append_only_update_graph 追加到实体/事件库.
    max_files=0 表示全部; >0 时按修改时间倒序取前 N 个.
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
    kg = KnowledgeGraph()
    res = kg.append_only_update(
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


class KnowledgeGraph:
    """
    压缩知识图谱系统，用于管理实体和事件，支持重复检测和更新。
    """

    def __init__(self):
        self.entities_file = tools.ENTITIES_FILE
        self.events_file = tools.EVENTS_FILE
        self.abstract_map_file = tools.ABSTRACT_MAP_FILE
        self.entities_tmp_file = tools.ENTITIES_TMP_FILE
        self.abstract_tmp_file = tools.ABSTRACT_TMP_FILE
        self.kg_file = tools.KNOWLEDGE_GRAPH_FILE
        self.merge_rules_file = tools.CONFIG_DIR / "entity_merge_rules.json" # 规则文件路径
        self.merge_rules = {} # 内存中的规则缓存
        self.settings = {}  # 延迟加载配置
        self.graph = {
            "entities": {},  # 实体ID -> 实体信息
            "events": {},   # 事件摘要 -> 事件信息
            "edges": []     # 边列表，连接实体和事件
        }
        self.llm_pool = None  # 延迟初始化
        self._tmp_loaded = []  # 记录已加载的tmp文件，刷新完成后清理
        self._load_merge_rules() # 初始化时加载规则
    def _init_llm_pool(self):
        """初始化LLM API池"""
        if self.llm_pool is None:
            try:
                self.llm_pool = get_llm_pool()
                tools.log("[知识图谱] LLM API池初始化成功")
            except Exception as e:
                tools.log(f"[知识图谱] ❌ 初始化LLM API池失败: {e}")
                self.llm_pool = None

    def load_data(self) -> bool:
        """加载实体与事件数据（仅使用SQLite主存储）"""
        try:
            # 从 SQLite 主存储加载数据
            from src.adapters.sqlite.store import get_store
            store = get_store()
            self.graph["entities"] = store.export_entities_json() or {}
            abstract_map = store.export_abstract_map_json() or {}

            # 转换 abstract_map 为事件格式
            self.graph["events"] = {
                abstract: {
                    "event_id": data.get("event_id", ""),
                    "abstract": abstract,
                    "entities": data.get("entities", []) or [],
                    "event_summary": data.get("event_summary", ""),
                    "event_types": data.get("event_types", []),
                    "entity_roles": data.get("entity_roles", {}),
                    "relations": data.get("relations", []),
                    "event_start_time": data.get("event_start_time", ""),
                    "event_start_time_text": data.get("event_start_time_text", ""),
                    "event_start_time_precision": data.get("event_start_time_precision", "unknown"),
                    "reported_at": data.get("reported_at", ""),
                    "sources": data.get("sources", []),
                    "first_seen": data.get("first_seen", ""),
                }
                for abstract, data in (abstract_map or {}).items()
                if isinstance(abstract, str) and isinstance(data, dict)
            }

            # 额外加载 tmp 新增数据（未合并的新增实体/事件）
            self._load_tmp_entities()
            self._load_tmp_events()

            self._build_edges()
            tools.log(f"[知识图谱] 数据加载成功: {len(self.graph['entities'])} 实体, {len(self.graph['events'])} 事件")
            return True
        except Exception as e:
            tools.log(f"[知识图谱] ❌ 加载数据失败: {e}")
            return False

    def _build_edges(self):
        """构建实体和事件之间的边（所有元组/边强制带时间 time）"""
        self.graph['edges'] = []
        for abstract, event in self.graph['events'].items():
            # 统一事件节点ID：EVT:<abstract>（兼容现有可视化与过滤逻辑）
            evt_node = f"EVT:{abstract}"

            # 边时间：优先 event_start_time，其次 reported_at/first_seen
            event_time = ""
            if isinstance(event.get("event_start_time"), str) and event.get("event_start_time").strip():
                event_time = event.get("event_start_time").strip()
            elif isinstance(event.get("reported_at"), str) and event.get("reported_at").strip():
                event_time = event.get("reported_at").strip()
            elif isinstance(event.get("first_seen"), str) and event.get("first_seen").strip():
                event_time = event.get("first_seen").strip()
            if not event_time:
                event_time = datetime.now(timezone.utc).isoformat()

            for entity in event.get('entities', []):
                if entity in self.graph['entities']:
                    roles = []
                    roles_map = event.get("entity_roles", {})
                    if isinstance(roles_map, dict):
                        r = roles_map.get(entity, [])
                        if isinstance(r, str):
                            roles = [r]
                        elif isinstance(r, list):
                            roles = [x for x in r if isinstance(x, str) and x.strip()]
                    self.graph['edges'].append({
                        "from": entity,
                        "to": evt_node,
                        "type": "involved_in",
                        "roles": roles,
                        "time": event_time
                    })

            # 额外构建实体-关系-实体边（带事件上下文）
            rels = event.get("relations", [])
            if isinstance(rels, list):
                for rel in rels:
                    if not isinstance(rel, dict):
                        continue
                    s = rel.get("subject")
                    p = rel.get("predicate")
                    o = rel.get("object")
                    if not isinstance(s, str) or not isinstance(p, str) or not isinstance(o, str):
                        continue
                    s, p, o = s.strip(), p.strip(), o.strip()
                    if not s or not p or not o:
                        continue
                    if s not in self.graph["entities"] or o not in self.graph["entities"]:
                        continue
                    rel_time = ""
                    if isinstance(rel.get("time"), str) and rel.get("time").strip():
                        rel_time = rel.get("time").strip()
                    if not rel_time:
                        rel_time = event_time
                    self.graph["edges"].append({
                        "from": s,
                        "to": o,
                        "type": "relation",
                        "predicate": p,
                        "event": abstract,
                        "time": rel_time
                    })

        # =========================
        # 事件-事件演化边：从 SQLite event_edges 投影到 knowledge_graph.json
        # 说明：event_edges 以 event_id 存储；这里映射到 abstract 再输出 EVT:<abstract> 供前端展示。
        # =========================
        try:
            from src.adapters.sqlite.store import canonical_event_id
            import sqlite3
            import json as _json

            # event_id -> abstract（优先选 canonical：sha1(evt:<abstract>) == event_id）
            evtid_to_abs = {}
            for abs_key, ev in (self.graph.get("events") or {}).items():
                if not isinstance(ev, dict):
                    continue
                eid = str(ev.get("event_id") or "").strip()
                if not eid:
                    continue
                # canonical 优先
                if canonical_event_id(abs_key) == eid:
                    evtid_to_abs[eid] = abs_key
                else:
                    evtid_to_abs.setdefault(eid, abs_key)

            db_path = str(getattr(tools, "SQLITE_DB_FILE", None) or "")
            if db_path:
                conn = sqlite3.connect(db_path)
                try:
                    rows = conn.execute(
                        """
                        SELECT from_event_id, to_event_id, edge_type, time, reported_at, confidence, evidence_json
                        FROM event_edges
                        """
                    ).fetchall()
                finally:
                    conn.close()

                for from_id, to_id, edge_type, t, rep, conf, ev_json in rows:
                    from_id = str(from_id or "")
                    to_id = str(to_id or "")
                    if from_id not in evtid_to_abs or to_id not in evtid_to_abs:
                        continue
                    from_abs = evtid_to_abs[from_id]
                    to_abs = evtid_to_abs[to_id]
                    try:
                        evidence = _json.loads(ev_json or "[]")
                        if not isinstance(evidence, list):
                            evidence = []
                    except Exception:
                        evidence = []
                    time_val = str(t or "").strip() or datetime.now(timezone.utc).isoformat()
                    self.graph["edges"].append(
                        {
                            "from": f"EVT:{from_abs}",
                            "to": f"EVT:{to_abs}",
                            "type": "event_edge",
                            "edge_type": str(edge_type or "related"),
                            "time": time_val,
                            "reported_at": str(rep or ""),
                            "confidence": float(conf or 0.0),
                            "evidence": [x for x in evidence if isinstance(x, str) and x.strip()],
                        }
                    )
        except Exception:
            # SQLite 不存在/表不存在/解析失败时，忽略事件演化边
            pass

    def build_graph(self) -> bool:
        """构建知识图谱"""
        return self.load_data()

    def _load_merge_rules(self):
        """加载实体合并规则"""
        if self.merge_rules_file.exists():
            try:
                with open(self.merge_rules_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.merge_rules = data.get("merge_rules", {})
                tools.log(f"[知识图谱] 已加载 {len(self.merge_rules)} 条实体合并规则")
            except Exception as e:
                tools.log(f"[知识图谱] ⚠️ 加载合并规则失败: {e}")
        else:
            self.merge_rules = {}

    def _load_tmp_entities(self):
        """加载并合并 tmp 实体数据"""
        if self.entities_tmp_file.exists():
            try:
                with open(self.entities_tmp_file, "r", encoding="utf-8") as f:
                    tmp_entities = json.load(f)
                    for name, data in tmp_entities.items():
                        if name in self.graph['entities']:
                            self._merge_entity_record(self.graph['entities'][name], data)
                        else:
                            self.graph['entities'][name] = data
                self._tmp_loaded.append(self.entities_tmp_file)
                tools.log(f"[知识图谱] 已加载 tmp 实体 {len(tmp_entities)} 条")
            except Exception as e:
                tools.log(f"[知识图谱] ⚠️ 加载 tmp 实体失败: {e}")

    def _load_tmp_events(self):
        """加载并合并 tmp 事件数据"""
        if self.abstract_tmp_file.exists():
            try:
                with open(self.abstract_tmp_file, "r", encoding="utf-8") as f:
                    tmp_events = json.load(f)
                    for abstract, data in tmp_events.items():
                        if abstract in self.graph['events']:
                            self._merge_event_record(self.graph['events'][abstract], data)
                        else:
                            self.graph['events'][abstract] = {
                                "abstract": abstract,
                                "entities": data.get("entities", []),
                                "event_summary": data.get("event_summary", ""),
                                "event_types": data.get("event_types", []),
                                "entity_roles": data.get("entity_roles", {}),
                                "relations": data.get("relations", []),
                                "event_start_time": data.get("event_start_time", ""),
                                "event_start_time_text": data.get("event_start_time_text", ""),
                                "event_start_time_precision": data.get("event_start_time_precision", "unknown"),
                                "sources": data.get("sources", []),
                                "first_seen": data.get("first_seen", "")
                            }
                self._tmp_loaded.append(self.abstract_tmp_file)
                tools.log(f"[知识图谱] 已加载 tmp 事件 {len(tmp_events)} 条")
            except Exception as e:
                tools.log(f"[知识图谱] ⚠️ 加载 tmp 事件失败: {e}")

    def _cleanup_tmp_files(self):
        """刷新完成后清理已加载的 tmp 文件"""
        for path in self._tmp_loaded:
            try:
                path.unlink(missing_ok=True)
                tools.log(f"[知识图谱] 🗑️ 已清理 tmp 文件: {path}")
            except Exception as e:
                tools.log(f"[知识图谱] ⚠️ 无法删除 tmp 文件 {path}: {e}")
        self._tmp_loaded = []

    def _save_merge_rules(self):
        """保存实体合并规则"""
        try:
            data = {
                "merge_rules": self.merge_rules,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            with open(self.merge_rules_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tools.log(f"[知识图谱] 已保存合并规则库 (共 {len(self.merge_rules)} 条)")
        except Exception as e:
            tools.log(f"[知识图谱] ❌ 保存合并规则失败: {e}")

    def _merge_entity_record(self, target: Dict[str, Any], source: Dict[str, Any]):
        """将source实体信息合并到target，不删除节点"""
        if not source:
            return
        # sources
        primary_sources = set()
        for s in target.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        for s in source.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        target['sources'] = list(primary_sources)

        # original_forms
        primary_forms = set()
        for f in target.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)
        for f in source.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)
        target['original_forms'] = list(primary_forms)

        # first_seen 取最早
        primary_first = target.get('first_seen', '')
        source_first = source.get('first_seen', '')
        if source_first and (not primary_first or source_first < primary_first):
            target['first_seen'] = source_first

    def _merge_event_record(self, target: Dict[str, Any], source: Dict[str, Any]):
        """将source事件信息合并到target，不删除节点"""
        if not source:
            return
        # sources
        primary_sources = set()
        for s in target.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        for s in source.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        target['sources'] = list(primary_sources)

        # entities union
        ents = set(target.get('entities', []))
        ents.update(source.get('entities', []))
        target['entities'] = list(ents)

        # event_types union（保序）
        def _merge_unique_list(a: Any, b: Any) -> List[str]:
            out: List[str] = []
            seen = set()
            for it in (a if isinstance(a, list) else []):
                if isinstance(it, str) and it and it not in seen:
                    seen.add(it)
                    out.append(it)
            for it in (b if isinstance(b, list) else []):
                if isinstance(it, str) and it and it not in seen:
                    seen.add(it)
                    out.append(it)
            return out

        src_types = source.get("event_types", [])
        if src_types:
            merged_types = _merge_unique_list(target.get("event_types", []), src_types)
            target["event_types"] = merged_types

        # entity_roles merge
        def _merge_roles(a: Any, b: Any) -> Dict[str, List[str]]:
            out: Dict[str, List[str]] = {}
            if isinstance(a, dict):
                for k, v in a.items():
                    if isinstance(k, str) and isinstance(v, list):
                        out[k] = [x for x in v if isinstance(x, str) and x]
            if isinstance(b, dict):
                for k, v in b.items():
                    if not isinstance(k, str):
                        continue
                    roles_list: List[str] = []
                    if isinstance(v, str):
                        roles_list = [v]
                    elif isinstance(v, list):
                        roles_list = [x for x in v if isinstance(x, str)]
                    roles_list = [x.strip() for x in roles_list if isinstance(x, str) and x.strip()]
                    if not roles_list:
                        continue
                    out[k] = _merge_unique_list(out.get(k, []), roles_list)
            return out

        src_roles = source.get("entity_roles", {})
        if src_roles:
            target["entity_roles"] = _merge_roles(target.get("entity_roles", {}), src_roles)

        # relations merge
        def _merge_relations(a: Any, b: Any) -> List[Dict[str, Any]]:
            out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            def ns(x: Any) -> str:
                return x.strip() if isinstance(x, str) else ""
            def add_one(rel: Any):
                if not isinstance(rel, dict):
                    return
                s = ns(rel.get("subject", ""))
                p = ns(rel.get("predicate", ""))
                o = ns(rel.get("object", ""))
                if not s or not p or not o:
                    return
                key = (s, p, o)
                ev = rel.get("evidence", [])
                if key not in out:
                    out[key] = {"subject": s, "predicate": p, "object": o, "evidence": []}
                # evidence 允许 str 或 list[str]
                ev_list: List[str] = []
                if isinstance(ev, str):
                    ev_list = [ev]
                elif isinstance(ev, list):
                    ev_list = [x for x in ev if isinstance(x, str)]
                ev_list = [x.strip() for x in ev_list if isinstance(x, str) and x.strip()]
                if ev_list:
                    cur = out[key].get("evidence", [])
                    if not isinstance(cur, list):
                        cur = []
                    for x in ev_list:
                        if x not in cur:
                            cur.append(x)
                    out[key]["evidence"] = cur
            for rel in (a if isinstance(a, list) else []):
                add_one(rel)
            for rel in (b if isinstance(b, list) else []):
                add_one(rel)
            return list(out.values())

        src_rels = source.get("relations", [])
        if src_rels:
            target["relations"] = _merge_relations(target.get("relations", []), src_rels)

        # event_start_time choose better
        def _choose_better_time(cur_t: Any, cur_txt: Any, cur_p: Any, inc_t: Any, inc_txt: Any, inc_p: Any) -> Tuple[str, str, str]:
            def ns(x: Any) -> str:
                return x.strip() if isinstance(x, str) else ""
            ct, ctxt, cp = ns(cur_t), ns(cur_txt), (ns(cur_p) or "unknown")
            it, itxt, ip = ns(inc_t), ns(inc_txt), (ns(inc_p) or "unknown")
            if not ct and it:
                return it, itxt, ip
            if ct and not it:
                return ct, ctxt, cp
            if not ct and not it:
                return "", "", cp
            rank = {"exact": 3, "date": 2, "relative": 1, "unknown": 0}
            rc, ri = rank.get(cp, 0), rank.get(ip, 0)
            if ri > rc:
                return it, itxt, ip
            if ri < rc:
                return ct, ctxt, cp
            if len(ct) >= 10 and ct[4:5] == "-" and ct[7:8] == "-" and len(it) >= 10 and it[4:5] == "-" and it[7:8] == "-":
                if it < ct:
                    return it, itxt, ip
            return ct, ctxt, cp

        nt, ntxt, np = _choose_better_time(
            target.get("event_start_time", ""),
            target.get("event_start_time_text", ""),
            target.get("event_start_time_precision", "unknown"),
            source.get("event_start_time", ""),
            source.get("event_start_time_text", ""),
            source.get("event_start_time_precision", "unknown"),
        )
        target["event_start_time"] = nt
        target["event_start_time_text"] = ntxt
        target["event_start_time_precision"] = np

        # first_seen 取最早
        primary_first = target.get('first_seen', '')
        source_first = source.get('first_seen', '')
        if source_first and (not primary_first or source_first < primary_first):
            target['first_seen'] = source_first

        # event_summary 若缺失则补
        if not target.get('event_summary') and source.get('event_summary'):
            target['event_summary'] = source['event_summary']

    def _ensure_settings_loaded(self):
        """确保配置已加载"""
        if not self.settings:
            self.settings = self._load_agent3_settings()

    def _load_agent3_settings(self) -> Dict[str, Any]:
        """
        加载agent3相关配置，使用统一配置管理器。
        """
        config_manager = get_config_manager()

        # 使用ConfigManager获取配置，默认值在ConfigManager中处理
        try:
            settings = {
                "entity_batch_size": config_manager.get_config_value("entity_batch_size", 80, "agent3_config"),
                "event_batch_size": config_manager.get_config_value("event_batch_size", 15, "agent3_config"),
                "event_bucket_days": config_manager.get_config_value("event_bucket_days", 7, "agent3_config"),
                "event_bucket_entity_overlap": config_manager.get_config_value("event_bucket_entity_overlap", 1, "agent3_config"),
                "event_bucket_max_size": config_manager.get_config_value("event_bucket_max_size", 40, "agent3_config"),
                "event_precluster_similarity": config_manager.get_config_value("event_precluster_similarity", 0.82, "agent3_config"),
                "event_precluster_limit": config_manager.get_config_value("event_precluster_limit", 300, "agent3_config"),
                "entity_precluster_similarity": config_manager.get_config_value("entity_precluster_similarity", 0.93, "agent3_config"),
                "entity_precluster_limit": config_manager.get_config_value("entity_precluster_limit", 500, "agent3_config"),
                "max_summary_chars": config_manager.get_config_value("max_summary_chars", 360, "agent3_config"),
                "entity_max_workers": config_manager.get_config_value("entity_max_workers", 3, "agent3_config"),
                "event_max_workers": config_manager.get_config_value("event_max_workers", 3, "agent3_config"),
                "rate_limit_per_sec": config_manager.get_config_value("rate_limit_per_sec", 1.0, "agent3_config"),
                "entity_evidence_per_entity": config_manager.get_config_value("entity_evidence_per_entity", 2, "agent3_config"),
                "entity_evidence_max_chars": config_manager.get_config_value("entity_evidence_max_chars", 400, "agent3_config"),
            }
            return settings
        except Exception as e:
            # 返回默认值
            return {
                "entity_batch_size": 80,
                "event_batch_size": 15,
                "event_bucket_days": 7,
                "event_bucket_entity_overlap": 1,
                "event_bucket_max_size": 40,
                "event_precluster_similarity": 0.82,
                "event_precluster_limit": 300,
                "entity_precluster_similarity": 0.93,
                "entity_precluster_limit": 500,
                "max_summary_chars": 360,
                "entity_max_workers": 3,
                "event_max_workers": 3,
                "rate_limit_per_sec": 1.0,
                "entity_evidence_per_entity": 2,
                "entity_evidence_max_chars": 400,
            }
    def _string_similarity(self, a: str, b: str) -> float:
        """
        字符串相似度(0-1)
        
        使用混合策略提高准确度：
        1. 完全匹配 → 1.0
        2. 归一化匹配（忽略大小写/空格/标点）→ 0.98
        3. Jaro-Winkler算法（适合短字符串和公司名）
        4. 语义相似度（如果可用）——支持跨语言匹配
        5. 中英文混合实体特殊处理
        """
        import re
        
        # 快速路径：完全相同
        if a == b:
            return 1.0
        
        # 归一化：去除空格、标点、转小写
        def normalize(s: str) -> str:
            # 去除所有非字母数字字符
            s = re.sub(r'[^\w]', '', s, flags=re.UNICODE)
            return s.lower()
        
        a_norm = normalize(a)
        b_norm = normalize(b)
        
        # 归一化后相同
        if a_norm and b_norm and a_norm == b_norm:
            return 0.98
        
        # 尝试导入Jaro-Winkler（更适合实体名称）
        try:
            from jellyfish import jaro_winkler_similarity
            jw_score = jaro_winkler_similarity(a.lower(), b.lower())
        except ImportError:
            # 降级到SequenceMatcher
            jw_score = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        
        # 中英文混合实体：尝试语义匹配，否则大幅降权
        a_is_chinese = self._is_chinese(a)
        b_is_chinese = self._is_chinese(b)
        
        if a_is_chinese != b_is_chinese:
            # 一个中文一个英文：尝试语义匹配
            try:
                from ...infra.semantic_matcher import get_semantic_matcher
                semantic_matcher = get_semantic_matcher()
                if semantic_matcher.is_available():
                    semantic_score = semantic_matcher.similarity(a, b)
                    if semantic_score is not None:
                        # 语义相似度优先，但要与字符串相似度加权平均
                        # 语义相似度70%权重 + 字符串相似度30%权重
                        return semantic_score * 0.7 + jw_score * 0.3
            except Exception as e:
                # 语义匹配失败，降级到字符串匹配
                pass
            
            # 如果语义匹配不可用，降权70%
            return jw_score * 0.3
        
        return jw_score

    def _is_chinese(self, text: str) -> bool:
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _entity_type(self, name: str) -> str:
        """
        轻量类型判定：geo/org/person/product/unknown
        仅用于阻断跨类型合并，宁可 unknown。
        """
        n = name or ""
        geo_kw = ["市", "省", "州", "县", "区", "镇", "乡", "郡", "岛", "府", "道", "自治", "共和国", "王国", "特区", "自治区"]
        org_kw = ["公司", "集团", "银行", "政府", "部", "局", "署", "院", "厅", "行", "党", "机构", "法院", "检察院", "委员会", "组织", "联盟", "理事会", "协会", "基金会", "大学", "学院", "学校", "工厂", "厂", "报", "电视", "新闻", "日报", "晚报", "周报", "社"]
        product_kw = ["系列", "版", "型", "型号", "Pro", "Ultra"]
        if any(k in n for k in geo_kw):
            return "geo"
        if any(k in n for k in org_kw):
            return "org"
        if " " in n or "·" in n:
            return "person"
        if any(k in n for k in product_kw):
            return "product"
        return "unknown"

    def _valid_entity_group(self, group: List[str]) -> bool:
        """使用LLM决策器判断实体组是否可以合并"""
        # 收集证据
        evidence_map = self._collect_entity_evidence(group)
        
        # 使用LLM决策器进行判断
        if self.llm_pool:
            try:
                from ...adapters.extraction import LLMEntityMergeDecider
                decider = LLMEntityMergeDecider(self.llm_pool)
                decision = decider.decide_entity_merges(group, evidence_map)
                
                # 如果有合并组，说明可以合并
                # 如果没有合并组，说明不应该合并
                can_merge = len(decision.merge_groups) > 0
                
                if not can_merge:
                    tools.log(f"[知识图谱] ⚠️ LLM决策器阻止合并: {group}")
                    # 记录详细分析结果
                    for analysis in decision.entity_analyses:
                        tools.log(f"[知识图谱] 实体分析 - {analysis.entity}: 类型={analysis.entity_type.value}, 置信度={analysis.confidence}, 理由={analysis.reasoning}")
                
                return can_merge
            except Exception as e:
                tools.log(f"[知识图谱] ⚠️ LLM决策器调用失败，回退到硬编码逻辑: {e}")
                # 回退到原有的硬编码逻辑
                types = set()
                for name in group:
                    t = self._entity_type(name)
                    if t != "unknown":
                        types.add(t)
                # 如果检测到多种已知类型则视为高风险
                if len(types) > 1:
                    tools.log(f"[知识图谱] ⚠️ 跨类型合并被阻止: {group} | types={types}")
                    return False
                return True
        else:
            # 如果没有LLM池，回退到原有的硬编码逻辑
            types = set()
            for name in group:
                t = self._entity_type(name)
                if t != "unknown":
                    types.add(t)
            # 如果检测到多种已知类型则视为高风险
            if len(types) > 1:
                tools.log(f"[知识图谱] ⚠️ 跨类型合并被阻止: {group} | types={types}")
                return False
            return True

    def _collect_entity_evidence(self, entities_batch: List[str]) -> Dict[str, List[str]]:
        """
        为实体批次收集相关事件摘要+实体，减少幻觉。
        """
        self._ensure_settings_loaded()
        per_entity = int(self.settings.get("entity_evidence_per_entity", 2))
        max_chars = int(self.settings.get("entity_evidence_max_chars", 400))
        evidence: Dict[str, List[str]] = {e: [] for e in entities_batch}
        for abstract, event in self.graph['events'].items():
            ents = event.get('entities', [])
            summary = self._trim_text(event.get('event_summary', "") or abstract, max_chars)
            for e in ents:
                if e in evidence and len(evidence[e]) < per_entity:
                    evidence[e].append(f"{abstract} | {', '.join(ents)} | {summary}")
        return evidence

    def _trim_text(self, text: str, max_chars: int) -> str:
        """控制文本长度，避免prompt过长"""
        if not text or max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _parse_time(self, ts: str) -> float:
        """尽量解析时间戳，失败返回0"""
        if not ts:
            return 0
        try:
            # 支持ISO字符串
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            except Exception:
                return 0

    def _bucket_events_by_time_and_entity(
        self,
        window_days: int,
        min_entity_overlap: int,
        max_bucket_size: int
    ) -> List[Dict[str, Any]]:
        """
        按时间窗口与实体交集分桶事件，减少跨期/跨主体混杂。
        """
        events_items = list(self.graph["events"].items())
        window_sec = window_days * 86400

        # 预排序，时间缺失放末尾
        def _sort_key(item):
            ts = self._parse_time(item[1].get("first_seen", ""))
            return ts if ts > 0 else float("inf")

        events_items.sort(key=_sort_key)

        buckets: List[Dict[str, Any]] = []
        for abstract, event in events_items:
            entities = set(event.get("entities", []))
            ts = self._parse_time(event.get("first_seen", ""))
            placed = False
            for bucket in buckets:
                if len(bucket["keys"]) >= max_bucket_size:
                    continue
                # 时间窗口判定
                if bucket["min_time"] and ts and ts < bucket["min_time"] - window_sec:
                    continue
                if bucket["max_time"] and ts and ts > bucket["max_time"] + window_sec:
                    continue
                # 实体交集判定
                if min_entity_overlap > 0:
                    if not entities or len(bucket["entities"].intersection(entities)) < min_entity_overlap:
                        continue
                # 命中，加入桶
                bucket["keys"].append(abstract)
                bucket["entities"].update(entities)
                if ts:
                    bucket["min_time"] = min(bucket["min_time"] or ts, ts)
                    bucket["max_time"] = max(bucket["max_time"] or ts, ts)
                placed = True
                break
            if not placed:
                buckets.append({
                    "keys": [abstract],
                    "entities": set(entities),
                    "min_time": ts if ts else None,
                    "max_time": ts if ts else None
                })
        tools.log(f"[知识图谱] 事件分桶完成，共 {len(buckets)} 个桶")
        return buckets

    def _precluster_entities_by_string(self, entities: List[str], threshold: float, limit: int) -> List[List[str]]:
        """
        基于字符串相似度的轻量预聚类，避免LLM过量输入。
        
        ⚠️ 重要：预聚类仅做粗筛，高相似度（>=threshold）的才进入候选组
        所有候选组必须经过LLM二次验证才能真正合并
        """
        if len(entities) == 0 or len(entities) > limit:
            return []
        res = []
        used = set()
        for i, ent in enumerate(entities):
            if ent in used:
                continue
            group = [ent]
            used.add(ent)
            for other in entities[i+1:]:
                if other in used:
                    continue
                # 字符串相似度检查
                if self._string_similarity(ent, other) >= threshold:
                    group.append(other)
                    used.add(other)
            if len(group) > 1:
                res.append(group)
        if res:
            tools.log(f"[知识图谱] 本地实体预聚类发现 {len(res)} 组可能重复（需LLM二次验证）")
        return res

    def _precluster_events_by_string(
        self,
        events_map: Dict[str, Dict[str, Any]],
        keys: List[str],
        threshold: float,
        limit: int,
        max_summary_chars: int
    ) -> List[List[str]]:
        """
        同桶事件的字符串近似聚类，减少LLM负担。
        """
        if len(keys) == 0 or len(keys) > limit:
            return []

        def norm_text(k: str) -> str:
            evt = events_map.get(k, {})
            summary = self._trim_text(evt.get("event_summary", "") or "", max_summary_chars)
            return (k + " " + summary).lower()

        texts = {k: norm_text(k) for k in keys}
        res = []
        used = set()
        for i, key in enumerate(keys):
            if key in used:
                continue
            base = texts.get(key, "")
            group = [key]
            used.add(key)
            for other in keys[i+1:]:
                if other in used:
                    continue
                if self._string_similarity(base, texts.get(other, "")) >= threshold:
                    group.append(other)
                    used.add(other)
            if len(group) > 1:
                res.append(group)
        if res:
            tools.log(f"[知识图谱] 本地事件预聚类发现 {len(res)} 组可能重复")
        return res

    def _apply_merge_rules(self) -> bool:
        """
        应用本地合并规则
        返回: 是否有更新
        """
        updated = False
        if not self.merge_rules:
            return False

        # 遍历现有实体，看是否匹配规则
        # 注意：需要在遍历时处理，避免字典大小变化问题，通常收集后再处理
        to_merge = []
        for entity in list(self.graph['entities'].keys()):
            if entity in self.merge_rules:
                target = self.merge_rules[entity]
                # 只有当目标实体也存在，或者目标就是我们想要统一到的名称时（这里简化为如果目标在库里或我们决定改名）
                # 为简单起见，我们假设规则是 A -> B，如果 A 存在，就尝试合并到 B。
                # 如果 B 不在库里，就把 A 重命名为 B。
                if target != entity:
                    to_merge.append((target, entity))

        for primary, duplicate in to_merge:
            # 如果目标实体不存在，先重命名
            if primary not in self.graph['entities'] and duplicate in self.graph['entities']:
                self.graph['entities'][primary] = self.graph['entities'][duplicate]
                del self.graph['entities'][duplicate]
                # 更新事件引用
                for abstract, event in self.graph['events'].items():
                    entities = event.get('entities', [])
                    if duplicate in entities:
                        event['entities'] = [primary if e == duplicate else e for e in entities]
                    # 同步更新 entity_roles key（如果存在）
                    roles_map = event.get("entity_roles", {})
                    if isinstance(roles_map, dict) and duplicate in roles_map:
                        dup_roles = roles_map.get(duplicate, [])
                        prim_roles = roles_map.get(primary, [])
                        merged = []
                        seen = set()
                        for r in (prim_roles if isinstance(prim_roles, list) else []):
                            if isinstance(r, str) and r and r not in seen:
                                seen.add(r)
                                merged.append(r)
                        for r in (dup_roles if isinstance(dup_roles, list) else []):
                            if isinstance(r, str) and r and r not in seen:
                                seen.add(r)
                                merged.append(r)
                        if merged:
                            roles_map[primary] = merged
                        roles_map.pop(duplicate, None)
                        event["entity_roles"] = roles_map

                    # 同步更新 relations 主客体
                    rels = event.get("relations", [])
                    if isinstance(rels, list):
                        changed = False
                        new_rels = []
                        for rel in rels:
                            if not isinstance(rel, dict):
                                continue
                            s = rel.get("subject", "")
                            o = rel.get("object", "")
                            if isinstance(s, str) and s.strip() == duplicate:
                                rel["subject"] = primary
                                changed = True
                            if isinstance(o, str) and o.strip() == duplicate:
                                rel["object"] = primary
                                changed = True
                            new_rels.append(rel)
                        if changed:
                            event["relations"] = new_rels
                tools.log(f"[知识图谱][规则] 重命名实体: {duplicate} -> {primary}")
                updated = True
            elif primary in self.graph['entities'] and duplicate in self.graph['entities']:
                # 如果都存在，则合并
                self._merge_entities(primary, duplicate)
                updated = True

        return updated

    def compress_with_llm(self) -> Dict[str, List[str]]:
        """
        使用LLM分析压缩知识图谱，输出重复的实体和事件抽象。
        分批处理以避免上下文超长。
        集成规则优先策略。
        """
        # 首先应用本地规则
        rule_applied = self._apply_merge_rules()
        if rule_applied:
            self._save_data()

        self._init_llm_pool()
        if self.llm_pool is None:
            tools.log("[知识图谱] ❌ LLM不可用，跳过压缩")
            return {"duplicate_entities": [], "duplicate_events": []}

        self._ensure_settings_loaded()
        # 并发与速率控制
        rate_limit = float(self.settings.get("rate_limit_per_sec", 1.0))
        limiter = RateLimiter(rate_limit) if rate_limit > 0 else None        # 处理实体和事件
        all_duplicate_entities = self._compress_entities_with_llm(limiter)
        all_duplicate_events = self._compress_events_with_llm(limiter)

        return {
            "duplicate_entities": all_duplicate_entities,
            "duplicate_events": all_duplicate_events
        }

    def _compress_entities_with_llm(self, limiter: Optional[RateLimiter]) -> List[List[str]]:
        """使用LLM压缩实体
        
        优化策略：
        1. 预聚类：找出高度相似的实体组（threshold=0.93）
        2. 批处理：只将预聚类组送给LLM二次验证，而不是所有实体
        这样可以避免不相关实体被错误分组
        """
        self._ensure_settings_loaded()
        all_duplicate_entities = []
        # 排序以增加相似实体相邻的概率
        entities_list = sorted(list(self.graph['entities'].keys()))
        if not entities_list:
            return all_duplicate_entities

        # 预聚类：找出高度相似的实体组
        precluster_entities = self._precluster_entities_by_string(
            entities_list,
            threshold=float(self.settings.get("entity_precluster_similarity", 0.93)),
            limit=int(self.settings.get("entity_precluster_limit", 500))
        )
        
        # ⚠️ 重要修改：只处理预聚类的组，不再对所有实体做批处理
        # 这避免了“民众党”与“布朗大学”这种不相关实体被送给LLM判断
        if precluster_entities:
            tools.log(f"[知识图谱] 预聚类发现 {len(precluster_entities)} 组高度相似实体，开始LLM二次验证")
            
            # 将预聚类组转换为批次格式
            entity_batches = [(idx, group) for idx, group in enumerate(precluster_entities)]
            
            if entity_batches:
                entity_results = self._process_entity_batches(entity_batches, limiter)
                all_duplicate_entities.extend(entity_results)
        else:
            tools.log(f"[知识图谱] 预聚类未发现高度相似的实体组，跳过LLM处理")

        return all_duplicate_entities

    def _process_entity_batches(self, entity_batches: List[Tuple[int, List[str]]],
                               limiter: Optional[RateLimiter]) -> List[List[str]]:
        """处理实体批次"""
        self._ensure_settings_loaded()
        entity_workers = int(self.settings.get("entity_max_workers", 3))
        all_results = []
        new_rules_count = 0

        def _run_entity_batch(idx: int, batch: List[str]) -> List[List[str]]:
            tools.log(f"[知识图谱] 处理实体批次 {idx+1}/{len(entity_batches)} (大小: {len(batch)})")
            prompt = self._prepare_entity_compression_prompt_strict(batch)
            response = self._call_llm_limited(prompt, timeout=90, limiter=limiter)
            return self._parse_entity_response(response) if response else []

        # 使用AsyncExecutor统一管理线程并发
        async_executor = AsyncExecutor()
        entity_task_results = async_executor.run_threaded_tasks(
            # 直接传入 (idx, batch) 避免通过 list.index 反查导致 ValueError
            tasks=entity_batches,
            func=lambda it: _run_entity_batch(it[0], it[1]),
            max_workers=entity_workers
        )

        for batch_dupes in entity_task_results:
            if batch_dupes:
                for group in batch_dupes:
                    if len(group) >= 2 and self._valid_entity_group(group):
                        primary, dupes = self._choose_primary_entity(group)
                        for duplicate in dupes:
                            if duplicate not in self.merge_rules:
                                self.merge_rules[duplicate] = primary
                                new_rules_count += 1
                        all_results.append([primary] + dupes)

        if new_rules_count > 0:
            self._save_merge_rules()

        return all_results

    def _compress_events_with_llm(self, limiter: Optional[RateLimiter]) -> List[List[str]]:
        """使用LLM压缩事件"""
        all_duplicate_events = []

        self._ensure_settings_loaded()

        events_list = sorted(list(self.graph['events'].keys()))
        if not events_list:
            return all_duplicate_events

        # 事件分桶
        bucket_days = int(self.settings.get("event_bucket_days", 7))
        bucket_overlap = int(self.settings.get("event_bucket_entity_overlap", 1))
        bucket_max_size = int(self.settings.get("event_bucket_max_size", 40))
        buckets = self._bucket_events_by_time_and_entity(bucket_days, bucket_overlap, bucket_max_size)

        if buckets:
            event_results = self._process_event_buckets(buckets, limiter)
            all_duplicate_events.extend(event_results)

        return all_duplicate_events

    def _process_event_buckets(self, buckets: List[Dict[str, Any]],
                              limiter: Optional[RateLimiter]) -> List[List[str]]:
        """处理事件桶"""
        self._ensure_settings_loaded()
        event_workers = int(self.settings.get("event_max_workers", 3))
        BATCH_SIZE_EVT = int(self.settings.get("event_batch_size", 15))
        evt_similarity = float(self.settings.get("event_precluster_similarity", 0.82))
        evt_limit = int(self.settings.get("event_precluster_limit", 300))
        max_summary_chars = int(self.settings.get("max_summary_chars", 360))

        def _run_event_bucket(idx: int, bucket: Dict[str, Any]) -> List[List[str]]:
            bucket_keys = bucket.get("keys", [])
            bucket_events = {k: self.graph['events'][k] for k in bucket_keys if k in self.graph['events']}
            local_dupes: List[List[str]] = []

            # 预聚类
            pre_clusters = self._precluster_events_by_string(
                bucket_events, bucket_keys, evt_similarity, evt_limit, max_summary_chars
            )
            if pre_clusters:
                local_dupes.extend(pre_clusters)

            if len(bucket_keys) <= 1:
                # tools.log(f"[知识图谱] 跳过事件桶 {idx+1}/{len(buckets)}（仅1条，无需去重）")
                return local_dupes

            # LLM批处理
            total_batches = (len(bucket_keys) - 1) // BATCH_SIZE_EVT + 1
            for i in range(0, len(bucket_keys), BATCH_SIZE_EVT):
                batch_keys = bucket_keys[i:i+BATCH_SIZE_EVT]
                batch_events = {
                    k: {
                        **bucket_events.get(k, {}),
                        "event_summary": self._trim_text(
                            bucket_events.get(k, {}).get("event_summary", "") or "",
                            max_summary_chars
                        )
                    }
                    for k in batch_keys
                }
                tools.log(
                    f"[知识图谱] 处理事件桶 {idx+1}/{len(buckets)} 的批次 {i//BATCH_SIZE_EVT + 1}/{total_batches} (大小: {len(batch_keys)})"
                )
                prompt = self._prepare_event_compression_prompt(batch_events)
                response = self._call_llm_limited(prompt, timeout=120, limiter=limiter)
                if response:
                    batch_dupes = self._parse_event_response(response)
                    local_dupes.extend(batch_dupes)
            return local_dupes

        # 使用AsyncExecutor统一管理线程并发
        async_executor = AsyncExecutor()
        event_task_results = async_executor.run_threaded_tasks(
            tasks=[bucket for bucket in buckets],
            func=lambda bucket: _run_event_bucket(buckets.index(bucket), bucket),
            max_workers=event_workers
        )

        all_results = []
        for batch_dupes in event_task_results:
            if batch_dupes:
                all_results.extend(batch_dupes)

        return all_results

    def _call_llm(self, prompt: str, timeout: int) -> Optional[str]:
        """统一LLM调用 - 使用工具函数"""
        return call_llm_with_retry(
            llm_pool=self.llm_pool,
            prompt=prompt,
            max_tokens=4000,
            timeout=timeout,
            retries=2
        )

    def _call_llm_limited(self, prompt: str, timeout: int, limiter: Optional[RateLimiter]) -> Optional[str]:
        """带全局QPS限制的LLM调用"""
        return call_llm_with_retry(
            llm_pool=self.llm_pool,
            prompt=prompt,
            max_tokens=4000,
            timeout=timeout,
            retries=2,
            limiter=limiter
        )

    def _choose_primary_entity(self, group: List[str]) -> (str, List[str]):
        """
        选择主实体：优先中文，其次 first_seen 最早，再其次名称长度。
        返回 (primary, duplicates)
        """
        if not group:
            return "", []
        best = None
        best_key = None
        for name in group:
            info = self.graph['entities'].get(name, {})
            is_cn = self._is_chinese(name)
            ts = self._parse_time(info.get('first_seen', ''))
            ts_key = ts if ts > 0 else float('inf')
            key = (0 if is_cn else 1, ts_key, len(name))
            if best_key is None or key < best_key:
                best = name
                best_key = key
        duplicates = [n for n in group if n != best]
        return best, duplicates

    def _prepare_entity_compression_prompt_strict(self, entities_batch: List[str]) -> str:
        evidence_map = self._collect_entity_evidence(entities_batch)
        evidence_lines = []
        for ent, evs in evidence_map.items():
            if evs:
                for ev in evs:
                    evidence_lines.append(f"{ent} <= {ev}")

        prompt = f"""你是一名知识图谱专家。任务：仅在有充分证据时认定实体为同一主体（别名/缩写/中英文/法定全称差异）。不要因为行业相似、上下级关系或地域相似而合并。

【高风险误判示例】
- 实体具有特化职能或功效或用途，不可合并
- 行使职能的组织、机构与其下辖的更具体职能的组织、机构不可合并
- "大学" 与 "联盟/协会/部门/央行" 不是同一主体
- 不同国家/地区的同名机构，不可合并
- 上市公司 vs 子公司/控股股东，不可合并
- 政府部门 vs 上级政府，不可合并
- 国家/省州/城市/区县 之间不得互并，也不得跨国合并
- 公司/机构 ≠ 产品/品牌/型号，不得互并
- 人名 ≠ 公司/机构/地理/产品，不得互并
- 媒体名称 ≠ 地点/政府/企业/人名
- 体育俱乐部/赛事 ≠ 城市/国家/政府/个人

【实体列表】
{json.dumps(entities_batch, ensure_ascii=False, indent=2)}

【证据（部分相关事件摘要，供参考，避免幻觉）】
格式: 实体 <= 摘要 | 参与实体 | 描述
{chr(10).join(evidence_lines) if evidence_lines else "（无可用事件，谨慎合并）"}

【要求】
- 主实体优先更通用（或更倾向中国人表达）、更详细、更精确、更学术（"更XX"按优先级顺序）
- 只输出确定为同一主体的组合；不确定就返回空。
- 优先严格匹配：同名、明显译名、缩写展开。
- 不要改写名称格式（不要添加书名号/引号/括号等标点）。
- 地理/机构类合并需同一国家/行政层级；人名不得与非人名合并；公司不得与产品/品牌合并。
- 如果没有重复，返回空列表。

【输出格式】
严格返回 JSON：
{{
  "duplicate_entities": [
    ["主实体", "别名或重复"],
    ["主实体2", "别名2", "别名3"]
  ]
}}
如果没有重复，返回 {{ "duplicate_entities": [] }}。只输出JSON。
"""
        return prompt

    def _prepare_event_compression_prompt(self, events_batch: Dict) -> str:
        prompt = f"""你是一名知识图谱专家。任务：仅在描述"同一具体事实"时才视为重复事件。不要合并不同主体、上下游关联或时间不同的相似事件。

【拒绝合并的情况示例】
- 行业相似但主体不同的事件
- 上游/下游/监管/联盟关系 ≠ 同一事件
- 时间间隔明显不同的多次事件
- 事件具有连续发生或一前一后的关系
- 对于相同实体，不同时间点发生的事件

【事件列表】
格式: 摘要 | 参与实体 | 事件描述
"""
        for abstract, event in events_batch.items():
            entities = event.get('entities', [])
            summary = event.get('event_summary', '')
            prompt += f"{abstract} | {', '.join(entities)} | {summary}\n"

        prompt += """
【任务】
找出语义上高度重叠、描述同一事实的事件。

【输出格式】
严格返回 JSON：
{
  "duplicate_events": [
    ["事件摘要1", "事件摘要2"],
    ["事件摘要3", "事件摘要4", "事件摘要5"]
  ]
}
如果没有重复，返回 { "duplicate_events": [] }。只输出JSON。
"""
        return prompt

    def _parse_entity_response(self, raw_content: str) -> List[List[str]]:
        try:
            data = self._extract_json(raw_content)
            res = data.get("duplicate_entities", [])
            return res if isinstance(res, list) else []
        except Exception:
            return []

    def _parse_event_response(self, raw_content: str) -> List[List[str]]:
        try:
            data = self._extract_json(raw_content)
            res = data.get("duplicate_events", [])
            return res if isinstance(res, list) else []
        except Exception:
            return []

    def _extract_json(self, text: str) -> Dict:
        """使用统一的JSON提取函数"""
        return extract_json_from_llm_response(text)

    def update_entities_and_events(self, duplicates: Dict[str, List[List[str]]]):
        """根据重复检测结果更新实体库和事件库"""
        updated = False

        # 合并重复实体
        for group in duplicates.get("duplicate_entities", []):
            if len(group) < 2:
                continue
            if not self._valid_entity_group(group):
                continue
            primary, dupes = self._choose_primary_entity(group)
            # 确保主实体存在；若不存在但有重复实体存在，可交换
            if primary not in self.graph['entities'] and dupes:
                for d in dupes:
                    if d in self.graph['entities']:
                        primary, dupes = d, [x for x in group if x != d]
                        break
            for duplicate in dupes:
                if duplicate in self.graph['entities'] and primary in self.graph['entities']:
                    self._merge_entities(primary, duplicate)
                    updated = True

        # 合并重复事件
        for group in duplicates.get("duplicate_events", []):
            if len(group) < 2:
                continue
            primary = group[0]
            for duplicate in group[1:]:
                if duplicate in self.graph['events']:
                    self._merge_events(primary, duplicate)
                    updated = True

        if updated:
            self._save_data()
            tools.log("[知识图谱] 实体和事件更新完成")
        else:
            tools.log("[知识图谱] 无重复项需要更新")

    def _merge_entities(self, primary: str, duplicate: str):
        """合并重复实体"""
        if primary not in self.graph['entities'] or duplicate not in self.graph['entities']:
            return

        primary_data = self.graph['entities'][primary]
        duplicate_data = self.graph['entities'][duplicate]

        # 合并sources (确保转换为可哈希的tuple或直接列表处理)
        primary_sources = set()
        for s in primary_data.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue # 暂时忽略复杂结构
            else: primary_sources.add(s)

        for s in duplicate_data.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)

        # 转回list
        primary_data['sources'] = list(primary_sources)

        # 合并original_forms
        primary_forms = set()
        for f in primary_data.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)

        for f in duplicate_data.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)

        # 将重复实体名也作为主实体的其他表述记录，防止丢失别名
        primary_forms.add(duplicate)
        primary_forms.add(primary)

        primary_data['original_forms'] = list(primary_forms)

        # 更新first_seen为更早的时间
        primary_first = primary_data.get('first_seen', '')
        duplicate_first = duplicate_data.get('first_seen', '')
        if duplicate_first and (not primary_first or duplicate_first < primary_first):
            primary_data['first_seen'] = duplicate_first

        # 删除重复实体
        del self.graph['entities'][duplicate]

        # 更新事件中的实体引用
        for abstract, event in self.graph['events'].items():
            entities = event.get('entities', [])
            if duplicate in entities:
                # 替换为primary，并去重
                new_entities = [primary if ent == duplicate else ent for ent in entities]
                # 去重
                unique_entities = []
                seen = set()
                for ent in new_entities:
                    if ent not in seen:
                        seen.add(ent)
                        unique_entities.append(ent)
                event['entities'] = unique_entities

            # 同步合并 entity_roles（不要求 duplicate 在 entities 中，防止历史脏数据）
            roles_map = event.get("entity_roles", {})
            if isinstance(roles_map, dict) and duplicate in roles_map:
                dup_roles = roles_map.get(duplicate, [])
                prim_roles = roles_map.get(primary, [])
                merged = []
                seen_r = set()
                for r in (prim_roles if isinstance(prim_roles, list) else []):
                    if isinstance(r, str) and r and r not in seen_r:
                        seen_r.add(r)
                        merged.append(r)
                for r in (dup_roles if isinstance(dup_roles, list) else []):
                    if isinstance(r, str) and r and r not in seen_r:
                        seen_r.add(r)
                        merged.append(r)
                if merged:
                    roles_map[primary] = merged
                roles_map.pop(duplicate, None)
                event["entity_roles"] = roles_map

            # 同步更新 relations 主客体
            rels = event.get("relations", [])
            if isinstance(rels, list):
                changed = False
                new_rels = []
                for rel in rels:
                    if not isinstance(rel, dict):
                        continue
                    s = rel.get("subject", "")
                    o = rel.get("object", "")
                    if isinstance(s, str) and s.strip() == duplicate:
                        rel["subject"] = primary
                        changed = True
                    if isinstance(o, str) and o.strip() == duplicate:
                        rel["object"] = primary
                        changed = True
                    new_rels.append(rel)
                if changed:
                    event["relations"] = new_rels

        tools.log(f"[知识图谱] 合并实体: {duplicate} -> {primary}")

    def _merge_events(self, primary: str, duplicate: str):
        """合并重复事件"""
        if primary not in self.graph['events'] or duplicate not in self.graph['events']:
            return

        primary_event = self.graph['events'][primary]
        duplicate_event = self.graph['events'][duplicate]

        # 合并sources
        primary_sources = set()
        for s in primary_event.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)

        for s in duplicate_event.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)

        primary_event['sources'] = list(primary_sources)

        # 合并entities
        primary_entities = set(primary_event.get('entities', []))
        duplicate_entities = set(duplicate_event.get('entities', []))
        primary_event['entities'] = list(primary_entities.union(duplicate_entities))

        # 合并 event_types（去重）
        def _merge_unique_list(a: Any, b: Any) -> List[str]:
            out: List[str] = []
            seen = set()
            for it in (a if isinstance(a, list) else []):
                if isinstance(it, str) and it and it not in seen:
                    seen.add(it)
                    out.append(it)
            for it in (b if isinstance(b, list) else []):
                if isinstance(it, str) and it and it not in seen:
                    seen.add(it)
                    out.append(it)
            return out

        primary_event["event_types"] = _merge_unique_list(primary_event.get("event_types", []), duplicate_event.get("event_types", []))

        # 合并 entity_roles（同实体角色去重）
        def _merge_roles(a: Any, b: Any) -> Dict[str, List[str]]:
            out: Dict[str, List[str]] = {}
            if isinstance(a, dict):
                for k, v in a.items():
                    if isinstance(k, str) and isinstance(v, list):
                        out[k] = [x for x in v if isinstance(x, str) and x]
            if isinstance(b, dict):
                for k, v in b.items():
                    if not isinstance(k, str):
                        continue
                    roles_list: List[str] = []
                    if isinstance(v, str):
                        roles_list = [v]
                    elif isinstance(v, list):
                        roles_list = [x for x in v if isinstance(x, str)]
                    roles_list = [x.strip() for x in roles_list if isinstance(x, str) and x.strip()]
                    if not roles_list:
                        continue
                    out[k] = _merge_unique_list(out.get(k, []), roles_list)
            return out

        primary_event["entity_roles"] = _merge_roles(primary_event.get("entity_roles", {}), duplicate_event.get("entity_roles", {}))

        # 合并 relations（三元组去重、evidence 去重）
        def _merge_relations(a: Any, b: Any) -> List[Dict[str, Any]]:
            out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            def ns(x: Any) -> str:
                return x.strip() if isinstance(x, str) else ""
            def add_one(rel: Any):
                if not isinstance(rel, dict):
                    return
                s = ns(rel.get("subject", ""))
                p = ns(rel.get("predicate", ""))
                o = ns(rel.get("object", ""))
                if not s or not p or not o:
                    return
                key = (s, p, o)
                if key not in out:
                    out[key] = {"subject": s, "predicate": p, "object": o, "evidence": []}
                ev = rel.get("evidence", [])
                ev_list: List[str] = []
                if isinstance(ev, str):
                    ev_list = [ev]
                elif isinstance(ev, list):
                    ev_list = [x for x in ev if isinstance(x, str)]
                ev_list = [x.strip() for x in ev_list if isinstance(x, str) and x.strip()]
                if ev_list:
                    cur = out[key].get("evidence", [])
                    if not isinstance(cur, list):
                        cur = []
                    for x in ev_list:
                        if x not in cur:
                            cur.append(x)
                    out[key]["evidence"] = cur
            for rel in (a if isinstance(a, list) else []):
                add_one(rel)
            for rel in (b if isinstance(b, list) else []):
                add_one(rel)
            return list(out.values())

        primary_event["relations"] = _merge_relations(primary_event.get("relations", []), duplicate_event.get("relations", []))

        # 合并 event_start_time：优先更高精度，其次更早
        def _choose_better_time(cur_t: Any, cur_txt: Any, cur_p: Any, inc_t: Any, inc_txt: Any, inc_p: Any) -> Tuple[str, str, str]:
            def ns(x: Any) -> str:
                return x.strip() if isinstance(x, str) else ""
            ct, ctxt, cp = ns(cur_t), ns(cur_txt), (ns(cur_p) or "unknown")
            it, itxt, ip = ns(inc_t), ns(inc_txt), (ns(inc_p) or "unknown")
            if not ct and it:
                return it, itxt, ip
            if ct and not it:
                return ct, ctxt, cp
            if not ct and not it:
                return "", "", cp
            rank = {"exact": 3, "date": 2, "relative": 1, "unknown": 0}
            rc, ri = rank.get(cp, 0), rank.get(ip, 0)
            if ri > rc:
                return it, itxt, ip
            if ri < rc:
                return ct, ctxt, cp
            if len(ct) >= 10 and ct[4:5] == "-" and ct[7:8] == "-" and len(it) >= 10 and it[4:5] == "-" and it[7:8] == "-":
                if it < ct:
                    return it, itxt, ip
            return ct, ctxt, cp

        nt, ntxt, np = _choose_better_time(
            primary_event.get("event_start_time", ""),
            primary_event.get("event_start_time_text", ""),
            primary_event.get("event_start_time_precision", "unknown"),
            duplicate_event.get("event_start_time", ""),
            duplicate_event.get("event_start_time_text", ""),
            duplicate_event.get("event_start_time_precision", "unknown"),
        )
        primary_event["event_start_time"] = nt
        primary_event["event_start_time_text"] = ntxt
        primary_event["event_start_time_precision"] = np

        # 更新first_seen
        primary_first = primary_event.get('first_seen', '')
        duplicate_first = duplicate_event.get('first_seen', '')
        if duplicate_first and (not primary_first or duplicate_first < primary_first):
            primary_event['first_seen'] = duplicate_first

        # 事件描述合并：保留更详细的
        if not primary_event.get('event_summary') and duplicate_event.get('event_summary'):
            primary_event['event_summary'] = duplicate_event['event_summary']

        # 删除重复事件
        del self.graph['events'][duplicate]

        tools.log(f"[知识图谱] 合并事件: {duplicate} -> {primary}")

    def _save_data(self):
        """保存更新后的数据到文件"""
        try:
            # 保存实体
            write_json_file(self.entities_file, self.graph['entities'], ensure_ascii=False, indent=2)

            # 保存事件（abstract_map格式）
            abstract_map = {}
            for abstract, event in self.graph['events'].items():
                abstract_map[abstract] = {
                    "event_id": event.get("event_id", ""),
                    "entities": event.get('entities', []),
                    "event_summary": event.get('event_summary', ''),
                    "event_types": event.get("event_types", []),
                    "entity_roles": event.get("entity_roles", {}),
                    "relations": event.get("relations", []),
                    "event_start_time": event.get("event_start_time", ""),
                    "event_start_time_text": event.get("event_start_time_text", ""),
                    "event_start_time_precision": event.get("event_start_time_precision", "unknown"),
                    "reported_at": event.get("reported_at", ""),
                    "sources": event.get('sources', []),
                    "first_seen": event.get('first_seen', '')
                }

            write_json_file(self.abstract_map_file, abstract_map, ensure_ascii=False, indent=2)

            # 保存知识图谱状态（可选）
            write_json_file(self.kg_file, self.graph, ensure_ascii=False, indent=2)

            tools.log("[知识图谱] 数据保存完成")
        except Exception as e:
            tools.log(f"[知识图谱] ❌ 保存数据失败: {e}")

    def append_only_update(self, events_list: List[Dict[str, Any]], default_source: str = "auto_pipeline", allow_append_original_forms: bool = True) -> Dict[str, int]:
        """
        只追加新数据，不改动已有实体/事件的旧字段。
        - 实体已存在：不改 first_seen/sources，默认仅可选地追加原始表述
        - 事件已存在（同 abstract）：跳过，不改旧事件
        """
        if not events_list:
            return {"added_entities": 0, "added_events": 0}

        if not self.build_graph():
            return {"added_entities": 0, "added_events": 0}

        # 准备 LLM 去重映射（仅映射"新实体"到"已有实体"，不改旧实体字段）
        self._init_llm_pool()
        merge_rules = self.merge_rules or {}
        existing_entities = set(self.graph["entities"].keys())

        # 收集新实体集合
        new_entities_set = set()
        for ev in events_list:
            ents = ev.get("entities", []) or []
            ents = [merge_rules.get(e, e) for e in ents if e]
            for e in ents:
                if e not in existing_entities:
                    new_entities_set.add(e)

        # 构建映射：新实体 -> 已有实体（LLM 判断可能同名）
        llm_merge_map: Dict[str, str] = {}
        if self.llm_pool and new_entities_set:
            # 分桶降低上下文长度：按首字母/字符分桶
            from collections import defaultdict
            # from concurrent.futures import ThreadPoolExecutor, as_completed  # 已迁移到AsyncExecutor
            buckets = defaultdict(list)
            for ent in new_entities_set:
                prefix = ent[0] if ent else "#"
                buckets[prefix].append(ent)

            llm_lock = threading.Lock()
            max_workers = int(self.settings.get("entity_max_workers", 3)) or 1
            async_executor = AsyncExecutor()

            def handle_bucket(prefix: str, bucket_new: List[str]):
                local_map = {}
                # 取同前缀的部分已有实体做对比，避免过长
                existing_subset = [e for e in existing_entities if e.startswith(prefix)]
                existing_subset = existing_subset[: max(10, min(80, len(existing_subset)))]
                candidates = list(existing_subset) + list(bucket_new)
                if len(candidates) < 2:
                    return local_map
                try:
                    prompt = self._prepare_entity_compression_prompt_strict(candidates)
                    resp = self._call_llm_limited(prompt, timeout=60, limiter=None)
                    groups = self._parse_entity_response(resp) if resp else []
                    for g in groups:
                        if len(g) < 2:
                            continue
                        primary, dupes = self._choose_primary_entity(g)
                        if primary in existing_entities:
                            for d in dupes:
                                if d in new_entities_set:
                                    local_map[d] = primary
                except Exception as e:
                    tools.log(f"[知识图谱] 追加模式 LLM 去重失败: {e}")
                return local_map

            # 使用AsyncExecutor统一管理线程并发
            bucket_results = async_executor.run_threaded_tasks(
                tasks=[(p, b) for p, b in buckets.items()],
                func=lambda pb: handle_bucket(pb[0], pb[1]),
                max_workers=max_workers
            )

            for res_map in bucket_results:
                if res_map:
                    with llm_lock:
                        llm_merge_map.update(res_map)

        added_entities = 0
        added_events = 0

        def normalize_entity(name: str) -> str:
            if not name:
                return name
            name = merge_rules.get(name, name)
            return llm_merge_map.get(name, name)

        for ev in events_list:
            ents = ev.get("entities", []) or []
            ents = [normalize_entity(e) for e in ents]
            ents_original = ev.get("entities_original") or ents
            src = ev.get("source", default_source)
            published_at = ev.get("published_at") or ev.get("datetime") or ""

            # 追加实体（仅当不存在）
            for ent, ent_orig in zip(ents, ents_original):
                if not ent:
                    continue
                if ent not in self.graph["entities"]:
                    self.graph["entities"][ent] = {
                        "first_seen": published_at or datetime.utcnow().isoformat(),
                        "sources": [src] if src else [],
                        "original_forms": [ent_orig] if ent_orig else []
                    }
                    added_entities += 1
                else:
                    # 不改旧字段，仅可选追加原始表述
                    if allow_append_original_forms and ent_orig:
                        forms = self.graph["entities"][ent].get("original_forms", [])
                        if ent_orig not in forms:
                            forms.append(ent_orig)
                            self.graph["entities"][ent]["original_forms"] = forms

            # 追加事件（仅当摘要不存在）
            abstract = ev.get("abstract")
            if abstract and abstract not in self.graph["events"]:
                self.graph["events"][abstract] = {
                    "abstract": abstract,
                    "entities": ents,
                    "event_summary": ev.get("event_summary", ""),
                    "event_types": ev.get("event_types", []),
                    "entity_roles": ev.get("entity_roles", {}),
                    "relations": ev.get("relations", []),
                    "event_start_time": ev.get("event_start_time", ""),
                    "event_start_time_text": ev.get("event_start_time_text", ""),
                    "event_start_time_precision": ev.get("event_start_time_precision", "unknown"),
                    "sources": [src] if src else [],
                    "first_seen": published_at
                }
                added_events += 1

        # 重建边并保存（只在有新增时）
        if added_entities or added_events:
            self._build_edges()
            self._save_data()
            tools.log(f"[知识图谱] 追加模式完成：新增实体 {added_entities}，新增事件 {added_events}")
        else:
            tools.log("[知识图谱] 追加模式：没有新增实体/事件")

        return {"added_entities": added_entities, "added_events": added_events}

    def refresh_graph(self):
        """刷新知识图谱：构建、压缩、更新"""
        tools.log("[知识图谱] 开始刷新知识图谱")

        # 构建图
        if not self.build_graph():
            tools.log("[知识图谱] ❌ 构建图失败")
            return

        # 压缩：使用LLM检测重复
        duplicates = self.compress_with_llm()

        # 更新实体和事件
        self.update_entities_and_events(duplicates)

        tools.log("[知识图谱] 知识图谱刷新完成")
        # 清理已加载的临时文件
        self._cleanup_tmp_files()

