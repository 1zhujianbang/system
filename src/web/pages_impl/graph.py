"""
知识图谱页面 - 优雅版

核心功能：PyVis 交互式图谱展示 + 实体聚焦
采用面向对象设计，提高代码可维护性和扩展性
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
from abc import ABC, abstractmethod

import streamlit as st
import streamlit.components.v1 as components
from collections import defaultdict

from src.web import utils
from src.web.services.run_store import cache_dir
from src.web.framework.user_context import get_user_context, render_user_context_controls
from src.interfaces.web.snapshot_protocol import (
    SnapshotLoader,
    SnapshotTransformer,
    GRAPH_TYPE_LABELS,
    validate_snapshot_dict,
)


class GraphRenderer(ABC):
    """图谱渲染器抽象基类"""
    
    def __init__(self):
        self.project_id = get_user_context().project_id
        self.cache_path = cache_dir(self.project_id) / "pyvis"
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def render(self) -> None:
        """渲染图谱的主入口"""
        pass
    
    def _load_entities(self) -> Dict[str, Any]:
        """加载实体数据"""
        return utils.load_entities() or {}
    
    def _load_events(self) -> Dict[str, Any]:
        """加载事件数据"""
        return utils.load_events() or {}
    
    def _get_kg_store(self):
        """获取知识图谱存储实例"""
        from src.adapters.sqlite.kg_read_store import SQLiteKGReadStore
        return SQLiteKGReadStore()
    
    def _format_timestamp(self, timestamp: str | None) -> str:
        """格式化时间戳为可读字符串"""
        if not timestamp or timestamp == "Unknown":
            return "Unknown"
        
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return timestamp[:10] if len(timestamp) >= 10 else "Unknown"
    
    def _extract_timestamps(self, events: List[Dict[str, Any]]) -> List[datetime]:
        """从事件列表中提取有效的时间戳"""
        timestamps = []
        for evt in events:
            ts = evt.get("event_start_time") or evt.get("reported_at")
            if ts:
                try:
                    # 统一时区处理，避免naive和aware datetime比较错误
                    if ts.endswith('Z'):
                        dt = datetime.fromisoformat(ts[:-1] + "+00:00")
                    else:
                        # 确保所有时间戳都有时区信息
                        dt = datetime.fromisoformat(ts)
                        # 如果是naive datetime，转换为UTC时区
                        if dt.tzinfo is None:
                            from datetime import timezone
                            dt = dt.replace(tzinfo=timezone.utc)
                    timestamps.append(dt)
                except Exception:
                    pass
        # 确保所有时间戳都是同一时区类型后再排序
        if timestamps:
            # 检查是否有时区信息
            has_timezone = any(ts.tzinfo is not None for ts in timestamps)
            if has_timezone:
                # 确保所有时间戳都有时区信息
                from datetime import timezone
                normalized_timestamps = []
                for ts in timestamps:
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    normalized_timestamps.append(ts)
                return normalized_timestamps
        return timestamps
    
    def _render_entity_event_list(
        self,
        entities: Dict[str, Any],
        events: Dict[str, Any],
        max_display: int = 100
    ) -> None:
        """渲染实体和事件列表（双列布局）"""
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander(f"🧠 实体列表 ({len(entities)})", expanded=False):
                for name in sorted(entities.keys())[:max_display]:
                    info = entities.get(name, {})
                    count = info.get("count", 1) if isinstance(info, dict) else 1
                    st.write(f"• **{name}** ({count})")
                if len(entities) > max_display:
                    st.caption(f"... 还有 {len(entities) - max_display} 个")
        
        with col2:
            with st.expander(f"🔗 事件列表 ({len(events)})", expanded=False):
                for abstract in sorted(events.keys())[:max_display]:
                    info = events.get(abstract, {})
                    summary = info.get("event_summary", abstract) if isinstance(info, dict) else abstract
                    st.write(f"• {summary[:80]}...")
                if len(events) > max_display:
                    st.caption(f"... 还有 {len(events) - max_display} 个")
    
    def _render_pyvis(
        self,
        nodes: List[Tuple[str, Dict]],
        edges: List[Tuple[str, str, Dict]],
        layout_config: Dict = None,
        directed: bool = False
    ) -> None:
        """通用 PyVis 渲染方法"""
        try:
            from pyvis.network import Network
            
            # 创建网络
            net = Network(
                height="720px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#333333",
                directed=directed
            )
            
            # 应用布局配置
            if layout_config:
                net.set_options(json.dumps(layout_config))
            
            # 添加节点
            for node_id, node_attrs in nodes:
                net.add_node(node_id, **node_attrs)
            
            # 添加边
            for u, v, edge_attrs in edges:
                net.add_edge(u, v, **edge_attrs)
            
            # 生成 HTML
            content_for_hash = json.dumps({
                "nodes": sorted([nid for nid, _ in nodes]),
                "edges": sorted([(u, v) for u, v, _ in edges])
            })
            graph_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:16]
            html_path = self.cache_path / f"graph_{graph_hash}.html"
            
            net.save_graph(str(html_path))
            
            # 读取并显示
            html_content = html_path.read_text(encoding="utf-8")
            components.html(html_content, height=720, scrolling=True)
            
        except ImportError:
            st.error("PyVis 未安装。请运行: pip install pyvis")
        except Exception as e:
            st.error(f"图谱渲染失败: {e}")


class SnapshotGraphRenderer(GraphRenderer):
    def _build_pyvis_payload(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        *,
        focus_node: str = "",
        max_nodes: int = 800,
        max_edges: int = 2500,
        min_degree: int = 0,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, Dict[str, Any]]]]:
        deg: Dict[str, int] = defaultdict(int)
        for e in edges:
            u = str(e.get("from", "")).strip()
            v = str(e.get("to", "")).strip()
            if not u or not v:
                continue
            deg[u] += 1
            deg[v] += 1

        if focus_node:
            target_nodes: Set[str] = {focus_node}
            adj: Dict[str, Set[str]] = defaultdict(set)
            for e in edges:
                u = str(e.get("from", "")).strip()
                v = str(e.get("to", "")).strip()
                if not u or not v:
                    continue
                adj[u].add(v)
                adj[v].add(u)
            frontier = {focus_node}
            for _ in range(2):
                nxt = set()
                for x in frontier:
                    nxt |= adj.get(x, set())
                nxt -= target_nodes
                target_nodes |= nxt
                frontier = nxt
        else:
            candidates = [nid for nid, d in deg.items() if d >= int(min_degree)]
            candidates_sorted = sorted(candidates, key=lambda x: deg.get(x, 0), reverse=True)
            target_nodes = set(candidates_sorted[: int(max_nodes) if int(max_nodes) > 0 else 800])

        filtered_edges = []
        for e in edges:
            u = str(e.get("from", "")).strip()
            v = str(e.get("to", "")).strip()
            if u in target_nodes and v in target_nodes:
                filtered_edges.append(e)
        filtered_edges = filtered_edges[: int(max_edges) if int(max_edges) > 0 else 2500]

        nodes_by_id = {str(n.get("id")): n for n in nodes if isinstance(n, dict) and str(n.get("id", "")).strip()}
        used_nodes: Set[str] = set()
        for e in filtered_edges:
            used_nodes.add(str(e.get("from", "")).strip())
            used_nodes.add(str(e.get("to", "")).strip())
        nodes2 = [nodes_by_id[nid] for nid in used_nodes if nid in nodes_by_id]

        pyvis_nodes: List[Tuple[str, Dict[str, Any]]] = []
        for n in nodes2:
            nid = str(n.get("id"))
            ntype = str(n.get("type") or "entity").strip() or "entity"
            label = str(n.get("label") or nid)
            color = str(n.get("color") or "#1f77b4")
            is_focus = bool(focus_node) and nid == focus_node
            size = 22
            if ntype == "event":
                size = 18
            if is_focus:
                size = 30
            d = deg.get(nid, 0)
            size = min(size + int(d / 3), 40)
            shape = "dot"
            if ntype == "relation_state":
                shape = "box"
            title = json.dumps(n, ensure_ascii=False, indent=2)[:4000]
            pyvis_nodes.append(
                (
                    nid,
                    {
                        "label": label[:80],
                        "color": "#e74c3c" if is_focus else color,
                        "shape": shape,
                        "size": size,
                        "title": title,
                        "borderWidth": 2,
                        "borderWidthSelected": 3,
                        "font": {"size": 12, "color": "#333333", "bold": is_focus},
                    },
                )
            )

        pyvis_edges: List[Tuple[str, str, Dict[str, Any]]] = []
        for e in filtered_edges:
            u = str(e.get("from", "")).strip()
            v = str(e.get("to", "")).strip()
            etype = str(e.get("type") or "").strip().lower()
            title = str(e.get("title") or "")
            t = str(e.get("time") or "")
            edge_title = f"{etype} | {title} | {t}".strip(" |")[:400]
            arrows = {"to": {"enabled": etype in {"before", "evolved_to", "evolve", "causes"}}}
            color = "#95a5a6"
            if etype in {"before"}:
                color = "#3498db"
            if etype in {"evolved_to", "evolve"}:
                color = "#9b59b6"
            pyvis_edges.append(
                (
                    u,
                    v,
                    {
                        "title": edge_title,
                        "width": 2,
                        "color": {"color": color, "highlight": "#2ecc71", "hover": "#2ecc71", "opacity": 0.6},
                        "smooth": {"enabled": True, "type": "dynamic", "roundness": 0.4},
                        "arrows": arrows,
                        "length": 150,
                    },
                )
            )

        return pyvis_nodes, pyvis_edges

    def render(self) -> None:
        st.subheader("📦 快照视图（统一协议）")

        loader = SnapshotLoader(snapshot_dir=Path("data/snapshots"))
        available = loader.list_available_types()

        with st.sidebar:
            st.header("📦 快照控制")
            graph_type = st.selectbox(
                "快照类型",
                options=available,
                format_func=lambda x: GRAPH_TYPE_LABELS.get(str(x), str(x)),
            )
            max_nodes = st.slider("最大节点数", 200, 5000, 800, 100)
            max_edges = st.slider("最大边数", 200, 10000, 2500, 100)
            min_degree = st.slider("最小度数", 0, 20, 0, 1)
            focus_enabled = st.checkbox("聚焦模式（2跳）", value=False)
            time_hours = st.slider("时间过滤（小时，0=不过滤）", 0, 24 * 30, 0, 12)

            gen = st.button("生成/刷新五图谱快照", use_container_width=True)

        if gen:
            with st.spinner("生成快照中..."):
                try:
                    from src.app.services_impl import get_kg_service

                    res = get_kg_service().generate_snapshots()
                    if not getattr(res, "success", False):
                        st.error(f"快照生成失败: {getattr(res, 'error', '')}")
                    else:
                        st.success("快照生成完成")
                except Exception as e:
                    st.error(f"快照生成异常: {e}")

        raw = loader.load_snapshot(graph_type)
        if raw is None:
            st.warning(f"未找到快照文件: data/snapshots/{graph_type}.json（或 KG 原始文件缺失）")
            st.stop()

        if graph_type == "KG":
            snapshot = SnapshotTransformer.from_kg_json(raw)
        else:
            snapshot = raw

        nodes = SnapshotTransformer.normalize_nodes(snapshot.get("nodes", []) if isinstance(snapshot, dict) else [])
        edges = SnapshotTransformer.normalize_edges(snapshot.get("edges", []) if isinstance(snapshot, dict) else [])
        meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
        graph_type2 = str(meta.get("graph_type") or graph_type)

        snapshot2 = {"meta": meta, "nodes": nodes, "edges": edges}
        if time_hours and int(time_hours) > 0:
            snapshot2 = SnapshotTransformer.filter_by_time(snapshot2, int(time_hours))
            nodes = snapshot2.get("nodes", [])
            edges = snapshot2.get("edges", [])

        entity_candidates = sorted([str(n.get("id")) for n in nodes if str(n.get("type", "")) == "entity"])
        focus_node = ""
        if focus_enabled and entity_candidates:
            focus_node = st.sidebar.selectbox("聚焦实体", options=[""] + entity_candidates, index=0)

        report = validate_snapshot_dict({"meta": {"graph_type": graph_type2, **meta}, "nodes": nodes, "edges": edges})

        st.info(f"📈 {GRAPH_TYPE_LABELS.get(graph_type2, graph_type2)}：{report['counts']['nodes']} 节点，{report['counts']['edges']} 边")
        if not report.get("ok", False):
            st.error("快照协议校验未通过")
            if report.get("errors"):
                st.json({"errors": report.get("errors")})
            if report.get("missing_nodes"):
                st.json({"missing_nodes_sample": report.get("missing_nodes")})
            if report.get("missing_edges"):
                st.json({"missing_edges_sample": report.get("missing_edges")})

        with st.expander("字段协议清单", expanded=False):
            st.write("必填字段")
            st.json({"node": report["required"]["node"], "edge": report["required"]["edge"]})
            if report.get("recommended", {}).get("node") or report.get("recommended", {}).get("edge"):
                st.write("推荐字段（按图谱类型）")
                st.json(report.get("recommended", {}))

        if focus_node:
            snapshot2 = SnapshotTransformer.filter_by_focus({"meta": meta, "nodes": nodes, "edges": edges}, focus_entity=focus_node, max_depth=2)
            nodes = snapshot2.get("nodes", [])
            edges = snapshot2.get("edges", [])

        pyvis_nodes, pyvis_edges = self._build_pyvis_payload(
            nodes,
            edges,
            focus_node=focus_node,
            max_nodes=int(max_nodes),
            max_edges=int(max_edges),
            min_degree=int(min_degree),
        )

        if not pyvis_nodes or not pyvis_edges:
            st.info("当前筛选条件下没有可显示的图谱数据。")
            st.stop()

        self._render_pyvis(
            nodes=pyvis_nodes,
            edges=pyvis_edges,
            layout_config={
                "physics": {
                    "enabled": True,
                    "barnesHut": {
                        "gravitationalConstant": -2500,
                        "centralGravity": 0.3,
                        "springLength": 140,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0,
                    },
                },
            },
            directed=True,
        )


def render() -> None:
    """主渲染函数"""
    render_user_context_controls()
    
    # --- 视图模式选择 ---
    view_mode = st.sidebar.selectbox(
        "📊 图谱类型",
        [
            "快照视图（统一协议）",
            "实体-事件关系图谱",
            "实体时序图谱",
            "实体关系图谱",
            "动态演化图谱",
            "因果传播图谱"
        ],
        help="选择不同的知识图谱视图"
    )
    
    # 根据视图模式渲染
    renderer_map = {
        "快照视图（统一协议）": SnapshotGraphRenderer(),
        "实体-事件关系图谱": EntityEventGraphRenderer(),
        "实体时序图谱": TimelineGraphRenderer(),
        "实体关系图谱": EntityRelationGraphRenderer(),
        "动态演化图谱": EvolutionGraphRenderer(),
        "因果传播图谱": CausalGraphRenderer()
    }
    
    renderer = renderer_map.get(view_mode)
    if renderer:
        renderer.render()


class EntityEventGraphRenderer(GraphRenderer):
    """实体-事件关系图谱渲染器"""
    
    def render(self) -> None:
        """渲染实体-事件关系图谱"""
        # --- 数据加载 ---
        with st.spinner("加载图谱数据..."):
            entities = self._load_entities()
            events = self._load_events()
        
        if not entities and not events:
            st.warning("知识图谱为空。请先运行流水线抓取数据。")
            st.stop()
        
        # --- 侧边栏控制 ---
        with st.sidebar:
            st.header("🔍 图谱控制")
            
            # 实体搜索
            all_entities = sorted(entities.keys())
            focus_entity = st.selectbox(
                "聚焦实体",
                options=["(全部)"] + all_entities,
                index=0,
                help="选择一个实体查看其关联"
            )
            
            # 显示设置
            max_nodes = st.slider("最大节点数", 200, 10000, 200, 200)
            
            st.divider()
            
            # 数据统计
            st.caption(f"📊 实体: {len(entities)}")
            st.caption(f"📊 事件: {len(events)}")
        
        # --- 构建图谱 ---
        edge_list = []
        for evt_abstract, evt_data in events.items():
            if not isinstance(evt_data, dict):
                continue
            evt_id = f"EVT:{evt_abstract}"
            evt_summary = evt_data.get("event_summary", evt_abstract)
            for ent in evt_data.get("entities", []):
                if ent in entities:
                    edge_list.append((evt_id, ent, {"title": evt_summary}))
        
        # 构建邻接表
        adj = defaultdict(set)
        for u, v, _ in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # --- 节点过滤 ---
        target_nodes = set()
        
        if focus_entity != "(全部)":
            # 聚焦模式：BFS 拓展
            target_nodes.add(focus_entity)
            frontier = {focus_entity}
            for _ in range(2):  # 2 跳深度
                next_frontier = set()
                for node in frontier:
                    next_frontier |= adj.get(node, set())
                next_frontier -= target_nodes
                target_nodes |= next_frontier
                frontier = next_frontier
        else:
            # 全局模式：按度数选 Top N
            deg = defaultdict(int)
            for u, v, _ in edge_list:
                deg[u] += 1
                deg[v] += 1
            
            # 分别选实体和事件
            entity_nodes = [n for n in deg if not n.startswith("EVT:")]
            event_nodes = [n for n in deg if n.startswith("EVT:")]
            
            top_entities = sorted(entity_nodes, key=lambda x: deg[x], reverse=True)[:max_nodes // 2]
            top_events = sorted(event_nodes, key=lambda x: deg[x], reverse=True)[:max_nodes // 2]
            
            target_nodes = set(top_entities) | set(top_events)
        
        # 过滤边
        filtered_edges = [
            (u, v, d) for u, v, d in edge_list
            if u in target_nodes and v in target_nodes
        ]
        
        if not filtered_edges:
            st.info("当前筛选条件下没有可显示的图谱数据。")
            st.stop()
        
        # --- 使用 PyVis 渲染 ---
        st.info(f"📈 图谱可视化 ({len(target_nodes)} 节点, {len(filtered_edges)} 边)")
        
        # 构建节点和边
        nodes = []
        edges = []
        added_nodes = set()
        
        for u, v, d in filtered_edges:
            for node in [u, v]:
                if node in added_nodes:
                    continue
                added_nodes.add(node)
                
                if node.startswith("EVT:"):
                    # 事件节点（橙色）
                    label = node[4:][:50] + "..." if len(node) > 54 else node[4:]
                    nodes.append((node, {
                        "label": label,
                        "color": "#ff7f0e",
                        "shape": "dot",
                        "size": 20,
                        "borderWidth": 2,
                        "borderWidthSelected": 3,
                        "font": {"size": 12, "color": "#333333"}
                    }))
                else:
                    # 实体节点（蓝色/红色）
                    is_focus = (focus_entity != "(全部)" and node == focus_entity)
                    color = "#e74c3c" if is_focus else "#1f77b4"
                    size = 28 if is_focus else 22
                    nodes.append((node, {
                        "label": node,
                        "color": color,
                        "shape": "dot",
                        "size": size,
                        "borderWidth": 2,
                        "borderWidthSelected": 3,
                        "font": {"size": 14 if is_focus else 12, "color": "#333333", "bold": is_focus}
                    }))
            
            # 添加边
            title = d.get("title", "")[:100]
            edges.append((u, v, {
                "title": title,
                "width": 2,
                "color": {
                    "color": "#95a5a6",
                    "highlight": "#3498db",
                    "hover": "#2ecc71",
                    "opacity": 0.6
                },
                "smooth": {
                    "enabled": True,
                    "type": "dynamic",
                    "roundness": 0.5
                },
                "arrows": {
                    "to": {
                        "enabled": False
                    }
                },
                "length": 150
            }))
        
        # 渲染图谱
        self._render_pyvis(
            nodes=nodes,
            edges=edges,
            layout_config={
                "physics": {
                    "enabled": True,
                    "barnesHut": {
                        "gravitationalConstant": -3000,
                        "centralGravity": 0.3,
                        "springLength": 150,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0
                    }
                }
            },
            directed=False
        )
        
        st.divider()
        
        # --- 实体/事件列表 ---
        self._render_entity_event_list(entities, events)


class TimelineGraphRenderer(GraphRenderer):
    """实体时序图谱渲染器"""
    
    def render(self) -> None:
        """渲染实体时序图谱"""
        st.subheader("📅 实体时序图谱")
        st.caption("显示实体的时间轴事件链")
        
        # 加载数据
        entities = self._load_entities()
        if not entities:
            st.warning("无实体数据。请先运行流程。")
            return
        
        # 选择实体
        all_entities = sorted(entities.keys())
        selected_entity = st.selectbox(
            "选择聚焦实体",
            options=all_entities,
            help="查看该实体的时间线事件"
        )
        
        if not selected_entity:
            return
        
        # 查询时序数据
        kg_store = self._get_kg_store()
        timeline = kg_store.fetch_entity_timeline(selected_entity)
        
        if not timeline:
            st.info(f"实体 '{selected_entity}' 没有相关事件。")
            return
        
        st.info(f"📈 找到 {len(timeline)} 个相关事件")
        
        # 构建时序节点和边
        nodes = [(selected_entity, {
            "label": selected_entity,
            "color": "#e74c3c",
            "shape": "box",
            "size": 30,
            "level": 0,
            "font": {"size": 16, "bold": True}
        })]
        edges = []
        
        for i, event in enumerate(timeline):
            event_id = f"evt_{i}"
            timestamp = event.get("event_start_time") or event.get("reported_at") or "Unknown"
            summary = event.get("event_summary") or event.get("abstract", "")[:50]
            
            # 格式化时间
            time_label = self._format_timestamp(timestamp)
            label = f"{time_label}\n{summary}"
            
            nodes.append((event_id, {
                "label": label,
                "color": "#ff7f0e",
                "shape": "dot",
                "size": 20,
                "level": i + 1,
                "title": event.get("event_summary", "")
            }))
            
            # 连接实体到事件
            edges.append((selected_entity, event_id, {
                "color": "#95a5a6",
                "width": 2
            }))
            
            # 连接相邻事件（时序）
            if i > 0:
                edges.append((f"evt_{i-1}", event_id, {
                    "arrows": "to",
                    "color": "#3498db",
                    "width": 1,
                    "dashes": True
                }))
        
        # 渲染图谱
        self._render_pyvis(
            nodes=nodes,
            edges=edges,
            layout_config={
                "layout": {
                    "hierarchical": {
                        "enabled": True,
                        "direction": "LR",
                        "sortMethod": "directed",
                        "nodeSpacing": 150,
                        "levelSeparation": 200
                    }
                },
                "physics": {
                    "enabled": False
                }
            },
            directed=True
        )


class EntityRelationGraphRenderer(GraphRenderer):
    """实体关系图谱渲染器"""
    
    def render(self) -> None:
        """渲染实体关系图谱"""
        st.subheader("🌐 实体关系图谱")
        st.caption("显示实体间的语义关系网络")
        
        # 查询实体关系
        kg_store = self._get_kg_store()
        
        min_co = st.slider("最小共现次数", 1, 10, 2, help="共同出现在多少个事件中")
        relations = kg_store.fetch_entity_relations(min_co_occurrence=min_co)
        
        if not relations:
            st.info("没有找到符合条件的实体关系。请降低共现次数阈值。")
            return
        
        st.info(f"📈 找到 {len(relations)} 个实体关系")
        
        # 构建节点和边
        nodes = []
        edges = []
        node_set = set()
        
        # 收集所有实体
        for rel in relations:
            node_set.add(rel["entity1"])
            node_set.add(rel["entity2"])
        
        # 添加实体节点
        for entity in node_set:
            nodes.append((entity, {
                "label": entity,
                "color": "#1f77b4",
                "shape": "dot",
                "size": 25,
                "borderWidth": 2,
                "font": {"size": 12, "color": "#333333"}
            }))
        
        # 添加关系边（颜色根据共现次数）
        for rel in relations:
            co_occurrence = rel["co_occurrence"]
            
            # 颜色映射：共现次数越多，颜色越深
            if co_occurrence >= 5:
                color = "#e74c3c"  # 红色：强关系
            elif co_occurrence >= 3:
                color = "#f39c12"  # 橙色：中等关系
            else:
                color = "#95a5a6"  # 灰色：弱关系
            
            width = min(co_occurrence, 5)  # 边宽度
            
            edges.append((rel["entity1"], rel["entity2"], {
                "title": f"共现 {co_occurrence} 次",
                "color": color,
                "width": width,
                "smooth": {"enabled": True, "type": "dynamic"}
            }))
        
        # 渲染图谱
        self._render_pyvis(
            nodes=nodes,
            edges=edges,
            layout_config={
                "physics": {
                    "enabled": True,
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.3,
                        "springLength": 150,
                        "springConstant": 0.04,
                        "damping": 0.09
                    }
                }
            },
            directed=False
        )
        
        # 显示图例
        with st.expander("🎨 图例说明"):
            st.markdown("""
            **边颜色**：
            - 🔴 红色：强关系（共现 ≥ 5 次）
            - 🟠 橙色：中等关系（共现 3-4 次）
            - ⚪ 灰色：弱关系（共现 2 次）
            
            **边宽度**：表示关系强度
            """)


class EvolutionGraphRenderer(GraphRenderer):
    """动态演化图谱渲染器"""
    
    def render(self) -> None:
        """渲染动态演化图谱"""
        st.subheader("⏱️ 动态演化图谱")
        st.caption("显示实体关系随时间的变化")
        
        # 查询数据
        kg_store = self._get_kg_store()
        events = kg_store.fetch_events()
        
        if not events:
            st.warning("无事件数据。")
            return
        
        # 提取时间范围
        timestamps = self._extract_timestamps(events)
        
        if not timestamps:
            st.warning("无有效的时间戳数据。")
            return
        
        timestamps.sort()
        
        # 时间滑块
        st.markdown("**选择时间点**")
        selected_time_idx = st.slider(
            "时间轴",
            0,
            len(timestamps) - 1,
            len(timestamps) - 1,
            format="%d",
            help="拖动查看不同时间点的关系状态"
        )
        
        current_time = timestamps[selected_time_idx]
        st.info(f"📅 当前时间：{current_time.strftime('%Y-%m-%d %H:%M')}")
        
        # 筛选到当前时间点的事件
        events_until_now = []
        for evt in events:
            ts = evt.get("event_start_time") or evt.get("reported_at")
            if ts:
                try:
                    # 统一时区处理，避免naive和aware datetime比较错误
                    if ts.endswith('Z'):
                        dt = datetime.fromisoformat(ts[:-1] + "+00:00")
                    else:
                        dt = datetime.fromisoformat(ts)
                    if dt <= current_time:
                        events_until_now.append(evt)
                except Exception:
                    pass
        
        st.caption(f"截至当前时间，共 {len(events_until_now)} 个事件")
        
        # 计算实体关系（基于截至当前的事件）
        # 简化处理：显示提示信息
        st.info("🚧 此功能需要更复杂的时序关系分析，当前显示为占位实现。")
        st.markdown("""
        **将实现的功能**：
        - 按时间轴动态显示实体关系的建立与消亡
        - 边的颜色深浅表示关系强度随时间的变化
        - 支持动画播放，查看关系演变过程
        """)


class CausalGraphRenderer(GraphRenderer):
    """因果传播图谱渲染器"""
    
    def render(self) -> None:
        """渲染因果传播图谱"""
        st.subheader("🌀 因果传播图谱")
        st.caption("显示事件的级联影响链")
        
        # 加载事件数据
        events = self._load_events()
        if not events:
            st.warning("无事件数据。")
            return
        
        # 选择核心事件
        event_list = list(events.keys())
        selected_event = st.selectbox(
            "选择核心事件",
            options=event_list,
            format_func=lambda x: (events[x].get("event_summary", x) if isinstance(events[x], dict) else x)[:80]
        )
        
        if not selected_event:
            return
        
        # 显示占位信息
        st.info("🚧 此功能需要复杂的因果推断逻辑，当前显示为占位实现。")
        
        st.markdown(f"""
        **选中的核心事件**：{events[selected_event].get('event_summary', selected_event) if isinstance(events[selected_event], dict) else selected_event}
        
        **将实现的功能**：
        - 以核心事件为中心，放射状展示后续事件
        - 第一层：直接后续事件（时序相邻 + 实体共现）
        - 第二层：受影响的实体（状态变化）
        - 第三层：次级影响事件
        - 颜色编码：正向影响（绿）/ 负向影响（红）/ 中性（灰）
        - 边的粗细表示影响强度
        """)
        
        # 简单示例：显示相关实体
        if isinstance(events[selected_event], dict):
            related_entities = events[selected_event].get("entities", [])
            if related_entities:
                st.write("**相关实体**：", ", ".join(related_entities[:10]))






