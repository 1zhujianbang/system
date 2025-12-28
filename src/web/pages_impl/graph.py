"""
知识图谱页面

核心功能：PyVis 交互式图谱展示 + 实体聚焦
采用面向对象设计，提高代码可维护性和扩展性
"""
from __future__ import annotations

import json
import hashlib
import re
import html as html_std
import os
from datetime import datetime, timezone
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


class GraphStyle:
    """图谱视觉样式配置"""
    # 颜色配置
    COLOR_ENTITY_DEFAULT = "#4FA6D8"  # 柔和蓝
    COLOR_ENTITY_FOCUS = "#FF6B6B"    # 柔和红
    COLOR_EVENT = "#FFB347"           # 柔和橙
    
    COLOR_EDGE_DEFAULT = "#BDC3C7"    # 浅灰
    COLOR_EDGE_HIGHLIGHT = "#2ECC71"  # 绿色
    
    # 关系强度颜色
    COLOR_RELATION_WEAK = "#BDC3C7"   # 灰色
    COLOR_RELATION_MEDIUM = "#F39C12" # 橙色
    COLOR_RELATION_STRONG = "#C0392B" # 深红
    
    # 形状配置
    SHAPE_ENTITY = "dot"
    SHAPE_EVENT_GE = "dot"   # GE视图中事件改回球形
    SHAPE_EVENT_GET = "box"  # GET视图中保持方块
    SHAPE_TIMELINE_NODE = "circle"
    
    # 字体配置 (基础配置，大小将动态计算)
    FONT_BASE = {"face": "arial", "color": "#2C3E50"}
    
    # 物理引擎默认配置
    PHYSICS_DEFAULT = {
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08,
            "damping": 0.4,
            "avoidOverlap": 0
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
    }
    
    # 备用物理引擎（用于稀疏图）
    PHYSICS_BARNES_HUT = {
        "barnesHut": {
            "gravitationalConstant": -3000,
            "centralGravity": 0.3,
            "springLength": 150,
            "springConstant": 0.04,
            "damping": 0.09,
            "avoidOverlap": 0.1
        },
        "solver": "barnesHut"
    }
    
    # 时序布局配置
    LAYOUT_HIERARCHICAL = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "nodeSpacing": 150,
                "levelSeparation": 200
            }
        },
        "physics": {"enabled": False}
    }

    LAYOUT_HIERARCHICAL_TIMELINE = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "nodeSpacing": 220,
                "levelSeparation": 260
            }
        },
        "physics": {"enabled": False}
    }

    @staticmethod
    def get_font_config(size: int, is_focus: bool = False) -> Dict[str, Any]:
        """根据节点大小动态计算字体配置"""
        # 字体大小约为节点大小的 50%-60%，最小 10px
        font_size = max(10, int(size * 0.6))
        config = GraphStyle.FONT_BASE.copy()
        config["size"] = font_size
        if is_focus:
            config["bold"] = True
        return config

    @staticmethod
    def _truncate_label(text: str, limit: int = 9) -> str:
        """统一截断逻辑：超过 limit+3 长度则截断为 limit + '...'"""
        if len(text) > limit + 3:
            return text[:limit] + "..."
        return text

    @staticmethod
    def prepare_html_tooltip(html: str) -> str:
        """
        预处理 HTML Tooltip：
        1. 压缩去除换行和多余空格
        """
        return " ".join(html.split())

    @staticmethod
    def _wrap_text_html(text: Any, width: int = 25) -> str:
        s = str(text or "")
        if not s:
            return ""
        chunks = [s[i : i + int(width)] for i in range(0, len(s), int(width))]
        return "<br/>".join(html_std.escape(c) for c in chunks)

    @staticmethod
    def _escape_attr(text: Any) -> str:
        return html_std.escape(str(text or ""), quote=True)

    @staticmethod
    def _normalize_event_types(event_data: Dict[str, Any]) -> List[str]:
        types: List[str] = []
        raw = event_data.get("event_types")
        if isinstance(raw, list):
            types = [str(x).strip() for x in raw if isinstance(x, str) and x.strip()]
        elif isinstance(raw, str) and raw.strip():
            types = [raw.strip()]
        else:
            raw2 = event_data.get("event_type")
            if isinstance(raw2, str) and raw2.strip():
                types = [raw2.strip()]
        return types

    @staticmethod
    def generate_event_tooltip(event_data: Dict[str, Any]) -> str:
        """生成美观的 HTML Table Tooltip"""
        summary = event_data.get("event_summary", "No Summary")
        time_str = event_data.get("event_start_time") or event_data.get("reported_at") or "Unknown"
        entities = event_data.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        entities = [str(x).strip() for x in entities if isinstance(x, str) and x.strip()]
        event_types = GraphStyle._normalize_event_types(event_data)
        
        # ... (省略中间的数据处理逻辑，保持不变) ...
        # 格式化时间显示
        try:
             if "T" in time_str:
                 dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                 time_display = dt.strftime("%Y-%m-%d %H:%M")
             else:
                 time_display = time_str
        except:
             time_display = time_str

        if entities:
            ent_badges = "".join(
                [
                    (
                        f'<span data-entity="{GraphStyle._escape_attr(e)}" '
                        f'style="background-color: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 4px; '
                        f'font-size: 11px; margin-right: 4px; display: inline-block; margin-bottom: 4px; '
                        f'border: 1px solid #bbdefb; cursor: pointer; user-select: none;">{html_std.escape(e)}</span>'
                    )
                    for e in entities
                ]
            )
        else:
            ent_badges = '<span style="color: #999; font-style: italic;">None</span>'

        if event_types:
            type_badges = "".join(
                [
                    (
                        f'<span data-event-type="{GraphStyle._escape_attr(t)}" '
                        f'style="background-color: #f3e5f5; color: #6a1b9a; padding: 2px 6px; border-radius: 4px; '
                        f'font-size: 11px; margin-right: 4px; display: inline-block; margin-bottom: 4px; '
                        f'border: 1px solid #e1bee7; user-select: none;">{html_std.escape(t)}</span>'
                    )
                    for t in event_types
                ]
            )
        else:
            type_badges = '<span style="color: #999; font-style: italic;">None</span>'
        summary_display = html_std.escape(str(summary or "")) or '<span style="color: #999; font-style: italic;">None</span>'

        html = f"""
        <table style="font-family: Arial, sans-serif; border-collapse: collapse; width: 300px; background-color: white; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color: #f8f9fa; border-bottom: 2px solid #e9ecef;">
                    <th colspan="2" style="padding: 10px; text-align: left; color: #343a40; font-size: 14px;">事件详情</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #f1f3f5;">
                    <td style="padding: 8px; width: 60px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">时间</td>
                    <td style="padding: 8px; color: #495057; font-size: 12px;">{html_std.escape(str(time_display))}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f3f5;">
                    <td style="padding: 8px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">类型</td>
                    <td style="padding: 8px;">
                        <div style="max-height: 90px; overflow: auto; padding-right: 4px;">
                            {type_badges}
                        </div>
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f3f5;">
                    <td style="padding: 8px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">摘要</td>
                    <td style="padding: 8px; color: #212529; font-size: 12px; line-height: 1.4;">{summary_display}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">涉及</td>
                    <td style="padding: 8px;">
                        <div style="max-height: 120px; overflow: auto; padding-right: 4px;">
                            {ent_badges}
                        </div>
                    </td>
                </tr>
            </tbody>
        </table>
        """
        return GraphStyle.prepare_html_tooltip(html)

    @staticmethod
    def generate_entity_tooltip(entity_name: str, info: Dict[str, Any]) -> str:
        """生成实体 Tooltip (表格样式)"""
        count = info.get("count", 0)
        html = f"""
        <table style="font-family: Arial, sans-serif; border-collapse: collapse; min-width: 180px; background-color: white; border-radius: 6px; overflow: hidden;">
            <tr style="background-color: #e3f2fd; border-bottom: 1px solid #bbdefb;">
                <th colspan="2" style="padding: 8px; text-align: left; color: #1565c0; font-size: 13px;">{html_std.escape(entity_name)}</th>
            </tr>
            <tr>
                <td style="padding: 8px; color: #666; font-size: 12px;">出现频次</td>
                <td style="padding: 8px; color: #333; font-weight: bold; font-size: 12px;">{html_std.escape(str(count))}</td>
            </tr>
        </table>
        """
        return GraphStyle.prepare_html_tooltip(html)

    @staticmethod
    def generate_relation_tooltip(entity1: str, entity2: str, items: List[Tuple[str, str]]) -> str:
        rows = ""
        for time_display, summary in items:
            rows += (
                "<tr style=\"border-bottom: 1px solid #f1f3f5;\">"
                f"<td style=\"padding: 6px 8px; width: 92px; color: #495057; font-size: 12px; vertical-align: top;\">{html_std.escape(str(time_display))}</td>"
                f"<td style=\"padding: 6px 8px; color: #212529; font-size: 12px; line-height: 1.4;\">{html_std.escape(str(summary or ''))}</td>"
                "</tr>"
            )

        if not rows:
            rows = (
                "<tr>"
                "<td colspan=\"2\" style=\"padding: 8px; color: #999; font-style: italic; font-size: 12px;\">None</td>"
                "</tr>"
            )

        html = f"""
        <table style="font-family: Arial, sans-serif; border-collapse: collapse; width: 360px; background-color: white; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color: #f8f9fa; border-bottom: 2px solid #e9ecef;">
                    <th colspan="2" style="padding: 10px; text-align: left; color: #343a40; font-size: 14px;">{html_std.escape(entity1)} ↔ {html_std.escape(entity2)}</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #f1f3f5;">
                    <td style="padding: 8px; width: 92px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">关系</td>
                    <td style="padding: 8px; color: #495057; font-size: 12px;">按时间排序的共现事件</td>
                </tr>
                <tr>
                    <td colspan="2" style="padding: 0;">
                        <div style="max-height: 180px; overflow: auto;">
                            <table style="border-collapse: collapse; width: 100%;">{rows}</table>
                        </div>
                    </td>
                </tr>
            </tbody>
        </table>
        """
        return GraphStyle.prepare_html_tooltip(html)

    @staticmethod
    def generate_entity_relation_tooltip(entity1: str, entity2: str, items: List[Tuple[str, str, str]]) -> str:
        rows = ""
        for time_display, predicate, summary in items:
            rows += (
                "<tr style=\"border-bottom: 1px solid #f1f3f5;\">"
                f"<td style=\"padding: 6px 8px; width: 92px; color: #495057; font-size: 12px; vertical-align: top;\">{html_std.escape(str(time_display))}</td>"
                f"<td style=\"padding: 6px 8px; width: 92px; color: #6a1b9a; font-size: 12px; vertical-align: top;\">{html_std.escape(str(predicate or ''))}</td>"
                f"<td style=\"padding: 6px 8px; color: #212529; font-size: 12px; line-height: 1.4;\">{html_std.escape(str(summary or ''))}</td>"
                "</tr>"
            )

        if not rows:
            rows = (
                "<tr>"
                "<td colspan=\"3\" style=\"padding: 8px; color: #999; font-style: italic; font-size: 12px;\">None</td>"
                "</tr>"
            )

        html = f"""
        <table style="font-family: Arial, sans-serif; border-collapse: collapse; width: 420px; background-color: white; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color: #f8f9fa; border-bottom: 2px solid #e9ecef;">
                    <th colspan="3" style="padding: 10px; text-align: left; color: #343a40; font-size: 14px;">{html_std.escape(entity1)} → {html_std.escape(entity2)}</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #f1f3f5;">
                    <td style="padding: 8px; width: 92px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">时间</td>
                    <td style="padding: 8px; width: 92px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">关系</td>
                    <td style="padding: 8px; color: #868e96; font-size: 12px; font-weight: bold; vertical-align: top;">事件</td>
                </tr>
                <tr>
                    <td colspan="3" style="padding: 0;">
                        <div style="max-height: 180px; overflow: auto;">
                            <table style="border-collapse: collapse; width: 100%;">{rows}</table>
                        </div>
                    </td>
                </tr>
            </tbody>
        </table>
        """
        return GraphStyle.prepare_html_tooltip(html)


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
        backend = str(os.getenv("KG_STORE_BACKEND") or "").strip().lower() or "sqlite"
        if backend == "neo4j":
            try:
                from src.adapters.graph_store.neo4j_adapter import get_neo4j_store

                return get_neo4j_store()
            except Exception:
                pass
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
    
    def _normalize_timestamp(self, ts: str | None) -> datetime | None:
        """标准化时间戳处理"""
        if not ts:
            return None
        try:
            # 统一移除 Z 后缀并处理时区
            ts_str = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_str)
            
            # 如果是 naive 时间（如 YYYY-MM-DD），假定为 UTC 0点
            if dt.tzinfo is None:
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _extract_timestamps(self, events: List[Dict[str, Any]]) -> List[datetime]:
        """从事件列表中提取有效的时间戳"""
        timestamps = []
        for evt in events:
            ts = evt.get("event_start_time") or evt.get("reported_at")
            dt = self._normalize_timestamp(ts)
            if dt:
                timestamps.append(dt)
        return sorted(timestamps)

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
            base_interaction = {
                "interaction": {
                    "hover": True,
                    "hoverConnectedEdges": True,
                    "tooltipDelay": 0,
                }
            }
            merged = self._deep_merge_dict(base_interaction, layout_config or {})
            net.set_options(json.dumps(merged))
            
            # 添加节点
            for node_id, node_attrs in nodes:
                net.add_node(node_id, **node_attrs)
            
            # 添加边
            for u, v, edge_attrs in edges:
                net.add_edge(u, v, **edge_attrs)
            
            # 生成 HTML
            nodes_for_hash = sorted([(str(nid), attrs or {}) for nid, attrs in nodes], key=lambda x: x[0])
            edges_for_hash = sorted(
                [(str(u), str(v), attrs or {}) for u, v, attrs in edges],
                key=lambda x: (x[0], x[1]),
            )
            content_for_hash = json.dumps(
                {"v": 3, "nodes": nodes_for_hash, "edges": edges_for_hash, "options": merged},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            graph_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:16]
            html_path = self.cache_path / f"graph_{graph_hash}.html"
            
            net.save_graph(str(html_path))
            
            # 读取并显示
            html_content = html_path.read_text(encoding="utf-8")

            html_content = self._postprocess_pyvis_html(html_content)

            components.html(html_content, height=720, scrolling=True)
            
        except ImportError:
            st.error("PyVis 未安装。请运行: pip install pyvis")
        except Exception as e:
            st.error(f"图谱渲染失败: {e}")

    def _deep_merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (base or {}).items():
            if isinstance(v, dict):
                out[k] = self._deep_merge_dict(v, {})
            else:
                out[k] = v
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = self._deep_merge_dict(out.get(k) or {}, v)
            else:
                out[k] = v
        return out

    def _postprocess_pyvis_html(self, html_content: str) -> str:
        html_title_fn = (
            "\n\nfunction htmlTitle(html) {\n"
            "  const container = document.createElement(\"div\");\n"
            "  container.innerHTML = html;\n"
            "  return container;\n"
            "}\n"
            "\nfunction mlInitTooltips(network, nodes, edges) {\n"
            "  var tooltip = document.getElementById('ml-tooltip');\n"
            "  if (!tooltip) {\n"
            "    tooltip = document.createElement('div');\n"
            "    tooltip.id = 'ml-tooltip';\n"
            "    tooltip.style.position = 'fixed';\n"
            "    tooltip.style.zIndex = '9999';\n"
            "    tooltip.style.maxWidth = '380px';\n"
            "    tooltip.style.background = 'transparent';\n"
            "    tooltip.style.visibility = 'hidden';\n"
            "    tooltip.style.opacity = '0';\n"
            "    tooltip.style.transition = 'opacity 260ms ease';\n"
            "    tooltip.style.pointerEvents = 'auto';\n"
            "    document.body.appendChild(tooltip);\n"
            "  }\n"
            "\n"
            "  var hideTimer = null;\n"
            "  var lastPointer = { x: 0, y: 0 };\n"
            "  var containerEl = null;\n"
            "  var overTooltip = false;\n"
            "  try { containerEl = network && network.body && network.body.container; } catch (e) { containerEl = null; }\n"
            "\n"
            "  function updatePointerFromParams(params) {\n"
            "    try {\n"
            "      if (params && params.event && params.event.srcEvent && typeof params.event.srcEvent.clientX === 'number') {\n"
            "        lastPointer = { x: params.event.srcEvent.clientX, y: params.event.srcEvent.clientY };\n"
            "        return;\n"
            "      }\n"
            "      if (containerEl && params && params.pointer && params.pointer.DOM) {\n"
            "        var r = containerEl.getBoundingClientRect();\n"
            "        lastPointer = { x: r.left + params.pointer.DOM.x, y: r.top + params.pointer.DOM.y };\n"
            "      }\n"
            "    } catch (e) {}\n"
            "  }\n"
            "\n"
            "  function setVisible(v) {\n"
            "    if (v) {\n"
            "      tooltip.style.visibility = 'visible';\n"
            "      tooltip.style.opacity = '1';\n"
            "    } else {\n"
            "      tooltip.style.opacity = '0';\n"
            "      window.setTimeout(function() {\n"
            "        if (tooltip.style.opacity === '0') tooltip.style.visibility = 'hidden';\n"
            "      }, 280);\n"
            "    }\n"
            "  }\n"
            "\n"
            "  function scheduleHide() {\n"
            "    if (hideTimer) window.clearTimeout(hideTimer);\n"
            "    hideTimer = window.setTimeout(function() { setVisible(false); }, 400);\n"
            "  }\n"
            "\n"
            "  function cancelHide() {\n"
            "    if (hideTimer) window.clearTimeout(hideTimer);\n"
            "    hideTimer = null;\n"
            "  }\n"
            "\n"
            "  function clampToViewport(x, y) {\n"
            "    var pad = 12;\n"
            "    tooltip.style.left = '0px';\n"
            "    tooltip.style.top = '0px';\n"
            "    var rect = tooltip.getBoundingClientRect();\n"
            "    var nx = x + pad;\n"
            "    var ny = y + pad;\n"
            "    var maxX = window.innerWidth - rect.width - pad;\n"
            "    var maxY = window.innerHeight - rect.height - pad;\n"
            "    if (nx > maxX) nx = Math.max(pad, maxX);\n"
            "    if (ny > maxY) ny = Math.max(pad, maxY);\n"
            "    tooltip.style.left = nx + 'px';\n"
            "    tooltip.style.top = ny + 'px';\n"
            "  }\n"
            "\n"
            "  function showHtml(content, x, y) {\n"
            "    if (!content) {\n"
            "      scheduleHide();\n"
            "      return;\n"
            "    }\n"
            "    cancelHide();\n"
            "    if (typeof content === 'string') {\n"
            "      tooltip.innerHTML = content;\n"
            "    } else if (content instanceof Element) {\n"
            "      tooltip.innerHTML = '';\n"
            "      tooltip.appendChild(content.cloneNode(true));\n"
            "    } else if (content && content.nodeType === 1) {\n"
            "      tooltip.innerHTML = '';\n"
            "      tooltip.appendChild(content.cloneNode(true));\n"
            "    } else {\n"
            "      tooltip.textContent = String(content);\n"
            "    }\n"
            "    setVisible(true);\n"
            "    clampToViewport(x, y);\n"
            "  }\n"
            "\n"
            "  tooltip.addEventListener('mouseenter', function() { overTooltip = true; cancelHide(); });\n"
            "  tooltip.addEventListener('mouseleave', function() { overTooltip = false; scheduleHide(); });\n"
            "  tooltip.addEventListener('click', function(e) {\n"
            "    var t = e.target;\n"
            "    while (t && t !== tooltip && !t.getAttribute('data-entity')) t = t.parentElement;\n"
            "    if (t && t.getAttribute && t.getAttribute('data-entity')) {\n"
            "      e.preventDefault();\n"
            "      var ent = t.getAttribute('data-entity');\n"
            "      try {\n"
            "        var node = nodes.get(ent);\n"
            "        if (node) {\n"
            "          network.selectNodes([ent]);\n"
            "          network.focus(ent, { scale: 1.2, animation: { duration: 320, easingFunction: 'easeInOutQuad' } });\n"
            "        }\n"
            "      } catch (err) {}\n"
            "    }\n"
            "  });\n"
            "\n"
            "  network.on('mousemove', function(params) {\n"
            "    updatePointerFromParams(params);\n"
            "    if (tooltip.style.visibility === 'visible') clampToViewport(lastPointer.x, lastPointer.y);\n"
            "  });\n"
            "\n"
            "  network.on('hoverNode', function(params) {\n"
            "    try {\n"
            "      updatePointerFromParams(params);\n"
            "      var n = nodes.get(params.node);\n"
            "      var html = (n && (n._ml_title || n.title)) || '';\n"
            "      showHtml(html, lastPointer.x, lastPointer.y);\n"
            "    } catch (err) {}\n"
            "  });\n"
            "  network.on('blurNode', function() { if (overTooltip) return; scheduleHide(); });\n"
            "\n"
            "  network.on('hoverEdge', function(params) {\n"
            "    try {\n"
            "      updatePointerFromParams(params);\n"
            "      var ed = edges.get(params.edge);\n"
            "      var html = (ed && (ed._ml_title || ed.title)) || '';\n"
            "      showHtml(html, lastPointer.x, lastPointer.y);\n"
            "    } catch (err) {}\n"
            "  });\n"
            "  network.on('blurEdge', function() { if (overTooltip) return; scheduleHide(); });\n"
            "}\n"
        )

        if "function htmlTitle(" not in html_content:
            inserted = False
            for insert_after in ("var filter = {", "var options, data;"):
                idx = html_content.find(insert_after)
                if idx == -1:
                    continue
                end_idx = html_content.find("};", idx)
                if end_idx != -1:
                    end_idx = end_idx + 2
                    html_content = html_content[:end_idx] + html_title_fn + html_content[end_idx:]
                    inserted = True
                    break

            if not inserted:
                html_content = re.sub(
                    r'(<script[^>]*type="text/javascript"[^>]*>\s*)',
                    r"\1" + html_title_fn,
                    html_content,
                    count=1,
                )

        if "var pyvisNodes" not in html_content:
            node_pattern = r"nodes\s*=\s*new vis\.DataSet\((\[[\s\S]*?\])\);"
            node_repl = (
                "var pyvisNodes = \\1;\n"
                "for (var i = 0; i < pyvisNodes.length; i++) {\n"
                "  if (pyvisNodes[i].title) {\n"
                "    pyvisNodes[i].title = htmlTitle(pyvisNodes[i].title);\n"
                "  }\n"
                "}\n"
                "nodes = new vis.DataSet(pyvisNodes);"
            )
            html_content = re.sub(node_pattern, node_repl, html_content, count=1)

        if "var pyvisEdges" not in html_content:
            edge_pattern = r"edges\s*=\s*new vis\.DataSet\((\[[\s\S]*?\])\);"
            edge_repl = (
                "var pyvisEdges = \\1;\n"
                "for (var i = 0; i < pyvisEdges.length; i++) {\n"
                "  if (pyvisEdges[i].title) {\n"
                "    pyvisEdges[i].title = htmlTitle(pyvisEdges[i].title);\n"
                "  }\n"
                "}\n"
                "edges = new vis.DataSet(pyvisEdges);"
            )
            html_content = re.sub(edge_pattern, edge_repl, html_content, count=1)

        if "mlInitTooltips(network, nodes, edges);" not in html_content:
            html_content = re.sub(
                r"(network\s*=\s*new vis\.Network\(container,\s*data,\s*options\);\s*)",
                r"\1\nmlInitTooltips(network, nodes, edges);\n",
                html_content,
                count=1,
            )

        return html_content


class SnapshotGraphRenderer(GraphRenderer):
    def _build_pyvis_payload(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        *,
        focus_node: str = "",
        max_nodes: int = 75,
        max_edges: int = 100,
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
        events_map = self._load_events()
        for n in nodes2:
            nid = str(n.get("id"))
            ntype = str(n.get("type") or "entity").strip() or "entity"
            label = str(n.get("label") or nid)
            
            # 样式逻辑
            is_focus = bool(focus_node) and nid == focus_node
            
            color = GraphStyle.COLOR_ENTITY_DEFAULT
            shape = GraphStyle.SHAPE_ENTITY
            size = 22
            
            # 生成 Tooltip
            if ntype == "event":
                evt_data: Dict[str, Any] = {}
                for cand in (n.get("abstract"), n.get("label"), n.get("description"), nid):
                    if isinstance(cand, str) and cand in events_map and isinstance(events_map.get(cand), dict):
                        evt_data = dict(events_map.get(cand) or {})
                        break

                evt_data = {
                    **evt_data,
                    "event_summary": evt_data.get("event_summary") or n.get("description") or n.get("label") or nid,
                    "event_start_time": evt_data.get("event_start_time") or n.get("time") or n.get("timestamp") or evt_data.get("reported_at"),
                    "reported_at": evt_data.get("reported_at") or n.get("reported_at"),
                    "entities": evt_data.get("entities") or [],
                    "event_types": evt_data.get("event_types") or [],
                }
                title = GraphStyle.generate_event_tooltip(evt_data)
                
                color = GraphStyle.COLOR_EVENT
                shape = GraphStyle.SHAPE_EVENT_GE 
                size = 18
            else:
                # 实体 Tooltip
                title = GraphStyle.generate_entity_tooltip(nid, {"count": deg.get(nid, 0)})
                
                if is_focus:
                    color = GraphStyle.COLOR_ENTITY_FOCUS
                    size = 30
                else:
                    raw_color = str(n.get("color") or "")
                    if raw_color and raw_color.startswith("#"):
                        color = raw_color

            d = deg.get(nid, 0)
            size = min(size + int(d / 3), 40)
            
            if ntype == "relation_state":
                shape = "box"
                
            # 统一截断 Label
            display_label = GraphStyle._truncate_label(label)
            
            pyvis_nodes.append(
                (
                    nid,
                    {
                        "label": display_label,
                        "color": color,
                        "shape": shape,
                        "size": size,
                        "_ml_title": title,
                        "borderWidth": 2,
                        "borderWidthSelected": 3,
                        "font": GraphStyle.get_font_config(size, is_focus),
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
            
            color = GraphStyle.COLOR_EDGE_DEFAULT
            if etype in {"before"}:
                color = "#3498db" # 保留特殊语义颜色
            if etype in {"evolved_to", "evolve"}:
                color = "#9b59b6"
                
            pyvis_edges.append(
                (
                    u,
                    v,
                    {
                        "_ml_title": edge_title,
                        "width": 2,
                        "color": {"color": color, "highlight": GraphStyle.COLOR_EDGE_HIGHLIGHT, "hover": GraphStyle.COLOR_EDGE_HIGHLIGHT, "opacity": 0.6},
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
            max_nodes = st.slider("最大节点数", 25, 500, 75, 10)
            max_edges = st.slider("最大边数", 25, 1000, 100, 10)
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

        if graph_type == "GET":
            # GET 类型应该使用时序渲染逻辑（或类似的层级展示），但 pyvis 的 timeline 需要特殊处理
            # 这里我们尝试复用 TimelineGraphRenderer 的部分逻辑，或者简单地使用层级布局
            # 为了简单起见，如果检测到是 GET，我们强制启用层级布局
            layout_config = GraphStyle.LAYOUT_HIERARCHICAL
        else:
            layout_config = {
                "physics": GraphStyle.PHYSICS_BARNES_HUT # Snapshot may be large, BarnesHut is safer
            }

        self._render_pyvis(
            nodes=pyvis_nodes,
            edges=pyvis_edges,
            layout_config=layout_config,
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
    """实体-事件关系图谱渲染器 (GE)"""
    
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
            
            # 1. 实体筛选
            all_entities = sorted(entities.keys())
            focus_entity = st.selectbox(
                "聚焦实体",
                options=["(全部)"] + all_entities,
                index=0,
                help="选择一个实体查看其关联"
            )

            all_event_types: Set[str] = set()
            for evt_data in events.values():
                if not isinstance(evt_data, dict):
                    continue
                raw_types = evt_data.get("event_types")
                if isinstance(raw_types, list):
                    for t in raw_types:
                        if isinstance(t, str) and t.strip():
                            all_event_types.add(t.strip())
                elif isinstance(raw_types, str) and raw_types.strip():
                    all_event_types.add(raw_types.strip())
            focus_event_type = st.selectbox(
                "聚焦事件类型",
                options=["(全部)"] + sorted(all_event_types),
                index=0,
                help="只显示包含该类型的事件"
            )
            
            # 2. 时间筛选
            timestamps = self._extract_timestamps(list(events.values()))
            if timestamps:
                min_time, max_time = min(timestamps), max(timestamps)
                # 转换为 date 对象以便 slider 使用
                min_date, max_date = min_time.date(), max_time.date()
                
                if min_date != max_date:
                    date_range = st.slider(
                        "时间范围",
                        min_value=min_date,
                        max_value=max_date,
                        value=(min_date, max_date)
                    )
                else:
                    date_range = (min_date, max_date)
            else:
                date_range = None

            # 3. 显示设置
            max_nodes = st.slider("最大节点数", 25, 750, 100, 5)
            physics_mode = st.selectbox("布局算法", ["ForceAtlas2 (推荐)", "BarnesHut (传统)"], index=0)
            
            st.divider()
            st.caption(f"📊 总实体: {len(entities)} | 总事件: {len(events)}")
        
        # --- 数据预处理与过滤 ---
        edge_list = []
        valid_events = set()
        
        for evt_abstract, evt_data in events.items():
            if not isinstance(evt_data, dict):
                continue

            if focus_event_type != "(全部)":
                evt_types = evt_data.get("event_types")
                evt_types_norm: List[str] = []
                if isinstance(evt_types, list):
                    evt_types_norm = [x.strip() for x in evt_types if isinstance(x, str) and x.strip()]
                elif isinstance(evt_types, str) and evt_types.strip():
                    evt_types_norm = [evt_types.strip()]
                if focus_event_type not in evt_types_norm:
                    continue
                
            # 时间过滤
            if date_range:
                evt_ts_str = evt_data.get("event_start_time") or evt_data.get("reported_at")
                evt_date = self._normalize_timestamp(evt_ts_str)
                if evt_date:
                    # 比较 date 部分
                    if not (date_range[0] <= evt_date.date() <= date_range[1]):
                        continue
            
            evt_id = f"EVT:{evt_abstract}"
            evt_summary = evt_data.get("event_summary", evt_abstract)
            
            has_valid_entity = False
            for ent in evt_data.get("entities", []):
                if ent in entities:
                    edge_list.append((evt_id, ent, {}))
                    has_valid_entity = True
            
            if has_valid_entity:
                valid_events.add(evt_id)
        
        if not edge_list:
             st.info("当前时间范围内没有关联数据。")
             st.stop()

        # --- 图谱拓扑构建 ---
        # 计算度数
        deg = defaultdict(int)
        for u, v, _ in edge_list:
            deg[u] += 1
            deg[v] += 1
            
        # 确定目标节点集合
        target_nodes = set()
        
        if focus_entity != "(全部)":
            # 聚焦模式：BFS 2跳
            target_nodes.add(focus_entity)
            adj = defaultdict(set)
            for u, v, _ in edge_list:
                adj[u].add(v)
                adj[v].add(u)
                
            frontier = {focus_entity}
            for _ in range(2):
                next_frontier = set()
                for node in frontier:
                    next_frontier |= adj.get(node, set())
                next_frontier -= target_nodes
                target_nodes |= next_frontier
                frontier = next_frontier
        else:
            # 全局模式：按度数 Top N
            # 优先保留高频实体和事件
            all_nodes_sorted = sorted(deg.keys(), key=lambda x: deg[x], reverse=True)
            target_nodes = set(all_nodes_sorted[:max_nodes])
        
        # 最终边过滤
        filtered_edges = [
            (u, v, d) for u, v, d in edge_list
            if u in target_nodes and v in target_nodes
        ]
        
        if not filtered_edges:
            st.info("筛选后无数据。")
            st.stop()
            
        st.info(f"📈 显示 {len(target_nodes)} 个节点 (实体/事件), {len(filtered_edges)} 条关联")

        # --- PyVis 节点与边构建 ---
        nodes = []
        edges = []
        added_nodes = set()
        
        for u, v, d in filtered_edges:
            for node in [u, v]:
                if node in added_nodes:
                    continue
                added_nodes.add(node)
                
                is_focus = (focus_entity != "(全部)" and node == focus_entity)
                
                if node.startswith("EVT:"):
                    # 事件节点
                    raw_text = node[4:]
                    # 截断逻辑
                    label = GraphStyle._truncate_label(raw_text)
                    
                    # 获取完整的事件数据用于 Tooltip
                    evt_data = events.get(raw_text, {})
                    title = GraphStyle.generate_event_tooltip(evt_data)
                    
                    size = 15 + min(deg[node], 10) # 稍微调小一点，因为是球形
                    
                    nodes.append((node, {
                        "label": label,
                        "_ml_title": title,
                        "color": GraphStyle.COLOR_EVENT,
                        "shape": GraphStyle.SHAPE_EVENT_GE,
                        "size": size,
                        "font": GraphStyle.get_font_config(size)
                    }))
                else:
                    # 实体节点
                    ent_info = entities.get(node, {})
                    title = GraphStyle.generate_entity_tooltip(node, ent_info)
                    
                    size = 25 if is_focus else 15 + min(deg[node], 15)
                    
                    # 截断实体名称
                    label = GraphStyle._truncate_label(node)
                    
                    nodes.append((node, {
                        "label": label,
                        "_ml_title": title,
                        "color": GraphStyle.COLOR_ENTITY_FOCUS if is_focus else GraphStyle.COLOR_ENTITY_DEFAULT,
                        "shape": GraphStyle.SHAPE_ENTITY,
                        "size": size,
                        "font": GraphStyle.get_font_config(size, is_focus)
                    }))
            
            # 边
            edges.append((u, v, {
                "color": GraphStyle.COLOR_EDGE_DEFAULT,
                "width": 1,
                "hoverWidth": 2
            }))
            
        # --- 渲染 ---
        physics_config = GraphStyle.PHYSICS_DEFAULT if physics_mode.startswith("ForceAtlas2") else GraphStyle.PHYSICS_BARNES_HUT
        
        self._render_pyvis(
            nodes=nodes,
            edges=edges,
            layout_config={"physics": physics_config},
            directed=False
        )
        
        st.divider()
        self._render_entity_event_list(entities, {k: v for k, v in events.items() if f"EVT:{k}" in target_nodes})


class TimelineGraphRenderer(GraphRenderer):
    """实体时序图谱渲染器 (GET)"""
    
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
        
        col1, col2 = st.columns([2, 1])
        with col1:
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

        events_map = self._load_events()
            
        # 提取时间并排序
        events_with_time = []
        for event in timeline:
            abstract_key = event.get("abstract")
            if isinstance(abstract_key, str) and abstract_key in events_map and isinstance(events_map.get(abstract_key), dict):
                merged = dict(events_map.get(abstract_key) or {})
                merged.update(event)
                event = merged

            ts_str = event.get("event_start_time") or event.get("reported_at")
            dt = self._normalize_timestamp(ts_str)
            if dt:
                events_with_time.append((dt, event))
                
        events_with_time.sort(key=lambda x: x[0])
        
        if not events_with_time:
             st.info("无法解析时间信息的事件。")
             return

        # 时间范围筛选
        min_date, max_date = events_with_time[0][0].date(), events_with_time[-1][0].date()
        
        with col2:
            if min_date != max_date:
                start_d, end_d = st.slider(
                    "时间范围",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date)
                )
            else:
                start_d, end_d = min_date, max_date
                
        # 过滤
        filtered_timeline = [
            (dt, evt) for dt, evt in events_with_time
            if start_d <= dt.date() <= end_d
        ]
        
        st.info(f"📈 显示 {len(filtered_timeline)} 个关键节点")
        
        # 构建时序节点和边
        nodes = []
        edges = []
        
        # 实体作为起始点
        entity_title = GraphStyle.generate_entity_tooltip(selected_entity, entities.get(selected_entity, {"count": 0}) if isinstance(entities, dict) else {"count": 0})
        nodes.append((selected_entity, {
            "label": selected_entity,
            "_ml_title": entity_title,
            "color": GraphStyle.COLOR_ENTITY_FOCUS,
            "shape": GraphStyle.SHAPE_ENTITY,
            "size": 30,
            "level": 0,
            "font": GraphStyle.get_font_config(30, is_focus=True)
        }))
        
        prev_node_id = selected_entity
        
        for i, (dt, event) in enumerate(filtered_timeline):
            event_id = f"evt_{i}"
            summary = event.get("event_summary") or event.get("abstract", "")[:50]
            
            # 格式化多行标签
            date_str = dt.strftime("%Y-%m-%d")
            # 自动换行 summary 并截断过长的文本
            summary_truncated = GraphStyle._truncate_label(summary, limit=60) # 限制总长度
            wrapped_summary = "\n".join([summary_truncated[i:i+15] for i in range(0, len(summary_truncated), 15)])
            label = f"[{date_str}]\n{wrapped_summary}"
            
            # 使用 HTML Tooltip
            title = GraphStyle.generate_event_tooltip(event)
            
            nodes.append((event_id, {
                "label": label, # 时序图的Label包含时间信息，且已经是多行，不应用通用截断
                "color": GraphStyle.COLOR_EVENT,
                "shape": GraphStyle.SHAPE_EVENT_GET,
                "margin": 10,
                "level": i + 1,
                "_ml_title": title,
                "font": {"face": "arial", "size": 12, "color": "#2C3E50"}, # Box shape font size works differently
                "shapeProperties": {"borderRadius": 6} # 圆角美化
            }))
            
            # 实体指向事件（虚线表示关联）
            edges.append((selected_entity, event_id, {
                "color": GraphStyle.COLOR_EDGE_DEFAULT,
                "width": 1,
                "dashes": True,
                "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.2}
            }))
            
            # 时间轴连线（实线）
            if prev_node_id != selected_entity:
                edges.append((prev_node_id, event_id, {
                    "arrows": "to",
                    "color": GraphStyle.COLOR_EDGE_HIGHLIGHT,
                    "width": 2
                }))
            prev_node_id = event_id
        
        # 渲染图谱
        self._render_pyvis(
            nodes=nodes,
            edges=edges,
            layout_config=GraphStyle.LAYOUT_HIERARCHICAL_TIMELINE,
            directed=True
        )


class EntityRelationGraphRenderer(GraphRenderer):
    """实体关系图谱渲染器 (EE)"""
    
    def render(self) -> None:
        """渲染实体关系图谱"""
        st.subheader("🌐 实体关系图谱")
        st.caption("显示实体间的语义关系网络")
        
        # 查询实体关系
        kg_store = self._get_kg_store()
        
        col1, col2 = st.columns(2)
        with col1:
             min_co = st.slider("最小共现次数", 1, 10, 2, help="共同出现在多少个事件中")
             max_nodes = st.slider("最大显示节点数", 10, 300, 100, 10, help="限制显示的实体数量，优先显示关系强的实体")
        
        relations = kg_store.fetch_entity_relations(min_co_occurrence=min_co)
        
        if not relations:
            st.info("没有找到符合条件的实体关系。请降低共现次数阈值。")
            return

        # 收集所有相关实体
        all_related_entities = set()
        entity_weights = defaultdict(int) # 计算实体权重（基于关系强度）
        
        for rel in relations:
            u, v = rel["entity1"], rel["entity2"]
            w = rel["co_occurrence"]
            all_related_entities.add(u)
            all_related_entities.add(v)
            entity_weights[u] += w
            entity_weights[v] += w
            
        with col2:
             focus_entity = st.selectbox("聚焦特定实体", ["(全部)"] + sorted(list(all_related_entities)))
        
        # 过滤
        if focus_entity != "(全部)":
            relations = [
                r for r in relations 
                if r["entity1"] == focus_entity or r["entity2"] == focus_entity
            ]
            # 聚焦模式下，只保留相关节点，且不受 max_nodes 严格限制（或者只限制二阶邻居）
            target_nodes = set()
            for r in relations:
                target_nodes.add(r["entity1"])
                target_nodes.add(r["entity2"])
        else:
            # 全局模式：基于权重截断
            sorted_entities = sorted(entity_weights.keys(), key=lambda x: entity_weights[x], reverse=True)
            target_nodes = set(sorted_entities[:max_nodes])
            
            # 过滤不在 target_nodes 中的关系
            relations = [
                r for r in relations
                if r["entity1"] in target_nodes and r["entity2"] in target_nodes
            ]
        
        st.info(f"📈 找到 {len(relations)} 个实体关系 (显示 {len(target_nodes)} 个节点)")

        # 构建节点和边
        nodes = []
        edges = []
        entities_map = self._load_entities()
        events_map = self._load_events()
        
        # 添加实体节点
        for entity in target_nodes:
            is_focus = (entity == focus_entity)
            weight = entity_weights.get(entity, 0)
            ent_info = entities_map.get(entity, {}) if isinstance(entities_map, dict) else {}
            
            # 动态大小
            size = 30 if is_focus else 15 + min(int(weight / 2), 20)
            
            # 截断 Label
            label = GraphStyle._truncate_label(entity)
            
            nodes.append((entity, {
                "label": label,
                "_ml_title": GraphStyle.generate_entity_tooltip(entity, ent_info if isinstance(ent_info, dict) else {}),
                "color": GraphStyle.COLOR_ENTITY_FOCUS if is_focus else GraphStyle.COLOR_ENTITY_DEFAULT,
                "shape": GraphStyle.SHAPE_ENTITY,
                "size": size,
                "font": GraphStyle.get_font_config(size, is_focus)
            }))
        
        # 添加关系边
        for rel in relations:
            co_occurrence = rel["co_occurrence"]
            
            # 颜色映射
            if co_occurrence >= 20:
                color = GraphStyle.COLOR_RELATION_STRONG
            elif co_occurrence >= 10:
                color = GraphStyle.COLOR_RELATION_MEDIUM
            else:
                color = GraphStyle.COLOR_RELATION_WEAK
            
            width = min(co_occurrence / 10 + 1, 5)

            abs_list = []
            raw_events = rel.get("events")
            if isinstance(raw_events, str) and raw_events.strip():
                abs_list = [x.strip() for x in raw_events.split(",") if x.strip()]

            items_with_dt: List[Tuple[datetime, str, str]] = []
            for abs_key in abs_list:
                evt_data = events_map.get(abs_key, {}) if isinstance(events_map, dict) else {}
                if not isinstance(evt_data, dict):
                    evt_data = {}
                ts_str = evt_data.get("event_start_time") or evt_data.get("reported_at") or evt_data.get("first_seen") or ""
                dt = self._normalize_timestamp(ts_str)
                dt_sort = dt or datetime.max.replace(tzinfo=timezone.utc)
                if dt:
                    time_display = dt.strftime("%Y-%m-%d %H:%M")
                else:
                    time_display = ts_str or "Unknown"
                summary = evt_data.get("event_summary") or abs_key
                items_with_dt.append((dt_sort, time_display, summary))

            items_with_dt.sort(key=lambda x: x[0])
            tooltip_items: List[Tuple[str, str]] = [(t, s) for _, t, s in items_with_dt]
            edge_tooltip = GraphStyle.generate_relation_tooltip(rel["entity1"], rel["entity2"], tooltip_items)

            edges.append((rel["entity1"], rel["entity2"], {
                "_ml_title": edge_tooltip,
                "color": color,
                "width": width,
                # "label": str(co_occurrence)  # 可选：显示次数
            }))

        rel_items_by_pair: Dict[Tuple[str, str], List[Tuple[datetime, str, str, str]]] = defaultdict(list)
        if isinstance(events_map, dict):
            for abs_key, evt_data in events_map.items():
                if not isinstance(evt_data, dict):
                    continue
                rels = evt_data.get("relations")
                if not isinstance(rels, list):
                    continue
                ts_str = (
                    evt_data.get("event_start_time")
                    or evt_data.get("reported_at")
                    or evt_data.get("first_seen")
                    or ""
                )
                dt_evt = self._normalize_timestamp(ts_str)
                dt_sort = dt_evt or datetime.max.replace(tzinfo=timezone.utc)
                if dt_evt:
                    time_display = dt_evt.strftime("%Y-%m-%d %H:%M")
                else:
                    time_display = ts_str or "Unknown"
                summary = str(evt_data.get("event_summary") or abs_key)
                for r in rels:
                    if not isinstance(r, dict):
                        continue
                    s = str(r.get("subject") or "").strip()
                    o = str(r.get("object") or "").strip()
                    p = str(r.get("predicate") or "").strip()
                    if not s or not o or not p:
                        continue
                    if s not in target_nodes or o not in target_nodes:
                        continue
                    rel_items_by_pair[(s, o)].append((dt_sort, time_display, p, summary))

        for (s, o), items in rel_items_by_pair.items():
            items_sorted = sorted(items, key=lambda x: x[0])[:200]
            tooltip_items = [(t, p, summ) for _, t, p, summ in items_sorted]
            edge_tooltip = GraphStyle.generate_entity_relation_tooltip(s, o, tooltip_items)
            edges.append((s, o, {
                "_ml_title": edge_tooltip,
                "color": {"color": "#ff9800", "highlight": "#ff9800", "hover": "#ff9800", "opacity": 0.75},
                "width": 1,
                "dashes": True,
                "arrows": "to",
                "smooth": {"enabled": True, "type": "dynamic", "roundness": 0.25},
            }))
        
        # 渲染图谱
        self._render_pyvis(
            nodes=nodes,
            edges=edges,
            layout_config={
                "physics": {
                    "solver": "forceAtlas2Based",
                    "minVelocity": 0.75,
                    "stabilization": {"enabled": True, "iterations": 220, "updateInterval": 25, "fit": True},
                    "forceAtlas2Based": {
                        "gravitationalConstant": -140,
                        "centralGravity": 0.01,
                        "springLength": 170,
                        "springConstant": 0.08,
                        "damping": 0.6,
                        "avoidOverlap": 0.3,
                    },
                }
            },
            directed=False
        )
        
        # 显示图例
        with st.expander("🎨 图例说明"):
            st.markdown(f"""
            **边颜色**：
            - <span style='color:{GraphStyle.COLOR_RELATION_STRONG}'>●</span> 强关系（共现 ≥ 20 次）
            - <span style='color:{GraphStyle.COLOR_RELATION_MEDIUM}'>●</span> 中等关系（共现 10-20 次）
            - <span style='color:{GraphStyle.COLOR_RELATION_WEAK}'>●</span> 弱关系（共现 ≤ 10 次）
            - <span style='color:#ff9800'>●</span> 实体关系（三元组，虚线带箭头）
            
            **边宽度**：表示关系强度
            """, unsafe_allow_html=True)


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






