import streamlit as st
import json
import networkx as nx
from pathlib import Path
import streamlit.components.v1 as components
import sys
from datetime import datetime
import altair as alt
import pandas as pd

# 添加项目根目录到 path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.web import utils

st.set_page_config(page_title="Knowledge Graph - Market Lens", page_icon="🕸️", layout="wide")

# --- 数据加载 ---
data_root = Path(__file__).resolve().parent.parent / "data"
kg_file = data_root / "knowledge_graph.json"
kg_vis_file = data_root / "kg_visual.json"
kg_timeline_file = data_root / "kg_visual_timeline.json"
with st.spinner("Loading graph data..."):
    entities = utils.load_entities()
    events = utils.load_events()

    kg_data = {}
    if kg_file.exists():
        try:
            kg_data = json.loads(kg_file.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"知识图谱文件解析失败，已回退：{e}")
            kg_data = {}

    kg_vis_data = {}
    if kg_vis_file.exists():
        try:
            kg_vis_data = json.loads(kg_vis_file.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"快照 kg_visual.json 解析失败，已回退原始图谱：{e}")
            kg_vis_data = {}
    else:
        st.info("未找到 kg_visual.json，将使用原始知识图谱数据。")

    kg_timeline_data = []
    if kg_timeline_file.exists():
        try:
            kg_timeline_data = json.loads(kg_timeline_file.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"时间线快照 kg_visual_timeline.json 解析失败，已回退原始事件：{e}")
            kg_timeline_data = []
    else:
        st.info("未找到 kg_visual_timeline.json，将使用原始事件数据。")

# --- 侧边栏控制 ---
with st.sidebar:
    mode = st.radio("数据源", ["事件-实体映射 (EA)", "压缩图谱 (KG)"], index=0)
    all_entities = list(entities.keys()) if mode == "事件-实体映射 (EA)" else list((kg_data.get("entities") or {}).keys())
    placeholder_label = "(All / Top Nodes - EA)" if mode == "事件-实体映射 (EA)" else "(All / Top Nodes - KG)"
    search_query = st.selectbox(
        "Focus on Entity", 
        options=[placeholder_label] + sorted(all_entities),
        index=0,
        help="Select an entity to view its specific connections."
    )
    hop_depth = st.slider("Hop Depth (聚焦模式)", 1, 4, 1, help="从选定实体出发，最多拓展的边数（实体-事件-实体-...）。")
    
    # 2. 显示设置
    max_nodes = st.slider("Max Nodes", 10, 3000, 500, help="Limit total nodes for better performance")
    physics_enabled = st.checkbox("Enable Physics", value=True)
    auto_timeline = st.checkbox("显示聚焦实体时间线", value=True, help="在下方时间线视图中自动使用当前聚焦实体（KG/EA 均可）")
    
    # 时间线参数
    entity_opts = sorted(list(entities.keys()))
    default_tl = "(请选择)"
    if auto_timeline and search_query not in ["(All / Top Nodes - EA)", "(All / Top Nodes - KG)", "(All / Top Nodes)"]:
        default_tl = search_query
    # 时间线实体直接复用当前聚焦实体（非 All/Top），否则为未选择
    timeline_entity = search_query if search_query not in [placeholder_label, "(All / Top Nodes)"] else "(请选择)"
    limit_events = st.slider("最多显示事件数", 10, 500, 200, 10)
    
    st.divider()
    if mode == "事件-实体映射 (EA)":
        st.caption(f"Total Entities: {len(entities)}")
        st.caption(f"Total Events: {len(events)}")
    else:
        if kg_vis_data:
            st.caption(f"KG (vis) Nodes: {len(kg_vis_data.get('nodes') or [])}")
            st.caption(f"KG (vis) Edges: {len(kg_vis_data.get('edges') or [])}")
        else:
            st.caption(f"KG Entities: {len(kg_data.get('entities') or {})}")
            st.caption(f"KG Events: {len(kg_data.get('events') or {})}")

if mode == "事件-实体映射 (EA)":
    if not entities or not events:
        st.warning("Knowledge Graph is empty. Run the pipeline to populate data.")
        st.stop()
else:
    # KG 模式优先用可视化快照
    if kg_vis_data:
        pass
    elif not kg_data or not kg_data.get("entities") or not kg_data.get("events"):
        st.warning("Knowledge Graph (KG) is empty.")
        st.stop()

edge_list = []
event_ids = set()
if mode == "事件-实体映射 (EA)":
    event_ids = {f"EVT:{k}" for k in events.keys()}
    for evt_abstract, evt_data in events.items():
        evt_id = f"EVT:{evt_abstract}"  #以此区分
        evt_summary = evt_data.get('event_summary', evt_abstract)
        for ent in evt_data.get('entities', []):
            if ent in entities:
                edge_list.append((evt_id, ent, {"title": evt_summary}))
else:
    if kg_vis_data:
        vis_nodes = kg_vis_data.get("nodes", [])
        vis_edges = kg_vis_data.get("edges", [])
        for n in vis_nodes:
            if n.get("type") == "event":
                event_ids.add(n.get("id"))
        for e in vis_edges:
            u, v = e.get("from"), e.get("to")
            edge_list.append((u, v, {"title": e.get("title", "")}))
    else:
        kg_entities = kg_data.get("entities", {})
        kg_events = kg_data.get("events", {})
        kg_edges = kg_data.get("edges", [])
        event_ids = set(kg_events.keys())
        for e in kg_edges:
            u = e.get("from")
            v = e.get("to")
            if not u or not v:
                continue
            title = ""
            evt_key = v[4:] if isinstance(v, str) and v.startswith("EVT:") else v
            if evt_key in kg_events:
                title = kg_events[evt_key].get("event_summary", "") or kg_events[evt_key].get("abstract", "")
            edge_list.append((u, v, {"title": title}))

# --- 过滤逻辑 ---
target_nodes = set()
from collections import defaultdict, deque
adj = defaultdict(set)
for u, v, _ in edge_list:
    adj[u].add(v)
    adj[v].add(u)

# 节点类型判断
def is_event_node(node: str) -> bool:
    if isinstance(node, str) and node.startswith("EVT:"):
        return True
    return node in event_ids

if search_query != "(All / Top Nodes)" and search_query != "(All / Top Nodes - EA)" and search_query != "(All / Top Nodes - KG)":
    # 1. 聚焦模式：从选定实体出发，按 hop_depth 做 BFS（实体-事件交替）
    target_nodes.add(search_query)
    frontier = {search_query}
    for _ in range(hop_depth):
        next_frontier = set()
        for node in frontier:
            next_frontier |= adj.get(node, set())
        next_frontier -= target_nodes
        target_nodes |= next_frontier
        frontier = next_frontier
else:
    # 2. 全局模式：按度数（连接数）取 Top N 实体 + 相关事件
    # 简单起见，先统计实体出现频率
    # 也可以直接用 edge_list 构建临时图计算度
    temp_G = nx.Graph()
    temp_G.add_edges_from([(u, v) for u, v, _ in edge_list])
    
    # 计算度
    degrees = dict(temp_G.degree())
    # 排序
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
    target_nodes = set(top_nodes)

# --- 构建最终可视化图 ---
visual_G = nx.Graph()

count = 0
for u, v, attr in edge_list:
    if u in target_nodes and v in target_nodes:
        # 添加节点（如果未添加）
        if u not in visual_G:
            # 判断类型
            if is_event_node(u):
                label = u[4:20] + "..." if isinstance(u, str) and u.startswith("EVT:") else str(u)[:20] + "..."
                visual_G.add_node(u, label=label, title=str(u)[4:] if isinstance(u, str) and u.startswith("EVT:") else str(u), group='Event', color='#ff7f0e', size=15)
            else:
                visual_G.add_node(u, label=str(u), group='Entity', color='#1f77b4', size=25)
        
        if v not in visual_G:
            if is_event_node(v):
                label = v[4:20] + "..." if isinstance(v, str) and v.startswith("EVT:") else str(v)[:20] + "..."
                visual_G.add_node(v, label=label, title=str(v)[4:] if isinstance(v, str) and v.startswith("EVT:") else str(v), group='Event', color='#ff7f0e', size=15)
            else:
                visual_G.add_node(v, label=str(v), group='Entity', color='#1f77b4', size=25)
        
        visual_G.add_edge(u, v, title=attr.get("title"))
        count += 1
        
def parse_dt(val: str):
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except Exception:
        return None

# 预计算时间线数据
rows = []
co_counter = {}
if timeline_entity and timeline_entity != "(请选择)":
    if kg_timeline_data:
        for evt in kg_timeline_data:
            ents = evt.get("entities", [])
            if timeline_entity in ents:
                t = parse_dt(evt.get("time"))
                if not t:
                    continue
                co_entities = [e for e in ents if e != timeline_entity]
                for ce in co_entities:
                    co_counter[ce] = co_counter.get(ce, 0) + 1
                rows.append({
                    "abstract": evt.get("abstract", ""),
                    "event_summary": evt.get("event_summary", ""),
                    "time_dt": t,
                    "co_entities": ", ".join(co_entities[:5]),
                    "co_entities_raw": co_entities,
                })
    else:
        for abstract, evt in events.items():
            ents = evt.get("entities", [])
            if timeline_entity in ents:
                t = parse_dt(evt.get("first_seen") or evt.get("published_at"))
                if not t:
                    continue
                co_entities = [e for e in ents if e != timeline_entity]
                for ce in co_entities:
                    co_counter[ce] = co_counter.get(ce, 0) + 1
                rows.append({
                    "abstract": abstract,
                    "event_summary": evt.get("event_summary", "") or abstract,
                    "time_dt": t,
                    "co_entities": ", ".join(co_entities[:5]),
                    "co_entities_raw": co_entities,
                })
    rows = sorted(rows, key=lambda x: x["time_dt"])[:limit_events]


KG, EntityDetails, Timeline, TimelineDetails= st.tabs(["KG", "Entity Details", "Timeline", "Timeline Details"])

with KG:
    # --- PyVis 渲染 ---
    try:
        from pyvis.network import Network
        import tempfile
        
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(visual_G)
        
        if physics_enabled:
            net.force_atlas_2based()
        else:
            net.toggle_physics(False)
            
        # 保存并读取
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, "r", encoding="utf-8") as f:
                html_string = f.read()
                
        components.html(html_string, height=710, scrolling=False)
        
    except ImportError:
        st.error("PyVis not installed. Run `pip install pyvis` to view the graph.")
        st.info(f"Nodes: {visual_G.number_of_nodes()}, Edges: {visual_G.number_of_edges()}")

# --- 节点详情面板 ---
with EntityDetails:
    if search_query != "(All / Top Nodes)":
        st.divider()
        st.subheader(f"📘 Entity Details: {search_query}")
        
        ent_info = entities.get(search_query, {})
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Sources:**", ", ".join(ent_info.get("sources", [])))
            st.write("**First Seen:**", ent_info.get("first_seen", "N/A"))
        with c2:
            st.write("**Aliases/Forms:**", ", ".join(ent_info.get("original_forms", [])))
            
        st.write("**Related Events:**")
        # 查找关联事件摘要
        related_evts = []
        for evt_abstract, evt_data in events.items():
            if search_query in evt_data.get('entities', []):
                related_evts.append(evt_data.get('event_summary') or evt_abstract)
                
        for evt in related_evts[:10]:
            st.text(f"• {evt}")
        if len(related_evts) > 10:
            st.caption(f"... and {len(related_evts)-10} more.")

with Timeline:
    if timeline_entity and timeline_entity != "(请选择)" and rows:
        try:
            from pyvis.network import Network
            from pathlib import Path
            import tempfile
            import streamlit as st

            net = Network(
                height="750px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#333333",
                directed=True,
                notebook=False
            )

            # 关键设置：只关闭 barnesHut 物理中的重力，让节点还能自动散开，但不乱飞
            net.force_atlas_2based()  # 或者用 barnes_hut 但调小 gravity
            # 或者更推荐下面这套参数（最稳定最美观）：
            net.set_options("""
            {
            "physics": {
                "enabled": true,
                "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springStrength": 0.08,
                "damping": 0.8,
                "avoidOverlap": 1
                },
                "maxVelocity": 50,
                "minVelocity": 10,
                "solver": "forceAtlas2Based",
                "timestep": 0.5,
                "stabilization": {
                "enabled": true,
                "iterations": 200,
                "updateInterval": 25
                }
            },
            "nodes": {
                "font": {
                "size": 16,
                "face": "arial"
                }
            },
            "edges": {
                "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
                },
                "smooth": false,
                "color": "#999999"
            }
            }
            """)

            # 1. 先添加所有实体节点（不固定位置）
            all_entities = set()
            for r in rows:
                for ce in r.get("co_entities_raw", [])[:8]:  # 限制一下数量防爆炸
                    all_entities.add(ce)

            for ent in all_entities:
                net.add_node(
                    f"ent_{ent}",
                    label=ent,
                    color="#1f77b4",
                    size=20,
                    shape="dot",
                    font={"color": "white", "size": 14},
                    title=ent
                )

            # 2. 添加事件节点：固定 x/y
            for idx, r in enumerate(rows):
                x = idx * 230
                ys = [0,60,-60]
                y = ys[idx%3]
                
                size = 30 + len(r.get("co_entities_raw", [])) * 3
                label = r.get("event_summary", "")[:50] + "..." if len(r.get("event_summary", "")) > 50 else r.get("event_summary", "")

                net.add_node(
                    f"evt_{idx}",
                    label=label,
                    title=r.get("event_summary", ""),
                    x=x,
                    y=y,
                    fixed={"x": True, "y": True},   # 固定事件节点位置！
                    physics=False,                  # 这行很关键：事件节点不参与物理
                    color="#ff7f0e",
                    size=size,
                    shape="dot",
                    font={"size": 18, "color": "white"},
                    shadow=True
                )

                # 添加边：实体 → 事件（箭头指向事件）
                for ce in r.get("co_entities_raw", [])[:8]:
                    net.add_edge(f"ent_{ce}", f"evt_{idx}", color="#aaaaaa", width=1.5)

            # 可选：加一个隐藏的“时间主线”让事件之间也有连线（更清晰）
            for i in range(len(rows)-1):
                net.add_edge(f"evt_{i}", f"evt_{i+1}", color="#ff7f0e", width=3, dashes=True)

            # 保存并显示
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                html_string = Path(tmp.name).read_text(encoding="utf-8")

            st.components.v1.html(html_string, height=800, scrolling=True)

        except ImportError:
            st.warning("请先安装 pyvis：`pip install pyvis`")


with TimelineDetails:
    st.subheader("时间线详情")
    if timeline_entity and timeline_entity != "(请选择)":
        if rows:
            df_tl = pd.DataFrame(rows)
            chart = alt.Chart(df_tl).mark_line(point=True).encode(
                x="time_dt:T",
                y=alt.value(0),
                tooltip=["time_dt:T", "event_summary:N", "co_entities:N"]
            ).properties(height=120, width="container")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_tl[["time_dt", "event_summary", "co_entities"]], hide_index=True, use_container_width=True)
            
            if co_counter:
                top_co = sorted(co_counter.items(), key=lambda x: x[1], reverse=True)[:10]
                st.caption("Top 共现实体")
                st.table({"entity": [x[0] for x in top_co], "count": [x[1] for x in top_co]})
        else:
            st.info("该实体没有可展示的带时间事件。")
    else:
        st.info("请选择一个实体查看时间线。")





