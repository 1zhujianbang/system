import streamlit as st
import json
import networkx as nx
from pathlib import Path
import streamlit.components.v1 as components
import sys

# 添加项目根目录到 path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.web import utils

st.set_page_config(page_title="Knowledge Graph - Market Lens", page_icon="🕸️", layout="wide")

st.title("🕸️ Knowledge Graph Explorer")

# --- 数据加载 ---
with st.spinner("Loading graph data..."):
    entities = utils.load_entities()
    events = utils.load_events()

if not entities or not events:
    st.warning("Knowledge Graph is empty. Run the pipeline to populate data.")
    st.stop()

# --- 侧边栏控制 ---
with st.sidebar:
    st.header("Graph Controls")
    
    # 1. 搜索/聚焦
    all_entities = list(entities.keys())
    search_query = st.selectbox(
        "Focus on Entity", 
        options=["(All / Top Nodes)"] + sorted(all_entities),
        index=0,
        help="Select an entity to view its specific connections."
    )
    
    st.divider()
    
    # 2. 显示设置
    max_nodes = st.slider("Max Nodes", 10, 3000, 500, help="Limit total nodes for better performance")
    physics_enabled = st.checkbox("Enable Physics", value=True)
    
    st.divider()
    st.caption(f"Total Entities: {len(entities)}")
    st.caption(f"Total Events: {len(events)}")

# --- 图构建逻辑 ---
G = nx.Graph()

# 预构建完整图（或至少是包含关系的图）
# 为了性能，我们在构建 NX 图时暂时只添加关系，不添加完整属性
edge_list = []
for evt_abstract, evt_data in events.items():
    # Event 节点
    evt_id = f"EVT:{evt_abstract}"  #以此区分
    # 限制 Event 节点属性
    evt_summary = evt_data.get('event_summary', evt_abstract)
    
    # 添加边 (Event -> Entity)
    for ent in evt_data.get('entities', []):
        if ent in entities:
            edge_list.append((evt_id, ent, {"title": evt_summary}))

# --- 过滤逻辑 ---
target_nodes = set()

if search_query != "(All / Top Nodes)":
    # 1. 聚焦模式：找到目标实体及其邻居
    target_nodes.add(search_query)
    
    # 找到所有涉及该实体的事件
    related_events = []
    for u, v, attr in edge_list:
        if u == search_query or v == search_query:
            neighbor = v if u == search_query else u
            target_nodes.add(neighbor)
            # 如果 neighbor 是事件，还得把事件的其他实体加进来（可选，2-hop）
            # 这里暂时只做 1-hop: Entity -> Event
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
            if u.startswith("EVT:"):
                label = u[4:20] + "..." 
                visual_G.add_node(u, label=label, title=u[4:], group='Event', color='#ff7f0e', size=15)
            else:
                visual_G.add_node(u, label=u, group='Entity', color='#1f77b4', size=25)
        
        if v not in visual_G:
            if v.startswith("EVT:"):
                label = v[4:20] + "..."
                visual_G.add_node(v, label=label, title=v[4:], group='Event', color='#ff7f0e', size=15)
            else:
                visual_G.add_node(v, label=v, group='Entity', color='#1f77b4', size=25)
        
        visual_G.add_edge(u, v, title=attr.get("title"))
        count += 1
        
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
