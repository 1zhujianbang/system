import streamlit as st
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import json

# 添加项目根目录到 path (用于导入 src 模块)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.web.config import DATA_DIR, LOGS_DIR
from src.web import utils

KG_FILE = DATA_DIR / "knowledge_graph.json"

@st.cache_data(ttl=60)
def load_kg_counts():
    """
    从 knowledge_graph.json 统计实体出现次数（基于 edges 的 from 字段）。
    """
    counts = {}
    if KG_FILE.exists():
        try:
            with open(KG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            edges = data.get("edges", [])
            for edge in edges:
                src = edge.get("from")
                if src:
                    counts[src] = counts.get(src, 0) + 1
            # 如果 edges 为空，尝试从 entities 节点补充一次计数
            if not counts and isinstance(data.get("entities"), dict):
                for name in data["entities"].keys():
                    counts[name] = 1
        except Exception:
            pass
    return counts

st.set_page_config(page_title="Dashboard - Market Lens", page_icon="📊", layout="wide")

# CSS 样式优化
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard")
st.markdown("Overview of your system status, data collection, and knowledge graph growth.")

# --- 数据加载 ---
with st.spinner("Loading metrics..."):
    # 1. 基础统计
    raw_news_files = utils.get_raw_news_files()
    news_count = len(raw_news_files)
    
    entities = utils.load_entities()
    entity_count = len(entities)
    kg_counts = load_kg_counts()
    
    events = utils.load_events()
    event_count = len(events)
    
    # 2. 计算最近更新时间
    last_update = "N/A"
    if LOGS_DIR.exists():
        log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            last_update = datetime.fromtimestamp(log_files[0].stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    # 3. 准备图表数据
    # 实体 Top 10
    top_entities_df = pd.DataFrame()
    if entities:
        # 组装实体数据，清理源字段与计数字段，避免前端渲染对象/NaN
        data = []
        for name, info in entities.items():
            name = str(name)
            if isinstance(info, dict):
                count = info.get("count", kg_counts.get(name, 1))
                src_raw = info.get("sources", [])
            else:
                count = kg_counts.get(name, 1)
                src_raw = []

            # count 数值化
            try:
                count = int(count)
            except Exception:
                count = 0

            # 源字段转字符串
            source = "unknown"
            if src_raw:
                first = src_raw[0]
                if isinstance(first, dict):
                    # 优先 name，其次 id/url
                    source = first.get("name") or first.get("id") or first.get("url") or "unknown"
                else:
                    source = str(first)

            data.append({"Entity": name, "Mentions": count, "Source": source})
        
        df_all = pd.DataFrame(data)
        if not df_all.empty:
            df_all["Mentions"] = pd.to_numeric(df_all["Mentions"], errors="coerce").fillna(0).astype(int)
            df_all["Entity"] = df_all["Entity"].astype(str)
            # 过滤掉全 0 的情况，避免图表 Infinity 警告
            if df_all["Mentions"].sum() > 0:
                top_entities_df = df_all.sort_values("Mentions", ascending=False).head(10)
            else:
                top_entities_df = pd.DataFrame()

# --- 核心指标卡片 ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📰 Raw News Files", news_count, delta="Total Collected", help="Number of raw news files in storage")
with col2:
    st.metric("🧠 Entities Tracked", entity_count, delta="Knowledge Nodes", help="Total unique entities in Knowledge Graph")
with col3:
    st.metric("🔗 Events Extracted", event_count, delta="Relationships", help="Total unique events extracted")
with col4:
    st.metric("🕒 Last Activity", last_update, help="Time of last system log update")

st.markdown("---")

# --- 图表区域 ---
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.subheader("🏆 Top Mentioned Entities")
    if not top_entities_df.empty:
        st.bar_chart(top_entities_df.set_index("Entity")["Mentions"], color="#4e79a7")
    else:
        st.info("No entity data available for visualization.")

with col_chart2:
    st.subheader("📡 Data Sources")
    if entities and not top_entities_df.empty:
        # 简单的源分布
        source_counts = df_all["Source"].value_counts().head(5)
        st.write("Distribution of entities by primary source:")
        st.dataframe(source_counts, use_container_width=True)
    else:
        st.info("No source data available.")

st.markdown("---")

# --- 系统活动日志 & 快捷入口 ---
c_log, c_action = st.columns([2, 1])

with c_log:
    st.subheader("📋 Recent System Logs")
    
    log_content = []
    try:
        log_target = LOGS_DIR / "agent1.log"
        if not log_target.exists() and LOGS_DIR.exists():
             # Fallback to latest
             log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
             if log_files: log_target = log_files[0]
             
        if log_target.exists():
            with open(log_target, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                # 反转显示，最新的在最上面
                for line in reversed(lines[-50:]):
                    if "ERROR" in line:
                        icon = "🔴"
                    elif "WARNING" in line:
                        icon = "qh"
                    elif "SUCCESS" in line or "✅" in line:
                        icon = "🟢"
                    else:
                        icon = "ℹ️"
                    log_content.append(f"{icon} {line.strip()}")
    except Exception as e:
        log_content = [f"Error reading logs: {e}"]

    # 使用 scrollable container
    with st.container(height=300):
        if log_content:
            for line in log_content:
                st.text(line)
        else:
            st.text("No logs found.")

with c_action:
    st.subheader("🚀 Quick Actions")
    with st.container(border=True):
        st.markdown("**Pipeline Operations**")
        if st.button("Go to Pipeline Builder", use_container_width=True):
            st.switch_page("pages/2_Pipeline_Builder.py")
            
        st.markdown("**Data Management**")
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            if st.button("Inspect Data", use_container_width=True):
                st.switch_page("pages/3_Data_Inspector.py")
        with c_a2:
            if st.button("View Graph", use_container_width=True):
                st.switch_page("pages/4_Knowledge_Graph.py")
                
        st.divider()
        st.caption("System Status: 🟢 Online")
