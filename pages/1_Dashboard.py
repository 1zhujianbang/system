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

st.set_page_config(page_title="新闻智能体系统 - 系统概览", page_icon="📊", layout="wide")

# 应用现代化样式
from src.web.styles import load_openai_style, create_modern_card, create_feature_grid, create_status_indicator
load_openai_style()

st.title("📰 新闻智能体系统概览")
st.markdown("### 实时监控系统状态、数据采集和知识图谱增长")

# 欢迎区域
welcome_col, status_col = st.columns([2, 1])

with welcome_col:
    create_modern_card(
        "欢迎使用",
        """
        <p>新闻智能体系统基于大语言模型和知识图谱技术，</p>
        <p>为您提供智能的新闻处理、实体提取和关系挖掘服务。</p>
        <br>
        <p><strong>🚀 核心功能：</strong></p>
        <ul>
            <li>📰 多源新闻采集</li>
            <li>🧠 智能实体提取</li>
            <li>🔗 知识图谱构建</li>
            <li>📊 实时可视化分析</li>
        </ul>
        """,
        "🎯"
    )

with status_col:
    st.markdown("### 系统状态")
    create_status_indicator("online", "数据处理服务")
    create_status_indicator("online", "API连接服务")
    create_status_indicator("online", "知识图谱引擎")
    create_status_indicator("online", "监控告警服务")

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
st.markdown("### 📊 核心指标")

# 使用响应式网格布局
metric_cols = st.columns(4)

with metric_cols[0]:
    st.metric(
        "📰 新闻文件",
        f"{news_count}",
        delta=f"+{len([f for f in raw_news_files if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days < 1])} 今日",
        help="存储的原始新闻文件总数"
    )

with metric_cols[1]:
    st.metric(
        "🧠 实体数量",
        f"{entity_count}",
        delta=f"{len([e for e in entities.values() if isinstance(e, dict) and (datetime.now().date() - datetime.fromisoformat(e.get('first_seen', '2024-01-01')).date()).days < 7])} 新增",
        help="知识图谱中的唯一实体节点"
    )

with metric_cols[2]:
    st.metric(
        "🔗 事件数量",
        f"{event_count}",
        delta=f"{len([e for e in events.values() if isinstance(e, dict) and (datetime.now().date() - datetime.fromisoformat(e.get('first_seen', '2024-01-01')).date()).days < 7])} 新增",
        help="提取的事件关系总数"
    )

with metric_cols[3]:
    st.metric(
        "🕒 最后活动",
        last_update if last_update != "N/A" else "从未",
        help="系统最后一次活动时间"
    )

st.markdown("---")

# --- 数据洞察面板 ---
st.markdown("### 🔍 数据洞察")

# 创建响应式图表布局
chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    with st.container(border=True):
        st.subheader("🏆 热门实体排名")
        if not top_entities_df.empty:
            # 美化图表样式 - 使用单一颜色主题
            st.bar_chart(
                top_entities_df.set_index("Entity")["Mentions"],
                color="#667eea",  # 使用单一主题色
                use_container_width=True
            )

            # 显示Top 3详情
            st.markdown("**🏅 排名详情:**")
            for i, (_, row) in enumerate(top_entities_df.head(3).iterrows()):
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
                st.markdown(f"{medal} **{row['Entity']}** - {row['Mentions']} 次提及")
        else:
            st.info("暂无实体数据可供可视化")

with chart_col2:
    with st.container(border=True):
        st.subheader("📡 数据来源分布")
        if entities and not top_entities_df.empty:
            # 计算数据源分布
            source_counts = df_all["Source"].value_counts().head(6)

            # 使用Streamlit原生图表替代Plotly（避免NumPy兼容性问题）
            import pandas as pd
            pie_data = pd.DataFrame({
                'Source': source_counts.index,
                'Count': source_counts.values
            })

            # 显示条形图作为饼图的替代
            st.bar_chart(
                pie_data.set_index('Source')['Count'],
                color='#667eea',  # 使用单一主题色
                use_container_width=True
            )

            # 显示来源统计
            st.markdown("**📊 详细统计:**")
            for source, count in source_counts.items():
                percentage = (count / len(df_all)) * 100
                st.markdown(f"• **{source}**: {count} 条 ({percentage:.1f}%)")
        else:
            st.info("暂无数据来源信息")

st.markdown("---")

# --- 系统活动日志 & 快捷入口 ---
c_log, c_action = st.columns([2, 1])

with c_log:
    st.subheader("📋 系统活动日志")

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
                for line in reversed(lines[-20:]):  # 只显示最近20条
                    if "ERROR" in line:
                        icon = "🔴"
                        level = "ERROR"
                    elif "WARNING" in line:
                        icon = "🟡"
                        level = "WARNING"
                    elif "SUCCESS" in line or "✅" in line:
                        icon = "🟢"
                        level = "SUCCESS"
                    else:
                        icon = "🔵"
                        level = "INFO"

                    # 格式化时间和内容
                    timestamp = line.split('[')[1].split(']')[0] if '[' in line else ""
                    message = line.split(']', 2)[-1].strip() if ']' in line else line.strip()
                    log_content.append(f"{icon} **{level}** {timestamp} {message}")
    except Exception as e:
        log_content = [f"❌ 读取日志失败: {e}"]

    # 使用现代化的滚动容器
    if log_content:
        st.markdown("""
            <div style="max-height: 300px; overflow-y: auto; background-color: #f8fafc; border-radius: 8px; padding: 1rem; border: 1px solid #e5e5e5;">
        """, unsafe_allow_html=True)
        for line in log_content:
            st.markdown(line)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("暂无系统日志")

with c_action:
    st.subheader("🚀 快捷操作")
    with st.container(border=True):
        st.markdown("**🔧 工作流管理**")

        # 美化按钮样式
        button_style = """
            <style>
            .quick-action-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin: 0.25rem 0;
                width: 100%;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
            }
            .quick-action-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)

        if st.button("🔧 构建Pipeline", use_container_width=True, key="dashboard_pipeline_button"):
            st.switch_page("pages/2_Pipeline_Builder.py")

        if st.button("🕵️ 检查数据", use_container_width=True, key="dashboard_data_button"):
            st.switch_page("pages/3_Data_Inspector.py")

        if st.button("🕸️ 查看图谱", use_container_width=True, key="dashboard_graph_button"):
            st.switch_page("pages/4_Knowledge_Graph.py")

        if st.button("⚙️ 系统设置", use_container_width=True, key="dashboard_settings_button"):
            st.switch_page("pages/5_System_Settings.py")

        st.divider()
        create_status_indicator("online", "系统运行正常")
