import streamlit as st
import pandas as pd
import sys
import json
from pathlib import Path

# 添加项目根目录到 path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.web import utils

st.set_page_config(page_title="新闻智能体系统 - 数据查看器", page_icon="🕵️", layout="wide")

st.title("🕵️ Data Inspector")
st.caption("Explore extracted entities, events, and raw news data.")

# 统一清洗列，避免 Arrow 混合类型报错
def normalize_mixed(val):
    if val is None:
        return ""
    if isinstance(val, (list, dict)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    return str(val)

# --- Tab 布局 ---
tab_entities, tab_events, tab_news, tab_tmp = st.tabs(["🧠 Entities", "🔗 Events", "📰 Raw News", "🗃️ Extracted Snapshots"])

# 1. 实体浏览
with tab_entities:
    col_filter, col_stat = st.columns([3, 1])
    with col_filter:
        entity_search = st.text_input("🔍 Search Entities", placeholder="e.g. Bitcoin, SEC...")
    entities_data = utils.load_entities()
    
    if entities_data:
        df_ent = pd.DataFrame.from_dict(entities_data, orient='index')
        df_ent.reset_index(inplace=True)
        df_ent.rename(columns={'index': 'Entity Name'}, inplace=True)
        if 'sources' in df_ent.columns:
            df_ent['sources'] = df_ent['sources'].apply(normalize_mixed)
        
        # 搜索过滤
        if entity_search:
            df_ent = df_ent[df_ent['Entity Name'].str.contains(entity_search, case=False, na=False)]
            
        with col_stat:
            st.metric("Total Entities", len(df_ent))

        # 主表格
        st.dataframe(
            df_ent, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Entity Name": st.column_config.TextColumn("Entity Name", width="medium"),
                "count": st.column_config.NumberColumn("Mentions", format="%d"),
                "first_seen": st.column_config.DatetimeColumn("First Seen", format="YYYY-MM-DD HH:mm"),
                "sources": st.column_config.ListColumn("Sources")
            }
        )
    else:
        st.info("未找到实体数据。")

# 2. 事件浏览
with tab_events:
    col_evt_search, _ = st.columns([3, 1])
    with col_evt_search:
        event_search = st.text_input("🔍 Search Events", placeholder="e.g. ETF, Regulation...")
    events_data = utils.load_events()
    
    if events_data:
        df_evt = pd.DataFrame.from_dict(events_data, orient='index')
        df_evt['abstract'] = df_evt.index
        
        # 必要的列
        cols = ['abstract', 'event_summary', 'entities', 'sources', 'first_seen']
        existing_cols = [c for c in cols if c in df_evt.columns]
        df_evt = df_evt[existing_cols]
        if 'sources' in df_evt.columns:
            df_evt['sources'] = df_evt['sources'].apply(normalize_mixed)

        if event_search:
            mask = df_evt['abstract'].str.contains(event_search, case=False, na=False) | \
                   df_evt['event_summary'].str.contains(event_search, case=False, na=False)
            df_evt = df_evt[mask]

        st.dataframe(
            df_evt, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "abstract": st.column_config.TextColumn("Event Abstract", width="medium"),
                "event_summary": st.column_config.TextColumn("Summary", width="large"),
                "entities": st.column_config.ListColumn("Involved Entities"),
                "first_seen": st.column_config.DatetimeColumn("Detected At", format="YYYY-MM-DD")
            }
        )
    else:
        st.info("未找到事件数据。")

# 3. 原始新闻 (Feed View)
with tab_news:
    c_file, c_view = st.columns([1, 3])
    
    with c_file:
        st.subheader("📁 Select File")
        files = utils.get_raw_news_files()
        if files:
            # 按时间排序
            files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
            selected_file = st.radio("Available Files", files, format_func=lambda x: x.name, label_visibility="collapsed")
        else:
            st.warning("未找到文件。")
            selected_file = None

    with c_view:
        if selected_file:
            st.subheader(f"📄 Content: {selected_file.name}")
            news_items = utils.load_raw_news_file(selected_file)
            
            if news_items:
                # 分页
                items_per_page = 10
                total_pages = max(1, (len(news_items) + items_per_page - 1) // items_per_page)
                
                # 页码控制
                if "news_page" not in st.session_state: st.session_state.news_page = 1
                
                col_pg1, col_pg2, col_pg3 = st.columns([1, 2, 1])
                with col_pg1:
                    if st.button("Previous", disabled=st.session_state.news_page <= 1):
                        st.session_state.news_page -= 1
                        st.rerun()
                with col_pg2:
                    st.write(f"Page {st.session_state.news_page} of {total_pages} (Total: {len(news_items)})")
                with col_pg3:
                    if st.button("Next", disabled=st.session_state.news_page >= total_pages):
                        st.session_state.news_page += 1
                        st.rerun()
                
                # 显示当前页数据
                start_idx = (st.session_state.news_page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_items = news_items[start_idx:end_idx]
                
                for item in page_items:
                    title = item.get("title", "No Title")
                    date = item.get("datetime") or item.get("formatted_time") or "Unknown Date"
                    source = item.get("source", "Unknown Source")
                    content = item.get("content", "")
                    
                    with st.expander(f"**{title}** | {source} | {date}"):
                        st.markdown(f"**Content:**\n{content}")
                        st.json(item, expanded=False)
            else:
                st.info("文件为空。")

# 4. 提取结果快照（只读 + 删除）
with tab_tmp:
    st.subheader("🗃️ Extracted Events Snapshots (tmp)")
    tmp_dir = ROOT_DIR / "data" / "tmp"
    files = sorted(tmp_dir.glob("extracted_events_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not files:
        st.info("未找到提取的快照文件。")
    else:
        data = []
        for f in files:
            try:
                count = sum(1 for _ in f.open("r", encoding="utf-8"))
            except Exception:
                count = 0
            data.append({
                "file": f.name,
                "rows": count,
                "path": str(f)
            })
        df_snap = pd.DataFrame(data)
        st.dataframe(df_snap, hide_index=True, use_container_width=True)

        selected = st.selectbox("选择要删除的文件（仅删除 tmp 快照）", [""] + [f.name for f in files])
        if selected:
            if st.button("🗑️ 删除所选快照", type="primary"):
                try:
                    target = tmp_dir / selected
                    if target.exists():
                        target.unlink()
                        st.success(f"已删除 {selected}")
                        st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
        
        st.divider()
        preview_file = st.selectbox("选择要预览的快照文件", [""] + [f.name for f in files], index=0)
        if preview_file:
            target = tmp_dir / preview_file
            try:
                rows = []
                with open(target, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if idx >= 50:
                            break
                        try:
                            obj = json.loads(line)
                            rows.append({
                                "abstract": obj.get("abstract") or obj.get("event_summary") or "",
                                "event_summary": obj.get("event_summary", ""),
                                "entities": normalize_mixed(obj.get("entities")),
                                "source": obj.get("source", ""),
                                "published_at": obj.get("published_at", ""),
                                "news_id": obj.get("news_id", ""),
                            })
                        except Exception:
                            continue
                if rows:
                    df_preview = pd.DataFrame(rows)
                    st.write(f"预览 {preview_file} （最多 50 行）")
                    st.dataframe(df_preview, hide_index=True, use_container_width=True)
                else:
                    st.info("文件为空或无法解析可展示字段。")
            except Exception as e:
                st.error(f"预览失败: {e}")