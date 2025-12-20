"""
主流程页面 - 简化版

核心功能：一键运行增量更新流程
"""
from __future__ import annotations

import streamlit as st
from datetime import datetime, timezone, timedelta

from src.web import utils
from src.web.config import DATA_DIR
from src.web.components.task_monitor import render_task_monitor
from src.web.services.pipeline_runner import get_global_pipeline_runner, append_history
from src.web.framework.user_context import can_write, get_user_context, render_user_context_controls


def render() -> None:
    render_user_context_controls()
    
    # 获取任务管理器
    task_manager = get_global_pipeline_runner()
    
    st.info("📰 新闻处理流程 ：一键运行：抓取新闻 → 提取实体/事件 → 更新知识图谱")
    if_thirty_days = st.checkbox("三十天",value=False)
    
    # --- 任务状态监控 ---
    render_task_monitor(task_manager)
    
    st.divider()
    
    # --- 数据统计 ---
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("加载数据..."):
        entities = utils.load_entities() or {}
        events = utils.load_events() or {}
        news_files = utils.get_raw_news_files()
    
    with col1:
        st.metric("📰 新闻文件", len(news_files))
    with col2:
        st.metric("🧠 实体数量", len(entities))
    with col3:
        st.metric("🔗 事件数量", len(events))
    
    st.divider()
    
    # --- 一键运行 ---
    def execute_pipeline(pipeline_def):
        """提交任务到后台管理器"""
        if not can_write():
            st.error("当前角色为 viewer：禁止启动流水线。")
            return
        if task_manager.is_running:
            st.warning("⚠️ 已有任务正在运行。请等待完成后再试。")
            return

        # 清除缓存
        try:
            st.cache_data.clear()
        except Exception:
            pass
        
        # 记录基线
        st.session_state["_run_baseline"] = {
            "entities": list(entities.keys()),
            "events": list(events.keys()),
        }

        history_idx = append_history(pipeline_def)
        run_id = ""
        try:
            run_id = st.session_state.pipeline_history[history_idx].get("run_id") or ""
        except Exception:
            run_id = ""
        
        project_id = get_user_context().project_id
        success = task_manager.start(pipeline_def, history_idx=history_idx, run_id=run_id, project_id=project_id)
        
        if success:
            st.toast("🚀 任务已启动！")
            st.rerun()
        else:
            try:
                if "pipeline_history" in st.session_state and 0 <= history_idx < len(st.session_state.pipeline_history):
                    st.session_state.pipeline_history.pop(history_idx)
            except Exception:
                pass
            st.error("启动任务失败。")

    # 构建默认的增量更新 Pipeline
    now_utc = datetime.now(timezone.utc)
    days = 30 if if_thirty_days else 1
    
    from_dt = (now_utc - timedelta(days=days)).date().isoformat()
    to_dt = now_utc.date().isoformat()
    from_val = f"{from_dt}T00:00:00.000Z"
    to_val = f"{to_dt}T23:59:59.999Z"
    # 获取可用的新闻源
    selected_sources = []
    df_sources = st.session_state.get("ingestion_apis")
    if df_sources is not None and not getattr(df_sources, "empty", True):
        selected_sources = df_sources[df_sources["enabled"] == True]["name"].tolist()
    
    # 如果没有配置，使用默认源
    if not selected_sources:
        st.session_state.ingestion_apis = utils.get_default_api_sources_df()
        df_sources = st.session_state.ingestion_apis
        selected_sources = df_sources[df_sources["enabled"] == True]["name"].tolist()

    # 显示数据源信息
    st.info(f"📡 数据源: {', '.join(selected_sources[:3])}{'...' if len(selected_sources) > 3 else ''} ({len(selected_sources)} 个)")
    
    # 运行按钮
    run_disabled = task_manager.is_running or (not selected_sources)

    if st.button("🚀 开始运行", type="primary", use_container_width=True, disabled=run_disabled):
        pipeline_def = {
            "name": "Incremental Update",
            "steps": [
                {
                    "id": "fetch_news",
                    "tool": "fetch_news_stream",
                    "inputs": {
                        "limit": 10,
                        "sources": selected_sources,
                        "from_": from_val,
                        "to": to_val,
                        "daily_incremental": True,  # 启用按天递增请求
                    },
                    "output": "raw_news_data",
                },
                {
                    "id": "process_news",
                    "tool": "batch_process_news",
                    "inputs": {"news_list": "$raw_news_data"},
                    "output": "extracted_events",
                },
                {
                    "id": "update_graph",
                    "tool": "append_only_update_graph",
                    "inputs": {
                        "events_list": "$extracted_events",
                        "allow_append_original_forms": True,
                    },
                    "output": "kg_update_result",
                },
                {
                    "id": "refresh_kg",
                    "tool": "refresh_knowledge_graph",
                    "inputs": {},
                    "output": "kg_refresh",
                },
                {
                    "id": "report",
                    "tool": "generate_markdown_report",
                    "inputs": {
                        "events_list": "$extracted_events",
                        "title": "Incremental Update Report",
                    },
                    "output": "final_report_md",
                },
            ],
        }
        execute_pipeline(pipeline_def)
    
    if task_manager.is_running:
        st.caption("⏳ 任务运行中，请等待...")
    
    st.divider()
    
    # --- 运行结果展示 ---
    st.subheader("📋 运行结果")
    
    # 显示最终报告
    if task_manager.final_report:
        with st.expander("📄 生成的报告", expanded=True):
            st.markdown(task_manager.final_report)
    
    # 显示运行日志
    if task_manager.logs:
        with st.expander(f"📝 运行日志 ({len(task_manager.logs)} 条)", expanded=False):
            log_text = "\n".join(task_manager.logs[-50:])  # 最近50条
            st.code(log_text, language="text")
    
    # 显示输出数据
    if task_manager.last_outputs:
        with st.expander("📊 输出数据", expanded=False):
            for key, value in task_manager.last_outputs.items():
                st.write(f"**{key}**:")
                if isinstance(value, list):
                    st.write(f"  列表，{len(value)} 项")
                elif isinstance(value, dict):
                    st.json(value)
                else:
                    st.write(value)
