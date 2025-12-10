import streamlit as st
import yaml
import asyncio
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import threading
from datetime import timezone
from dotenv import dotenv_values
from typing import Dict, Any  

# 添加项目根目录到 path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
ENV_PATH = ROOT_DIR / "config" / ".env.local"

from src.core.registry import FunctionRegistry
from src.core.engine import PipelineEngine
from src.core.context import PipelineContext
from src.data.api_client import DataAPIPool, get_apis_config
from src.web import utils
import src.functions.data_fetch
import src.functions.extraction
import src.functions.graph_ops
import src.functions.reporting

st.set_page_config(page_title="Pipeline Builder - Market Lens", page_icon="⛓️", layout="wide")

# --- 全局任务管理器 ---

class GlobalTaskManager:
    def __init__(self):
        self.is_running = False
        self.logs = []
        self.status_info = {"label": "Idle", "state": "idle", "expanded": False}
        self.current_step_idx = 0
        self.total_steps = 0
        self.final_report = None
        self._lock = threading.Lock()
        
    def start_task(self, pipeline_def):
        if self.is_running:
            return False
            
        self.is_running = True
        self.logs = []
        self.status_info = {"label": "Starting...", "state": "running", "expanded": True}
        self.final_report = None
        self.current_step_idx = 0
        steps = pipeline_def.get("steps", [])
        self.total_steps = len(steps)
        
        def _worker():
            asyncio.run(self._async_runner(pipeline_def))
            
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return True
        
    async def _async_runner(self, pipeline_def):
        def log_callback(entry):
            with self._lock:
                ts = entry['timestamp'].split('T')[1][:8]
                msg = f"[{ts}] [{entry['level']}] {entry['message']}"
                self.logs.append(msg)
                # 保留最近 1000 条日志
                if len(self.logs) > 1000:
                    self.logs.pop(0)

        context = PipelineContext(log_callback=log_callback)
        engine = PipelineEngine(context)
        
        steps = pipeline_def.get("steps", [])
        
        try:
            for i, step in enumerate(steps):
                step_id = step.get('id')
                self.current_step_idx = i + 1
                
                # 更新状态
                with self._lock:
                    self.status_info = {
                        "label": f"Executing Step {self.current_step_idx}/{self.total_steps}: **{step_id}**", 
                        "state": "running", 
                        "expanded": True
                    }
                
                # 执行任务
                await engine.run_task(step)
                
            # 完成
            with self._lock:
                self.status_info = {"label": "✅ Pipeline Execution Completed!", "state": "complete", "expanded": False}
                self.final_report = context.get("final_report_md")
                
        except Exception as e:
            with self._lock:
                self.status_info = {"label": f"❌ Execution Failed: {str(e)}", "state": "error", "expanded": True}
                self.logs.append(f"[System] Error: {str(e)}")
        finally:
            self.is_running = False

@st.cache_resource
def get_task_manager():
    return GlobalTaskManager()

task_manager = get_task_manager()

# --- UI 组件 ---

st.title("Task Center & Pipeline Builder")

# 任务监控区 (始终显示在顶部)
def render_task_monitor():
    if task_manager.is_running or task_manager.status_info["state"] != "idle":
        with st.container(border=True):
            col_status, col_ctrl = st.columns([4, 1])
            
            with col_status:
                # 使用 st.status 展示状态
                state = task_manager.status_info["state"]
                label = task_manager.status_info["label"]
                expanded = task_manager.status_info["expanded"]
                
                status_container = st.status(label, expanded=expanded, state=state)
                
                # 显示最后几条日志
                with status_container:
                    st.write("Recent Logs:")
                    with task_manager._lock:
                        recent_logs = task_manager.logs[-10:]
                    st.code("\n".join(recent_logs) if recent_logs else "Initializing...", language="text")
            
            with col_ctrl:
                if task_manager.is_running:
                    st.caption("Running in background...")
                    if st.button("🔄 Refresh View"):
                        st.rerun()
                    # 自动刷新逻辑 (实验性)
                    time.sleep(2)
                    st.rerun()
                else:
                    if st.button("Clear Status"):
                        task_manager.status_info["state"] = "idle"
                        st.rerun()

        # 结果展示
        if not task_manager.is_running and task_manager.final_report:
            with st.expander("📄 Final Report Result", expanded=True):
                st.markdown(task_manager.final_report)
                st.download_button(
                    "Download Report", 
                    task_manager.final_report, 
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                )

render_task_monitor()

# 初始化 Session State
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = []

# 初始化 API 配置 (仅运行一次)
if "ingestion_apis" not in st.session_state:
    st.session_state.ingestion_apis = utils.get_default_api_sources_df()

if "expansion_tasks" not in st.session_state:
    # 初始化 expansion_tasks (空) —— 使用当前支持的字段
    st.session_state.expansion_tasks = pd.DataFrame(
        columns=["enabled", "keyword", "limit", "category", "from", "to", "sortby"]
    ).astype({
        "enabled": "bool",
        "keyword": "str",
        "limit": "int",
        "category": "str",
        "from": "str",
        "to": "str",
        "sortby": "str",
    })

# 初始化 .env.local Key-Value
def load_env_df():
    kv = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    rows = [{"key": k, "value": v or ""} for k, v in kv.items()]
    return pd.DataFrame(rows, columns=["key", "value"])

if "env_kv" not in st.session_state:
    st.session_state.env_kv = load_env_df()

# --- 辅助函数 ---

def execute_pipeline(pipeline_def):
    """提交任务到后台管理器"""
    if task_manager.is_running:
        st.warning("⚠️ A task is already running. Please wait for it to finish.")
        return

    success = task_manager.start_task(pipeline_def)
    if success:
        st.toast("🚀 Task started in background!")
        st.rerun()
    else:
        st.error("Failed to start task.")

def render_input_field(step_idx, p_name, p_info, current_inputs, step):
    """
    智能渲染输入组件
    """
    p_type = p_info.get('type', 'Any')
    p_required = p_info.get('required', False)
    default_val = p_info.get('default')
    
    label = f"{p_name}{' *' if p_required else ''}"
    help_text = f"Type: {p_type}" + (f", Default: {default_val}" if default_val else "")
    key = f"in_{step_idx}_{p_name}"
    
    current_val = current_inputs.get(p_name, default_val)
    is_ref = isinstance(current_val, str) and current_val.startswith("$")
    
    if is_ref:
         new_val = st.text_input(label + " (Variable)", value=current_val, key=key, help=help_text)
         step["inputs"][p_name] = new_val
         return

    if "bool" in p_type.lower():
        val = st.checkbox(label, value=bool(current_val) if current_val is not None else False, key=key, help=help_text)
        step["inputs"][p_name] = val
        
    elif "int" in p_type.lower():
        val = st.number_input(label, value=int(current_val) if current_val is not None else 0, step=1, key=key, help=help_text)
        step["inputs"][p_name] = int(val)
        
    elif "list" in p_type.lower() or "dict" in p_type.lower():
        val_str = str(current_val) if current_val is not None else "[]"
        new_val_str = st.text_area(label + " (JSON/List)", value=val_str, height=100, key=key, help="Enter valid JSON or $variable")
        if new_val_str.startswith("$"):
            step["inputs"][p_name] = new_val_str
        else:
            try:
                import ast
                if new_val_str.strip():
                    parsed = ast.literal_eval(new_val_str)
                    step["inputs"][p_name] = parsed
            except:
                step["inputs"][p_name] = new_val_str
    else:
        val = st.text_input(label, value=str(current_val) if current_val is not None else "", key=key, help=help_text)
        if val:
            step["inputs"][p_name] = val

# --- 场景化模块渲染 ---
def render_configuration_tab():
    st.header("📚 Configuration")
    source_config, source_api_pool= st.tabs(["Source Configuration","Source API Pool"])
    with source_config:
        st.subheader("Source Configuration")
        st.write("Source Configuration")
        edited_df = st.data_editor(
            st.session_state.ingestion_apis,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "name": st.column_config.TextColumn("Source Name", required=True),
                "type": st.column_config.SelectboxColumn("API Type", options=["gnews"], required=True),
                "language": st.column_config.SelectboxColumn("Language", options=["ar", "zh", "nl", "en", "fr", "de", "el", "he", "hi", "id", "it", "ja", "ml", "mr", "no", "pt", "pa", "ro", "ru", "es", "sv", "ta", "te", "tr", "uk"]),
                "timeout": st.column_config.NumberColumn("Timeout (s)"),
                "country": st.column_config.SelectboxColumn("Country", options=["ar", "au", "br", "ca", "cn", "co", "eg", "fr", "de", "gr", "hk", "in", "id", "ie", "il", "it", "jp", "my", "mx", "nl", "no", "pk", "pe", "ph", "pt", "ro", "ru", "sg", "es", "se", "ch", "tw", "tr", "ua", "gb", "us"]),
            },
            key="ingestion_editor_main"
        )
        st.session_state.ingestion_apis = edited_df
        
        # 实时显示选中的源数量，并保存当前选择
        selected_apis = edited_df[edited_df["enabled"] == True]["name"].tolist()
        st.session_state["ingestion_selected_apis"] = selected_apis
        st.caption(f"✅ Selected Sources: {len(selected_apis)}")

        # GNews 可选参数配置
        gnews_params = st.session_state.get("gnews_params", {})
        with st.expander("GNews 可选参数", expanded=False):
            category = st.selectbox(
                "Category",
                ["", "general", "world", "business", "technology", "sports", "science", "health", "entertainment"],
                index=0,
                help="留空则不指定分类"
            )
            query = st.text_input("Query (关键词搜索，可空)")
            col_from, col_to = st.columns(2)
            min_date = datetime(2020, 1, 1).date()
            max_date = datetime.now().date()
            with col_from:
                d_from = st.date_input("From 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_from_date")
                t_from = st.time_input("From 时间", value=None, key="gnews_from_time")
            with col_to:
                d_to = st.date_input("To 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_to_date")
                t_to = st.time_input("To 时间", value=None, key="gnews_to_time")

            def combine(dt, tm):
                if dt is None:
                    return None
                tm = tm or datetime.min.time()
                # 统一使用 UTC 输出 ISO8601
                return datetime.combine(dt, tm, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

            from_iso = combine(d_from, t_from)
            to_iso = combine(d_to, t_to)
            nullable = st.text_input("Nullable", value=gnews_params.get("nullable", ""), help="如 description,content")
            truncate = st.text_input("Truncate", value=gnews_params.get("truncate", ""), help="如 content")
            sortby = st.selectbox("Sortby", ["", "publishedAt", "relevance"], index=0)
            in_fields = st.text_input("In fields", value=gnews_params.get("in_fields", ""), help="如 title,description")
            page = st.number_input("Page", min_value=1, value=gnews_params.get("page", 1), step=1)

            # 保存到 session_state
            st.session_state["gnews_params"] = {
                "category": category or None,
                "query": query or None,
                "from_": from_iso,
                "to": to_iso,
                "nullable": nullable or None,
                "truncate": truncate or None,
                "sortby": sortby or None,
                "in_fields": in_fields or None,
                "page": int(page) if page else None,
            }

    with source_api_pool:
        st.subheader("Source API Pool")
        st.caption("编辑并保存到 config/.env.local（覆盖写入，不保留注释/空行）。上方“保存该行/表格修改”仅更新内存，需在下方点击保存才写入文件。")
        edited_env = st.session_state.env_kv
        if st.checkbox("显示 Key/Value 表格编辑器", value=False, key="env_editor_toggle"):
            edited_env = st.data_editor(
                st.session_state.env_kv,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "key": st.column_config.TextColumn("Key", required=True),
                    "value": st.column_config.TextColumn("Value", required=False, help="可输入占位符，注意避免泄露敏感值")
                },
                key="env_editor"
            )
            st.session_state.env_kv = edited_env
        def try_parse_json(val: str):
            if not isinstance(val, str):
                return None, "非字符串"
            txt = val.strip()
            if not txt:
                return None, "空值"
            try:
                obj = json.loads(txt)
                return obj, ""
            except Exception as e:
                return None, str(e)

        for _, row in edited_env.iterrows():
            idx = row.name
            k = str(row.get("key", "")).strip()
            v = str(row.get("value", "")).strip()
            if not k:
                continue
            parsed, err = try_parse_json(v)
            with st.expander(f"{k}", expanded=False):
                if parsed is not None:
                    pretty_txt = json.dumps(parsed, ensure_ascii=False, indent=2)
                    new_txt = st.text_area(
                        "JSON 编辑（保存即写回 value）",
                        value=pretty_txt,
                        key=f"json_edit_{idx}",
                        height=200
                    )
                    if st.button("保存该行", key=f"save_json_{idx}", use_container_width=True):
                        try:
                            parsed_new = json.loads(new_txt)
                            # 写回表格缓存，保持紧凑存储
                            edited_env.at[idx, "value"] = json.dumps(parsed_new, ensure_ascii=False)
                            st.session_state.env_kv = edited_env
                            st.success("已更新该行的 value")
                        except Exception as e:
                            st.error(f"JSON 解析失败: {e}")
                    # 表格化展示（优先 list[dict] 或 dict -> DataFrame），否则用 json
                    def to_table(obj):
                        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
                            return pd.DataFrame(obj), "list"
                        if isinstance(obj, dict):
                            return pd.DataFrame([obj]), "dict"
                        return None, ""
                    df_preview, kind = to_table(parsed)
                    if df_preview is not None and not df_preview.empty:
                        edited_df = st.data_editor(
                            df_preview,
                            num_rows="dynamic",
                            use_container_width=True,
                            key=f"env_table_{idx}"
                        )
                        if st.button("保存表格修改", key=f"save_table_{idx}", use_container_width=True):
                            try:
                                if kind == "list":
                                    new_obj = edited_df.to_dict(orient="records")
                                else:
                                    new_obj = edited_df.to_dict(orient="records")[0] if not edited_df.empty else {}
                                edited_env.at[idx, "value"] = json.dumps(new_obj, ensure_ascii=False)
                                st.session_state.env_kv = edited_env
                                st.success("已根据表格修改更新 value")
                            except Exception as e:
                                st.error(f"写回 JSON 失败: {e}")
                    else:
                        st.json(parsed)
                else:
                    st.warning(f"无法解析为 JSON：{err}")

        if st.button("💾 保存到 .env.local", type="primary", use_container_width=True):
            lines = []
            for _, row in edited_env.iterrows():
                k = str(row.get("key", "")).strip()
                if not k:
                    continue
                v = str(row.get("value", "")).strip()
                lines.append(f"{k}={v}")
            ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
            ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
            st.success(f"已写入 {ENV_PATH.name} ，共 {len(lines)} 条记录")



def render_ingestion_tab():
    st.header("📥 Data Ingestion")
    st.caption("Fetch news from sources (Feed/Search) and extract events.")
    
    st.subheader("Processing Parameters")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
            st.markdown("##### 📥 Fetch Settings")
            news_limit = st.number_input("Limit (per source)", 1, 10, 5, 1, help="Max news items to fetch per source.")
            
    with col_p2:
            st.markdown("##### ⚙️ Pipeline Actions")
            auto_update_kg = st.checkbox("Auto Update Knowledge Graph", True, help="Automatically extract entities and update the graph.")
            enable_report = st.checkbox("Generate Summary Report", True, help="Create a markdown report after processing.")

    st.subheader("🚀 Ready to Start?")
    
    # 汇总配置
    current_df = st.session_state.ingestion_apis
    selected_sources = current_df[current_df["enabled"] == True]["name"].tolist()
    
    st.write("Summary:")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sources Selected", len(selected_sources))
    c2.metric("Max Items", news_limit)
    c3.metric("Auto-Update KG", "Yes" if auto_update_kg else "No")
    gnews_params = st.session_state.get("gnews_params", {})

    st.subheader("图谱更新模式")
    col_mode_ing, col_forms_ing = st.columns(2)
    with col_mode_ing:
        append_only_ing = st.checkbox("仅追加（不改旧数据）- Ingestion", value=True, help="不修改已有实体/事件，只新增不存在的记录")
    with col_forms_ing:
        allow_append_forms_ing = st.checkbox("追加旧实体的 original_forms - Ingestion", value=True, help="仅在仅追加模式下生效；关闭则完全不改旧实体字段")
    
    if not selected_sources:
        st.error("❌ No sources selected. Please go back to 'Data Sources' tab.")
        btn_disabled = True
    else:
        btn_disabled = False
        
    if st.button("Start Ingestion Task", type="primary", disabled=btn_disabled, use_container_width=True):
        pipeline_def = {
            "name": "Data Ingestion Task",
            "steps": [
                {
                    "id": "fetch_news",
                    "tool": "fetch_news_stream",
                    "inputs": {
                        "limit": news_limit, 
                        "sources": selected_sources,
                        # GNews 可选参数透传（仅当有值）
                        **{k: v for k, v in gnews_params.items() if v}
                    },
                    "output": "raw_news_data"
                },
                {
                    "id": "process_news",
                    "tool": "batch_process_news",
                    "inputs": {"news_list": "$raw_news_data"},
                    "output": "extracted_events"
                },
                {
                    "id": "save_events_tmp",
                    "tool": "save_extracted_events_tmp",
                    "inputs": {"events": "$extracted_events"},
                    "output": "events_path"
                },
                {
                    "id": "update_graph_from_ingestion" if not append_only_ing else "append_graph_from_ingestion",
                    "tool": "update_graph_data" if not append_only_ing else "append_only_update_graph",
                    "inputs": {"events_list": "$extracted_events", "allow_append_original_forms": allow_append_forms_ing} if append_only_ing else {"events_list": "$extracted_events"},
                    "output": "kg_update_result_ingestion"
                },
                {
                    "id": "refresh_kg_after_ingestion",
                    "tool": "refresh_knowledge_graph",
                    "inputs": {},
                    "output": "kg_refresh_result_ingestion"
                },
                {
                    "id": "report_ingestion",
                    "tool": "generate_markdown_report",
                    "inputs": {"events_list": "$extracted_events", "title": "Ingestion Extracted Events Report"},
                    "output": "ingestion_report_md"
                }
            ]
        }
        
        if auto_update_kg:
            pipeline_def["steps"].append({
                "id": "update_kg",
                "tool": "update_graph_data",
                "inputs": {"events_list": "$extracted_events"},
                "output": "update_status"
            })
            
        if enable_report:
            pipeline_def["steps"].append({
                "id": "generate_report",
                "tool": "generate_markdown_report",
                "inputs": {"events_list": "$extracted_events", "title": f"Ingestion Report {datetime.now().strftime('%Y-%m-%d')}"},
                "output": "final_report_md"
            })
            
        execute_pipeline(pipeline_def)

def render_expansion_tab():
    st.header("🔍 Knowledge Expansion")
    st.caption("Search for news based on keywords to discover new entities.")
    
    st.subheader("Define Search Tasks")
    st.info("Manage search keywords. You can add entities from the Knowledge Graph or manually type new keywords in the table.")
    # 选择启用的搜索 API（与 Configuration 共用同一配置）
    selected_apis = st.session_state.ingestion_apis[st.session_state.ingestion_apis["enabled"] == True]["name"].tolist()
    
    # 工具栏：从下拉列表添加
    entities = utils.load_entities()
    if entities:
        all_entity_names = sorted(list(entities.keys()))
        
        c_add_sel, c_add_btn = st.columns([3, 1])
        with c_add_sel:
            selected_entities = st.multiselect(
                "Select Entities from Graph", 
                options=all_entity_names,
                placeholder="Choose entities to add..."
            )
        with c_add_btn:
            st.write("") # Spacer
            st.write("") 
            if st.button("➕ Add Selected", use_container_width=True):
                if selected_entities:
                    new_rows = []
                    # 获取现有关键词以避免重复
                    existing_kws = set()
                    if not st.session_state.expansion_tasks.empty:
                        existing_kws = set(st.session_state.expansion_tasks["keyword"].tolist())
                        
                    count = 0
                    for ent in selected_entities:
                        if ent not in existing_kws:
                            new_rows.append({
                                "enabled": True,
                                "keyword": ent,
                                "limit": 5,
                                "category": "general",
                                "from": "",
                                "to": "",
                                "sortby": ""
                            })
                            count += 1
                    
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        st.session_state.expansion_tasks = pd.concat(
                            [st.session_state.expansion_tasks, new_df], 
                            ignore_index=True
                        )
                        st.success(f"Added {count} new tasks!")
                        st.rerun()
                    else:
                        st.warning("Selected entities are already in the list.")
                else:
                    st.warning("Please select entities first.")
    
    # 日期时间快捷填充（可选，使用日历选择器生成 ISO8601 字符串并批量填充）
    with st.expander("日期时间快捷填充（可选）", expanded=False):
        col_from, col_to = st.columns(2)
        min_date = datetime(2020, 1, 1).date()
        max_date = datetime.now().date()
        with col_from:
            d_from = st.date_input("From 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_from_date_expansion_picker")
            t_from = st.time_input("From 时间", value=None, key="gnews_from_time_expansion_picker")
        with col_to:
            d_to = st.date_input("To 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_to_date_expansion_picker")
            t_to = st.time_input("To 时间", value=None, key="gnews_to_time_expansion_picker")

        def combine(dt, tm):
            if dt is None:
                return None
            tm = tm or datetime.min.time()
            return datetime.combine(dt, tm, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

        from_iso = combine(d_from, t_from)
        to_iso = combine(d_to, t_to)
        st.caption(f"From (ISO8601): {from_iso or '未设置'}")
        st.caption(f"To   (ISO8601): {to_iso or '未设置'}")

        apply_from_all = st.checkbox("将 From 填充到所有行（若设置）", value=False, key="apply_from_all_expansion")
        apply_to_all = st.checkbox("将 To 填充到所有行（若设置）", value=False, key="apply_to_all_expansion")

        if st.button("应用到任务表", type="primary", key="apply_datetime_expansion"):
            df = st.session_state.expansion_tasks.copy()
            if from_iso:
                if apply_from_all:
                    df["from"] = from_iso
                else:
                    df.loc[(df["from"].isna()) | (df["from"] == ""), "from"] = from_iso
            if to_iso:
                if apply_to_all:
                    df["to"] = to_iso
                else:
                    df.loc[(df["to"].isna()) | (df["to"] == ""), "to"] = to_iso
            st.session_state.expansion_tasks = df
            st.success("已应用到任务表，请在下方表格确认。")
            st.rerun()

    # 任务表格编辑器
    edited_tasks = st.data_editor(
        st.session_state.expansion_tasks,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "keyword": st.column_config.TextColumn("Keyword", required=True, help="Type manually or added from dropdown"),
            "limit": st.column_config.NumberColumn("Limit", min_value=1, max_value=10, default=5),
            "category": st.column_config.TextColumn("Category", help="GNews category，如 general/business/...，可空"),
            "from": st.column_config.TextColumn("From (ISO8601)", help="可空，如 2025-12-01T00:00:00Z"),
            "to": st.column_config.TextColumn("To (ISO8601)", help="可空，如 2025-12-31T23:59:59Z"),
            "sortby": st.column_config.SelectboxColumn("Sortby", options=["", "publishedAt", "relevance"]),
        },
        key="expansion_tasks_editor"
    )
    st.session_state.expansion_tasks = edited_tasks

    st.subheader("🚀 Run Expansion")
    
    # 过滤出启用的任务
    active_tasks = st.session_state.expansion_tasks[st.session_state.expansion_tasks["enabled"] == True]
    
    c1, c2 = st.columns(2)
    c1.metric("Selected APIs", len(selected_apis))
    c2.metric("Active Tasks", len(active_tasks))

    st.subheader("图谱更新模式")
    col_mode, col_forms = st.columns(2)
    with col_mode:
        append_only_mode = st.checkbox("仅追加（不改旧数据）", value=True, help="不修改已有实体/事件，只新增不存在的记录")
    with col_forms:
        allow_append_forms = st.checkbox("追加旧实体的 original_forms", value=True, help="仅在仅追加模式下生效；关闭则完全不改旧实体字段")
    
    if st.button("Start Expansion Task", type="primary", use_container_width=True):
        if not selected_apis:
            st.error("Please select at least one Search API.")
            return
        if active_tasks.empty:
            st.error("Please define and enable at least one Search Task.")
            return
            
        # 构建 Pipeline：为每个启用任务生成一个步骤
        pipeline_steps = []
        for idx, row in active_tasks.iterrows():
            kw = row["keyword"]
            step_id = f"search_{kw.replace(' ', '_')}_{idx}"
            
            pipeline_steps.append({
                    "id": step_id,
                    "tool": "search_news_by_keywords", 
                    "inputs": {
                        "keywords": [kw], # 工具期望列表
                        "apis": selected_apis,
                        "limit": int(row.get("limit", 50)),
                        "category": row.get("category") or None,
                        "from": row.get("from") or None,
                        "to": row.get("to") or None,
                        "sortby": row.get("sortby") or None
                    },
                    "output": f"results_{idx}"
            })
            # 先对拓展结果做事件提取
            pipeline_steps.append({
                "id": f"extract_{kw.replace(' ', '_')}_{idx}",
                "tool": "batch_process_news",
                "inputs": {"news_list": f"$results_{idx}"},
                "output": f"extracted_events_{idx}"
            })
            # 将提取的事件暂存到 tmp，方便后续预览
            pipeline_steps.append({
                "id": f"save_events_{kw.replace(' ', '_')}_{idx}",
                "tool": "save_extracted_events_tmp",
                "inputs": {"events": f"$extracted_events_{idx}"},
                "output": f"events_path_{idx}"
            })
            pipeline_steps.append({
                "id": f"persist_{kw.replace(' ', '_')}_{idx}",
                "tool": "persist_expanded_news_tmp",
                "inputs": {
                    "expanded_news": f"$results_{idx}"
                },
                "output": f"persist_result_{idx}"
            })
        # 汇总所有任务的提取结果用于后续更新/报告
        all_extracted_keys = [f"$extracted_events_{i}" for i in range(len(active_tasks))]
        if append_only_mode:
            pipeline_steps.append({
                "id": "append_graph_from_expansion",
                "tool": "append_only_update_graph",
                "inputs": {
                    "events_list": all_extracted_keys,
                    "allow_append_original_forms": allow_append_forms
                },
                "output": "kg_update_result"
            })
        else:
            pipeline_steps.append({
                "id": "update_graph_from_expansion",
                "tool": "update_graph_data",
                "inputs": {"events_list": all_extracted_keys},
                "output": "kg_update_result"
            })
        pipeline_steps.append({
            "id": "refresh_kg_after_expansion",
            "tool": "refresh_knowledge_graph",
            "inputs": {},
            "output": "kg_refresh_result"
        })
        pipeline_steps.append({
            "id": "report_expansion",
            "tool": "generate_markdown_report",
            "inputs": {
                "events_list": all_extracted_keys,
                "title": "Expansion Extracted Events Report"
            },
            "output": "expansion_report_md"
        })
        
        pipeline_def = {
            "name": "Knowledge Expansion Batch",
            "steps": pipeline_steps
        }
        execute_pipeline(pipeline_def)

def render_maintenance_tab():
    st.header("🕸️ Graph Maintenance")
    
    # 当前临时数据快照（实体/事件）
    data_dir = ROOT_DIR / "data"
    entities_tmp_file = data_dir / "tmp" / "entities_tmp.json"
    events_tmp_file = data_dir / "tmp" / "abstract_to_event_map_tmp.json"
    extracted_dir = data_dir / "tmp"
    deduped_dir = data_dir / "tmp" / "deduped_news"
    raw_dir = data_dir / "tmp" / "raw_news"

    @st.cache_data(ttl=60)
    def load_json_cached(path: Path):
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"{path.name} 读取失败: {e}")
        return {}

    entities_tmp = load_json_cached(entities_tmp_file)
    events_tmp = load_json_cached(events_tmp_file)
    @st.cache_data(ttl=60)
    def list_extracted_files(base: Path):
        files = sorted(base.glob("extracted_events_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(f) for f in files]
    extracted_files = list_extracted_files(extracted_dir)

    @st.cache_data(ttl=60)
    def list_news_files(base: Path, pattern: str):
        files = sorted(base.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(f) for f in files]
    deduped_files = list_news_files(deduped_dir, "*.jsonl")
    raw_files = list_news_files(raw_dir, "*.jsonl")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("临时实体（缓存条数）", len(entities_tmp))
    c2.metric("临时事件（缓存条数）", len(events_tmp))
    c3.metric("提取结果文件数", len(extracted_files))
    c4.metric("去重新闻文件数", len(deduped_files))
    c5.metric("原始新闻文件数", len(raw_files))

    with st.expander("查看临时实体 / 事件示例", expanded=False):
        if entities_tmp:
            df_ent = pd.DataFrame(
                [
                    {
                        "name": k,
                        "first_seen": v.get("first_seen", ""),
                        "sources": ",".join(
                            [
                                (
                                    s.get("name")
                                    or s.get("id")
                                    or s.get("url")
                                    or str(s)
                                )
                                if isinstance(s, dict)
                                else str(s)
                                for s in v.get("sources", [])
                            ]
                        )[:80],
                    }
                    for k, v in list(entities_tmp.items())[:50]
                ]
            )
            st.write("临时实体（最多50条预览）")
            st.dataframe(df_ent, use_container_width=True)
        else:
            st.info("暂无临时实体数据")

        if events_tmp:
            df_evt = pd.DataFrame(
                [{"abstract": k, "first_seen": v.get("first_seen", ""), "entities": ",".join(v.get("entities", []))[:80]} for k, v in list(events_tmp.items())[:50]]
            )
            st.write("临时事件（最多50条预览）")
            st.dataframe(df_evt, use_container_width=True)
        else:
            st.info("暂无临时事件数据")
        
        if extracted_files:
            st.write("提取结果文件（最新5个）")
            st.table({"path": extracted_files[:5]})
        else:
            st.info("暂无提取结果文件")

        if deduped_files:
            st.write("去重新闻文件（最新5个）")
            st.table({"path": deduped_files[:5]})
        else:
            st.info("暂无去重新闻文件")

        if raw_files:
            st.write("原始新闻文件（最新5个）")
            st.table({"path": raw_files[:5]})
        else:
            st.info("暂无原始新闻文件")
    
    with st.form("maintenance_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Deduplication")
            strict = st.checkbox("Strict Mode", True)
            thresh = st.slider("Similarity", 0.5, 1.0, 0.9)
        with c2:
            st.subheader("Cleaning")
            rm_iso = st.checkbox("Remove Isolated Nodes")
        
        st.subheader("导入 tmp 抽取结果")
        use_tmp_events = st.checkbox("刷新前先追加 tmp/extracted_events_*.jsonl", value=True)
        max_tmp_files = st.number_input("最多读取文件数（0=全部）", min_value=0, value=0, step=1)
        allow_forms_tmp = st.checkbox("追加旧实体 original_forms（追加模式）", value=True)
            
        submitted = st.form_submit_button("🚀 Run Maintenance", type="primary", use_container_width=True)
        
    if submitted:
        pipeline_def = {
            "name": "Graph Maintenance",
            # 调用 Agent3 刷新压缩知识图谱（基于现有实体/事件文件）
            "steps": [
                *(
                    [
                        {
                            "id": "append_tmp_events",
                            "tool": "append_tmp_extracted_events",
                            "inputs": {
                                "max_files": int(max_tmp_files),
                                "allow_append_original_forms": allow_forms_tmp
                            },
                            "output": "tmp_append_result"
                        }
                    ]
                    if use_tmp_events else []
                ),
                {
                    "id": "refresh_kg",
                    "tool": "refresh_knowledge_graph",
                    "inputs": {},
                    "output": "status"
                }
            ]
        }
        execute_pipeline(pipeline_def)
        # 清理临时缓存文件并刷新缓存
        try:
            for p in [entities_tmp_file, events_tmp_file]:
                if p.exists():
                    p.unlink()
            st.cache_data.clear()
            st.success("已清理临时缓存文件")
        except Exception as e:
            st.warning(f"清理缓存失败: {e}")

def render_custom_builder():
    st.header("🛠️ Custom Pipeline Builder")
    
    col_builder, col_preview = st.columns([1.5, 1])
    
    with col_builder:
        # 工具栏
        c_add, c_save, c_load = st.columns([2, 1, 1])
        with c_add:
            tools = FunctionRegistry.get_all_tools()
            selected_tool = st.selectbox("Select Tool", list(tools.keys()), label_visibility="collapsed")
            if st.button("Add Step", use_container_width=True):
                 st.session_state.pipeline_steps.append({
                    "id": f"step_{len(st.session_state.pipeline_steps) + 1}",
                    "tool": selected_tool,
                    "inputs": {}
                })
                 st.rerun()

        # 步骤编辑
        if not st.session_state.pipeline_steps:
            st.info("No steps added. Select a tool to start.")
        else:
            for i, step in enumerate(st.session_state.pipeline_steps):
                tool_name = step["tool"]
                tool_meta = tools.get(tool_name, {})
                
                with st.expander(f"Step {i+1}: {tool_name}", expanded=False):
                    c_id, c_del = st.columns([4, 1])
                    step["id"] = c_id.text_input("ID", step["id"], key=f"id_{i}")
                    if c_del.button("🗑️", key=f"del_{i}"):
                        st.session_state.pipeline_steps.pop(i)
                        st.rerun()
                        
                    st.caption(tool_meta.get("description", ""))
                    
                    # 参数编辑区
                    params = tool_meta.get("parameters", {})
                    if params:
                        for p_name, p_info in params.items():
                            render_input_field(i, p_name, p_info, step.get("inputs", {}), step)
                    
                    step["output"] = st.text_input("Output to ($var)", step.get("output", ""), key=f"out_{i}")

    with col_preview:
        st.subheader("Preview")
        pipeline_def = {"name": "Custom Pipeline", "steps": st.session_state.pipeline_steps}
        st.code(yaml.dump(pipeline_def, sort_keys=False), language="yaml")
        
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            execute_pipeline(pipeline_def)

def render_snapshots_tab():
    st.header("📸 Knowledge Graph Snapshots")
    st.caption("生成/查看可视化快照（kg_visual.json / kg_visual_timeline.json）")
    if st.button("生成快照", type="primary"):
        try:
            from src.functions.graph_ops import generate_kg_visual_snapshots
            res = generate_kg_visual_snapshots()
            st.success(f"生成完成: {res}")
        except Exception as e:
            st.error(f"生成失败: {e}")
    data_root = ROOT_DIR / "data"
    vis_path = data_root / "kg_visual.json"
    tl_path = data_root / "kg_visual_timeline.json"
    st.write("快照文件路径：")
    st.write(f"- 图谱快照: {vis_path}")
    st.write(f"- 时间线快照: {tl_path}")
    for p in [vis_path, tl_path]:
        if p.exists():
            ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"{p.name} 已存在，大小 {p.stat().st_size} 字节，修改时间 {ts}")
        else:
            st.warning(f"{p.name} 尚未生成")


def render_tool_explorer_tab():
    st.header("Tool Explorer")
    st.caption("自动发现所有注册工具 · 支持搜索、预览、复制、一键执行")

    # 1. 自动加载所有真实工具（核心！）
    all_tools = FunctionRegistry.get_all_tools()  # <-- 你的真实注册表
    if not all_tools:
        st.warning("未检测到已注册的工具，请检查 FunctionRegistry")
        return

    # 2. 可选：给工具打 tag 分组（推荐在函数上加 @tool 装饰器时顺便加 category）
    #    如果你还没加，这里提供一个默认分组逻辑
    CATEGORY_ORDER = {
        "Data Fetch": ["fetch", "search", "scrape", "crawl"],
        "Extraction": ["extract", "process", "parse", "llm"],
        "Graph Ops": ["graph", "update", "refresh", "merge", "kg", "node", "edge"],
        "Reporting": ["report", "markdown", "summary", "export"],
        "Utility": ["save", "load", "tmp", "debug", "test"],
    }

    def get_category(tool_name: str) -> str:
        name_lower = tool_name.lower()
        for cat, keywords in CATEGORY_ORDER.items():
            if any(k in name_lower for k in keywords):
                return cat
        return "Other"

    # 添加分类
    categorized = {}
    for name, meta in all_tools.items():
        cat = meta.get("category") or get_category(name)  # 支持手动 category
        categorized.setdefault(cat, []).append((name, meta))

    # 3. 搜索框
    search = st.text_input("Search Tools", placeholder="输入工具名或描述关键词...", key="tool_search")
    
    # 过滤
    if search:
        filtered = {}
        for cat, tools in categorized.items():
            matched = []
            for name, meta in tools:
                if (search.lower() in name.lower() or 
                    (meta.get("description") and search.lower() in meta.get("description", "").lower())):
                    matched.append((name, meta))
            if matched:
                filtered[cat] = matched
        categorized = filtered

    # 4. 主渲染区 - 响应式卡片流
    for category, tools in categorized.items():
        with st.expander(f"**{category}** · {len(tools)} tools", expanded=True):
            cols = st.columns(3, gap="medium")  # 每行3个卡片，可改成2或4
            for idx, (tool_name, meta) in enumerate(tools):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"**`{tool_name}`**")
                        
                        desc = meta.get("description") or "No description"
                        st.caption(desc)

                        # 参数表单（可编辑）
                        params = meta.get("parameters", {})
                        if params:
                            with st.form(key=f"form_{tool_name}_{idx}", clear_on_submit=False, border=False):
                                inputs = {}
                                for p_name, p_info in params.items():
                                    p_type = p_info.get("type", "str")
                                    default = p_info.get("default")
                                    desc = p_info.get("description", "")
                                    
                                    label = f"{p_name}{' *' if p_info.get('required') else ''}"
                                    
                                    if p_type == "bool":
                                        val = st.checkbox(label, value=bool(default), help=desc)
                                    elif p_type in ("int", "integer"):
                                        val = st.number_input(label, value=int(default) if default is not None else 0, step=1, help=desc)
                                    elif p_type == "float":
                                        val = st.number_input(label, value=float(default) if default is not None else 0.0, step=0.1, help=desc)
                                    elif p_type in ("list", "dict", "json"):
                                        val_str = json.dumps(default, ensure_ascii=False) if default is not None else "[]"
                                        val_input = st.text_area(label, value=val_str, height=80, help=desc + "\n支持 JSON 或 $变量引用")
                                        if val_input.strip().startswith("$"):
                                            inputs[p_name] = val_input.strip()
                                        else:
                                            try:
                                                inputs[p_name] = json.loads(val_input) if val_input.strip() else None
                                            except:
                                                inputs[p_name] = val_input  # 允许字符串
                                    else:
                                        val = st.text_input(label, value=str(default) if default is not None else "", help=desc)
                                        inputs[p_name] = val if val else None

                                col_run, col_copy = st.columns([1, 2])
                                with col_run:
                                    run_now = st.form_submit_button("Run", type="primary", use_container_width=True)
                                with col_copy:
                                    copy_step = st.form_submit_button("Copy as Step", use_container_width=True)

                                # 一键运行（调试神器！）
                                if run_now:
                                    with st.spinner(f"Running {tool_name}..."):
                                        try:
                                            context = PipelineContext()
                                            engine = PipelineEngine(context)
                                            result = asyncio.run(engine.run_task({
                                                "id": f"debug_{tool_name}",
                                                "tool": tool_name,
                                                "inputs": inputs
                                            }))
                                            st.success("Success!")
                                            st.json(result, expanded=False)
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                # 一键复制为 Pipeline Step
                                if copy_step:
                                    step_yaml = {
                                        "id": tool_name,
                                        "tool": tool_name,
                                        "inputs": inputs,
                                        "output": f"{tool_name}_result"
                                    }
                                    yaml_str = yaml.dump(step_yaml, sort_keys=False, allow_unicode=True)
                                    st.code(yaml_str, language="yaml")
                                    st.toast("已复制到剪贴板（模拟）", icon="clipboard")
                                    # 真实复制到剪贴板（Streamlit 1.30+）
                                    try:
                                        st.code(yaml_str, language="yaml")
                                        st.success("Step YAML 已生成，可直接粘贴到 Custom Builder")
                                    except:
                                        pass
                        else:
                            st.info("No parameters")

                        # 底部标签
                        tags = []
                        if meta.get("async"): tags.append("async")
                        if "llm" in tool_name.lower(): tags.append("LLM")
                        if tags:
                            st.caption(" · ".join(tags))

# --- 主导航 ---
tabs = st.tabs(["Configuration","Ingestion", "Expansion", "Maintenance", "Snapshots", "Tools", "Custom Builder"])

with tabs[0]: render_configuration_tab()
with tabs[1]: render_ingestion_tab()
with tabs[2]: render_expansion_tab()
with tabs[3]: render_maintenance_tab()
with tabs[4]: render_snapshots_tab()
with tabs[5]: render_tool_explorer_tab()
with tabs[6]: render_custom_builder()

