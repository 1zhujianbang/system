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
from src.utils.tool_function import tools
tools = tools()

# 添加项目根目录到 path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
ENV_PATH = ROOT_DIR / "config" / ".env.local"

from src.core.registry import FunctionRegistry
from src.core.engine import PipelineEngine
from src.core.context import PipelineContext
from src.web import utils
import src.functions.data_fetch
import src.functions.extraction
import src.functions.graph_ops
import src.functions.reporting

st.set_page_config(page_title="新闻智能体系统 - 流水线构建器", page_icon="⛓️", layout="wide")

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
                
            # 完成 - 更新历史记录状态
            with self._lock:
                self.status_info = {"label": "✅ Pipeline Execution Completed!", "state": "complete", "expanded": False}
                self.final_report = context.get("final_report_md")

                # 更新历史记录
                import streamlit as st
                if "pipeline_history" in st.session_state and st.session_state.pipeline_history:
                    # 更新最新记录的状态
                    st.session_state.pipeline_history[-1]["status"] = "success"
                
        except Exception as e:
            with self._lock:
                self.status_info = {"label": f"❌ Execution Failed: {str(e)}", "state": "error", "expanded": True}
                self.logs.append(f"[System] Error: {str(e)}")

                # 更新历史记录
                import streamlit as st
                if "pipeline_history" in st.session_state and st.session_state.pipeline_history:
                    # 更新最新记录的状态
                    st.session_state.pipeline_history[-1]["status"] = "failed"
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
                    if st.button("🔄 Refresh View", key="pipeline_refresh_view_monitor"):
                        st.rerun()
                    # 自动刷新逻辑 (实验性)
                    time.sleep(2)
                    st.rerun()
                else:
                    if st.button("Clear Status", key="pipeline_clear_status_monitor"):
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
        st.warning("⚠️ 已有任务正在运行。请等待完成后再试。")
        return

    success = task_manager.start_task(pipeline_def)
    if success:
        # 记录执行历史
        if "pipeline_history" not in st.session_state:
            st.session_state.pipeline_history = []

        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": pipeline_def.get("name", "Unknown Pipeline"),
            "steps": len(pipeline_def.get("steps", [])),
            "status": "running"  # 初始状态
        }
        st.session_state.pipeline_history.append(history_entry)

        st.toast("🚀 Task started in background!")
        st.rerun()
    else:
        st.error("启动任务失败。")

class InputRenderer:
    """输入字段渲染器"""

    def __init__(self, step_idx, p_name, p_info, current_inputs, step):
        self.step_idx = step_idx
        self.p_name = p_name
        self.p_info = p_info
        self.current_inputs = current_inputs
        self.step = step

        self.p_type = p_info.get('type', 'Any')
        self.p_required = p_info.get('required', False)
        self.default_val = p_info.get('default')

    def render(self):
        """渲染输入字段"""
        current_val = self.current_inputs.get(self.p_name, self.default_val)
        is_ref = isinstance(current_val, str) and current_val.startswith("$")

        if is_ref:
            self._render_variable_input(current_val)
            return

        # 根据类型渲染不同的输入组件
        if "bool" in self.p_type.lower():
            self._render_bool_input(current_val)
        elif "int" in self.p_type.lower():
            self._render_int_input(current_val)
        elif "list" in self.p_type.lower() or "dict" in self.p_type.lower():
            self._render_json_input(current_val)
        else:
            self._render_text_input(current_val)

    def _get_label_and_help(self):
        """获取标签和帮助文本"""
        label = f"{self.p_name}{' *' if self.p_required else ''}"
        help_text = f"Type: {self.p_type}"
        if self.default_val is not None:
            help_text += f", Default: {self.default_val}"
        return label, help_text

    def _get_key(self):
        """获取组件唯一键"""
        return f"in_{self.step_idx}_{self.p_name}"

    def _render_variable_input(self, current_val):
        """渲染变量引用输入"""
        label, help_text = self._get_label_and_help()
        key = self._get_key()

        new_val = st.text_input(label + " (Variable)", value=current_val, key=key, help=help_text)
        self.step["inputs"][self.p_name] = new_val

    def _render_bool_input(self, current_val):
        """渲染布尔输入"""
        label, help_text = self._get_label_and_help()
        key = self._get_key()

        val = st.checkbox(label, value=bool(current_val) if current_val is not None else False, key=key, help=help_text)
        self.step["inputs"][self.p_name] = val

    def _render_int_input(self, current_val):
        """渲染整数输入"""
        label, help_text = self._get_label_and_help()
        key = self._get_key()

        val = st.number_input(label, value=int(current_val) if current_val is not None else 0, step=1, key=key, help=help_text)
        self.step["inputs"][self.p_name] = int(val)

    def _render_json_input(self, current_val):
        """渲染JSON输入"""
        label, help_text = self._get_label_and_help()
        key = self._get_key()

        val_str = str(current_val) if current_val is not None else "[]"
        new_val_str = st.text_area(label + " (JSON/List)", value=val_str, height=100, key=key, help="Enter valid JSON or $variable")
        if new_val_str.startswith("$"):
            self.step["inputs"][self.p_name] = new_val_str
        else:
            self._parse_json_value(new_val_str)

    def _render_text_input(self, current_val):
        """渲染文本输入"""
        label, help_text = self._get_label_and_help()
        key = self._get_key()

        val = st.text_input(label, value=str(current_val) if current_val is not None else "", key=key, help=help_text)
        if val:
            self.step["inputs"][self.p_name] = val

    def _parse_json_value(self, value_str):
        """安全解析JSON值"""
        try:
            import json
            if value_str.strip():
                # 使用安全的JSON解析
                parsed = json.loads(value_str)
                self.step["inputs"][self.p_name] = parsed
        except json.JSONDecodeError:
            # 如果不是有效的JSON，尝试基础的Python字面量解析
            self._parse_literal_value(value_str)

    def _parse_literal_value(self, value_str):
        """解析基础字面量值"""
        try:
            stripped = value_str.strip()
            if stripped in ('True', 'False'):
                self.step["inputs"][self.p_name] = stripped == 'True'
            elif stripped == 'None':
                self.step["inputs"][self.p_name] = None
            elif self._is_numeric(stripped):
                self.step["inputs"][self.p_name] = float(stripped) if '.' in stripped else int(stripped)
            else:
                # 对于复杂类型或字符串，直接存储原值
                self.step["inputs"][self.p_name] = value_str
        except:
            self.step["inputs"][self.p_name] = value_str

    @staticmethod
    def _is_numeric(value):
        """检查是否为数值"""
        test_val = value.replace('.', '').replace('-', '')
        return test_val.isdigit() or (value.startswith('-') and value[1:].replace('.', '').isdigit())

def render_input_field(step_idx, p_name, p_info, current_inputs, step):
    """智能渲染输入组件（兼容性接口）"""
    renderer = InputRenderer(step_idx, p_name, p_info, current_inputs, step)
    renderer.render()

# --- 场景化模块渲染 ---
def render_source_configuration():
    """渲染数据源配置"""
    st.subheader("Source Configuration")
    st.write("Source Configuration")

    # 渲染数据源编辑器
    edited_df = render_source_editor()
    st.session_state.ingestion_apis = edited_df

    # 显示选中源统计
    selected_apis = edited_df[edited_df["enabled"] == True]["name"].tolist()
    st.session_state["ingestion_selected_apis"] = selected_apis
    st.caption(f"✅ Selected Sources: {len(selected_apis)}")

    # GNews参数配置
    render_gnews_params()

def render_source_editor():
    """渲染数据源编辑器"""
    return st.data_editor(
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

def render_gnews_params():
    """渲染GNews参数配置"""
    gnews_params = st.session_state.get("gnews_params", {})
    with st.expander("GNews 可选参数", expanded=False):
        # 基本参数
        category = st.selectbox(
            "Category",
            ["", "general", "world", "business", "technology", "sports", "science", "health", "entertainment"],
            index=0,
            help="留空则不指定分类",
            key="gnews_category_source_config"
        )
        query = st.text_input("Query (关键词搜索，可空)", key="gnews_query_source_config")

        # 日期时间选择
        from_iso, to_iso = render_datetime_range()

        # 高级参数
        nullable = st.text_input("Nullable", value=gnews_params.get("nullable", ""), help="如 description,content", key="gnews_nullable_source_config")
        truncate = st.text_input("Truncate", value=gnews_params.get("truncate", ""), help="如 content", key="gnews_truncate_source_config")
        sortby = st.selectbox("Sortby", ["", "publishedAt", "relevance"], index=0, key="gnews_sortby_source_config")
        in_fields = st.text_input("In fields", value=gnews_params.get("in_fields", ""), help="如 title,description", key="gnews_infields_source_config")
        page = st.number_input("Page", min_value=1, value=gnews_params.get("page", 1), step=1, key="gnews_page_source_config")

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

def render_datetime_range():
    """渲染日期时间范围选择器"""
    col_from, col_to = st.columns(2)
    min_date = datetime(2020, 1, 1).date()
    max_date = datetime.now().date()

    with col_from:
        d_from = st.date_input("From 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_from_date_source_config")
        t_from = st.time_input("From 时间", value=None, key="gnews_from_time_source_config")

    with col_to:
        d_to = st.date_input("To 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_to_date_source_config")
        t_to = st.time_input("To 时间", value=None, key="gnews_to_time_source_config")

    def combine(dt, tm):
        if dt is None:
            return None
        tm = tm or datetime.min.time()
        # 统一使用 UTC 输出 ISO8601
        return datetime.combine(dt, tm, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    from_iso = combine(d_from, t_from)
    to_iso = combine(d_to, t_to)

    return from_iso, to_iso

def render_source_api_pool():
    """渲染数据源API池配置"""
    st.subheader("Source API Pool")
    st.caption("编辑并保存到 config/.env.local（覆盖写入，不保留注释/空行）。上方\"保存该行/表格修改\"仅更新内存，需在下方点击保存才写入文件。")

    # 环境变量编辑器
    edited_env = render_env_editor()

    # 渲染每个环境变量的详细编辑器
    render_env_detail_editors(edited_env)

    # 保存按钮
    render_env_save_button(edited_env)

def render_env_editor():
    """渲染环境变量编辑器"""
    edited_env = st.session_state.env_kv
    if st.checkbox("显示 Key/Value 表格编辑器", value=False, key="env_editor_toggle_source_api"):
        edited_env = st.data_editor(
            st.session_state.env_kv,
            num_rows="dynamic",
            use_container_width=True,
                    column_config={
                        "key": st.column_config.TextColumn("Key", required=True),
                        "value": st.column_config.TextColumn("Value", required=False, help="可输入占位符，注意避免泄露敏感值")
                    },
                    key="env_editor_config_tab"
        )
        st.session_state.env_kv = edited_env
    return edited_env

def render_env_detail_editors(edited_env):
    """渲染环境变量详细编辑器"""
    for _, row in edited_env.iterrows():
        idx = row.name
        k = str(row.get("key", "")).strip()
        v = str(row.get("value", "")).strip()
        if not k:
            continue

        parsed, err = try_parse_json(v)
        with st.expander(f"{k}", expanded=False):
            if parsed is not None:
                render_json_editor(idx, parsed)
            else:
                st.warning(f"无法解析为 JSON：{err}")

def render_json_editor(idx, parsed):
    """渲染JSON编辑器"""
    pretty_txt = json.dumps(parsed, ensure_ascii=False, indent=2)
    new_txt = st.text_area(
        "JSON 编辑（保存即写回 value）",
        value=pretty_txt,
        key=f"json_edit_{idx}",
        height=200
    )
    if st.button("保存该行", key=f"save_json_source_api_{idx}", use_container_width=True):
        save_json_row(idx, new_txt)

    # 表格化展示
    render_table_editor(idx, parsed)

def save_json_row(idx, new_txt):
    """保存JSON行"""
    try:
        parsed_new = json.loads(new_txt)
        # 写回表格缓存，保持紧凑存储
        st.session_state.env_kv.at[idx, "value"] = json.dumps(parsed_new, ensure_ascii=False)
        st.success("已更新该行的 value")
    except Exception as e:
        st.error(f"JSON 解析失败: {e}")

def render_table_editor(idx, parsed):
    """渲染表格编辑器"""
    df_preview, kind = to_table(parsed)
    if df_preview is not None and not df_preview.empty:
        edited_df = st.data_editor(
            df_preview,
            num_rows="dynamic",
            use_container_width=True,
            key=f"env_table_{idx}"
        )
        if st.button("保存表格修改", key=f"save_table_{idx}", use_container_width=True):
            save_table_modification(idx, edited_df, kind)

def save_table_modification(idx, edited_df, kind):
    """保存表格修改"""
    try:
        if kind == "list":
            new_obj = edited_df.to_dict(orient="records")
        else:
            new_obj = edited_df.to_dict(orient="records")[0] if not edited_df.empty else {}
        st.session_state.env_kv.at[idx, "value"] = json.dumps(new_obj, ensure_ascii=False)
        st.success("已根据表格修改更新 value")
    except Exception as e:
        st.error(f"写回 JSON 失败: {e}")

def to_table(obj):
    """将对象转换为表格格式"""
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return pd.DataFrame(obj), "list"
    if isinstance(obj, dict):
        return pd.DataFrame([obj]), "dict"
    return None, ""

def render_env_save_button(edited_env):
    """渲染环境变量保存按钮"""
    if st.button("💾 保存到 .env.local", type="primary", use_container_width=True, key="system_save_env_local_source_api"):
        save_env_file(edited_env)

def save_env_file(edited_env):
    """保存环境变量文件"""
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

def try_parse_json(val: str):
    """尝试解析JSON字符串"""
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

def render_configuration_tab():
    """渲染配置标签页"""
    st.header("📚 Configuration")
    source_config, source_api_pool = st.tabs(["Source Configuration", "Source API Pool"])

    with source_config:
        render_source_configuration()

    with source_api_pool:
        render_source_api_pool()

        # GNews 可选参数配置
        gnews_params = st.session_state.get("gnews_params", {})
        with st.expander("GNews 可选参数", expanded=False):
            category = st.selectbox(
                "Category",
                ["", "general", "world", "business", "technology", "sports", "science", "health", "entertainment"],
                index=0,
                help="留空则不指定分类",
                key="gnews_category_config_tab"
            )
            query = st.text_input("Query (关键词搜索，可空)", key="gnews_query_config_tab")
            col_from, col_to = st.columns(2)
            min_date = datetime(2020, 1, 1).date()
            max_date = datetime.now().date()
            with col_from:
                d_from = st.date_input("From 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_from_date_config_tab")
                t_from = st.time_input("From 时间", value=None, key="gnews_from_time_config_tab")
            with col_to:
                d_to = st.date_input("To 日期", value=None, min_value=min_date, max_value=max_date, key="gnews_to_date_config_tab")
                t_to = st.time_input("To 时间", value=None, key="gnews_to_time_config_tab")

            def combine(dt, tm):
                if dt is None:
                    return None
                tm = tm or datetime.min.time()
                # 统一使用 UTC 输出 ISO8601
                return datetime.combine(dt, tm, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

            from_iso = combine(d_from, t_from)
            to_iso = combine(d_to, t_to)
            nullable = st.text_input("Nullable", value=gnews_params.get("nullable", ""), help="如 description,content", key="gnews_nullable_config_tab")
            truncate = st.text_input("Truncate", value=gnews_params.get("truncate", ""), help="如 content", key="gnews_truncate_config_tab")
            sortby = st.selectbox("Sortby", ["", "publishedAt", "relevance"], index=0, key="gnews_sortby_config_tab")
            in_fields = st.text_input("In fields", value=gnews_params.get("in_fields", ""), help="如 title,description", key="gnews_infields_config_tab")
            page = st.number_input("Page", min_value=1, value=gnews_params.get("page", 1), step=1, key="gnews_page_config_tab")

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
        if st.checkbox("显示 Key/Value 表格编辑器", value=False, key="env_editor_toggle_config_tab"):
            edited_env = st.data_editor(
                st.session_state.env_kv,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "key": st.column_config.TextColumn("Key", required=True),
                    "value": st.column_config.TextColumn("Value", required=False, help="可输入占位符，注意避免泄露敏感值")
                },
                key="env_editor_source_api"
            )
            st.session_state.env_kv = edited_env

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
                        key=f"json_edit_config_tab_{idx}",
                        height=200
                    )
                    if st.button("保存该行", key=f"save_json_config_tab_{idx}", use_container_width=True):
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
                            key=f"env_table_config_tab_{idx}"
                        )
                        if st.button("保存表格修改", key=f"save_table_config_tab_{idx}", use_container_width=True):
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

        if st.button("💾 保存到 .env.local", type="primary", use_container_width=True, key="system_save_env_local_config_tab"):
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
    """渲染数据摄入标签页"""
    st.header("📥 Data Ingestion")
    st.caption("Fetch news from sources (Feed/Search) and extract events.")

    # 处理参数配置
    params = render_ingestion_parameters()

    # 渲染配置摘要
    render_ingestion_summary(params)

    # 图谱更新模式配置
    update_config = render_graph_update_mode()

    # 执行按钮
    if render_ingestion_execution_button(params, update_config):
        execute_ingestion_pipeline(params, update_config)

def render_ingestion_parameters():
    """渲染摄入参数配置"""
    st.subheader("Processing Parameters")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("##### 📥 Fetch Settings")
        news_limit = st.number_input("Limit (per source)", 1, 10, 5, 1, help="Max news items to fetch per source.")
    with col_p2:
        st.markdown("##### ⚙️ Pipeline Actions")
        auto_update_kg = st.checkbox("Auto Update Knowledge Graph", True, help="Automatically extract entities and update the graph.")
        enable_report = st.checkbox("Generate Summary Report", True, help="Create a markdown report after processing.")
    return {
        'news_limit': news_limit,
        'auto_update_kg': auto_update_kg,
        'enable_report': enable_report
    }

def render_ingestion_summary(params):
    """渲染摄入配置摘要"""
    st.subheader("🚀 Ready to Start?")

    # 汇总配置
    current_df = st.session_state.ingestion_apis
    selected_sources = current_df[current_df["enabled"] == True]["name"].tolist()

    st.write("Summary:")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sources Selected", len(selected_sources))
    c2.metric("Max Items", params['news_limit'])
    c3.metric("Auto-Update KG", "Yes" if params['auto_update_kg'] else "No")

    return selected_sources

def render_graph_update_mode():
    """渲染图谱更新模式配置"""
    st.subheader("图谱更新模式")
    col_mode_ing, col_forms_ing = st.columns(2)
    with col_mode_ing:
        append_only_ing = st.checkbox("仅追加（不改旧数据）- Ingestion", value=True, help="不修改已有实体/事件，只新增不存在的记录")
    with col_forms_ing:
        allow_append_forms_ing = st.checkbox("追加旧实体的 original_forms - Ingestion", value=True, help="仅在仅追加模式下生效；关闭则完全不改旧实体字段")
    return {
        'append_only': append_only_ing,
        'allow_append_forms': allow_append_forms_ing
    }

def render_ingestion_execution_button(params, update_config):
    """渲染摄入执行按钮"""
    selected_sources = render_ingestion_summary(params)

    if not selected_sources:
        st.error("❌ 未选择数据源。请返回'Data Sources'标签页进行选择。")
        return False

    return st.button("Start Ingestion Task", type="primary", use_container_width=True, key="pipeline_start_ingestion_task")

def execute_ingestion_pipeline(params, update_config):
    """执行数据摄入流水线"""
    current_df = st.session_state.ingestion_apis
    selected_sources = current_df[current_df["enabled"] == True]["name"].tolist()
    gnews_params = st.session_state.get("gnews_params", {})

    pipeline_def = {
        "name": "Data Ingestion Task",
        "steps": [
            {
                "id": "fetch_news",
                "tool": "fetch_news_stream",
                "inputs": {
                    "limit": params['news_limit'],
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
                "id": "update_graph_from_ingestion" if not update_config['append_only'] else "append_graph_from_ingestion",
                "tool": "update_graph_data" if not update_config['append_only'] else "append_only_update_graph",
                "inputs": {"events_list": "$extracted_events", "allow_append_original_forms": update_config['allow_append_forms']} if update_config['append_only'] else {"events_list": "$extracted_events"},
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

    if params['auto_update_kg']:
        pipeline_def["steps"].append({
            "id": "update_kg",
            "tool": "update_graph_data",
            "inputs": {"events_list": "$extracted_events"},
            "output": "update_status"
        })

    if params['enable_report']:
        pipeline_def["steps"].append({
            "id": "generate_report",
            "tool": "generate_markdown_report",
            "inputs": {"events_list": "$extracted_events", "title": f"Ingestion Report {datetime.now().strftime('%Y-%m-%d')}"},
            "output": "final_report_md"
        })

    execute_pipeline(pipeline_def)

def render_entity_selector():
    """渲染实体选择器组件"""
    entities = utils.load_entities()
    if not entities:
        return

    all_entity_names = sorted(list(entities.keys()))

    c_add_sel, c_add_btn = st.columns([3, 1])
    with c_add_sel:
        selected_entities = st.multiselect(
            "Select Entities from Graph",
            options=all_entity_names,
            placeholder="Choose entities to add..."
        )
    with c_add_btn:
        st.write("")  # Spacer
        st.write("")
        if st.button("➕ Add Selected", use_container_width=True, key="expansion_add_selected_entities"):
            add_selected_entities(selected_entities)

def add_selected_entities(selected_entities):
    """添加选中的实体到任务列表"""
    if not selected_entities:
        st.warning("请先选择实体。")
        return

    # 获取现有关键词以避免重复
    existing_kws = set()
    if not st.session_state.expansion_tasks.empty:
        existing_kws = set(st.session_state.expansion_tasks["keyword"].tolist())

    new_rows = []
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
        st.success(f"已添加 {count} 个新任务！")
        st.rerun()
    else:
        st.warning("选中的实体已在列表中。")

def render_datetime_picker():
    """渲染日期时间选择器"""
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
            apply_datetime_to_tasks(from_iso, to_iso, apply_from_all, apply_to_all)

def apply_datetime_to_tasks(from_iso, to_iso, apply_from_all, apply_to_all):
    """应用日期时间设置到任务表"""
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

def render_expansion_tab():
    """渲染知识拓展标签页"""
    st.header("🔍 Knowledge Expansion")
    st.caption("Search for news based on keywords to discover new entities.")

    st.subheader("Define Search Tasks")
    st.info("管理搜索关键词。您可以从知识图谱添加实体，或在表格中手动输入新关键词。")

    # 选择启用的搜索 API
    selected_apis = st.session_state.ingestion_apis[st.session_state.ingestion_apis["enabled"] == True]["name"].tolist()

    # 实体选择器
    render_entity_selector()

    # 日期时间选择器
    render_datetime_picker()

    # 任务表格编辑器
    render_expansion_tasks_editor()

    # 运行控制面板
    render_expansion_run_panel(selected_apis)

def render_expansion_tasks_editor():
    """渲染拓展任务编辑器"""
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

def render_expansion_run_panel(selected_apis):
    """渲染拓展任务运行控制面板"""
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
    if st.button("Start Expansion Task", type="primary", use_container_width=True, key="pipeline_start_expansion_task"):
        execute_expansion_pipeline(selected_apis, active_tasks, append_only_mode, allow_append_forms)

def execute_expansion_pipeline(selected_apis, active_tasks, append_only_mode, allow_append_forms):
    """执行拓展流水线"""
    if not selected_apis:
        st.error("Please select at least one Search API.")
        return
    if active_tasks.empty:
        st.error("Please define and enable at least one Search Task.")
        return

    # 构建 Pipeline：为每个启用任务生成一个步骤
    pipeline_steps = create_expansion_pipeline_steps(selected_apis, active_tasks, append_only_mode, allow_append_forms)

    pipeline_def = {
        "name": "Knowledge Expansion Batch",
        "steps": pipeline_steps
    }
    execute_pipeline(pipeline_def)

class ExpansionPipelineBuilder:
    """知识扩展流水线构建器"""

    def __init__(self):
        self.steps = []

    def add_search_steps(self, selected_apis, active_tasks):
        """添加搜索步骤"""
        for idx, row in active_tasks.iterrows():
            kw = row["keyword"]
            step_id = f"search_{kw.replace(' ', '_')}_{idx}"

            # 搜索步骤
            self.steps.append({
                "id": step_id,
                "tool": "search_news_by_keywords",
                "inputs": {
                    "keywords": [kw],  # 工具期望列表
                    "apis": selected_apis,
                    "limit": int(row.get("limit", 50)),
                    "category": row.get("category") or None,
                    "from": row.get("from") or None,
                    "to": row.get("to") or None,
                    "sortby": row.get("sortby") or None
                },
                "output": f"results_{idx}"
            })

            # 事件提取步骤
            self.steps.append({
                "id": f"extract_{kw.replace(' ', '_')}_{idx}",
                "tool": "batch_process_news",
                "inputs": {"news_list": f"$results_{idx}"},
                "output": f"extracted_events_{idx}"
            })

            # 临时保存步骤
            self.steps.append({
                "id": f"save_events_{kw.replace(' ', '_')}_{idx}",
                "tool": "save_extracted_events_tmp",
                "inputs": {"events": f"$extracted_events_{idx}"},
                "output": f"events_path_{idx}"
            })

            # 持久化步骤
            self.steps.append({
                "id": f"persist_{kw.replace(' ', '_')}_{idx}",
                "tool": "persist_expanded_news_tmp",
                "inputs": {"expanded_news": f"$results_{idx}"},
                "output": f"persist_result_{idx}"
            })

    def add_graph_update_steps(self, active_tasks, append_only_mode, allow_append_forms):
        """添加图谱更新步骤"""
        all_extracted_keys = [f"$extracted_events_{i}" for i in range(len(active_tasks))]

        if append_only_mode:
            self.steps.append({
                "id": "append_graph_from_expansion",
                "tool": "append_only_update_graph",
                "inputs": {
                    "events_list": all_extracted_keys,
                    "allow_append_original_forms": allow_append_forms
                },
                "output": "kg_update_result"
            })
        else:
            self.steps.append({
                "id": "update_graph_from_expansion",
                "tool": "update_graph_data",
                "inputs": {"events_list": all_extracted_keys},
                "output": "kg_update_result"
            })

    def add_final_steps(self, active_tasks):
        """添加最终步骤（刷新和报告）"""
        all_extracted_keys = [f"$extracted_events_{i}" for i in range(len(active_tasks))]

        # 刷新图谱
        self.steps.append({
            "id": "refresh_kg_after_expansion",
            "tool": "refresh_knowledge_graph",
            "inputs": {},
            "output": "kg_refresh_result"
        })

        # 生成报告
        self.steps.append({
            "id": "report_expansion",
            "tool": "generate_markdown_report",
            "inputs": {
                "events_list": all_extracted_keys,
                "title": "Expansion Extracted Events Report"
            },
            "output": "expansion_report_md"
        })

    def build(self, selected_apis, active_tasks, append_only_mode, allow_append_forms):
        """构建完整的扩展流水线"""
        self.steps = []

        # 添加搜索相关步骤
        self.add_search_steps(selected_apis, active_tasks)

        # 添加图谱更新步骤
        self.add_graph_update_steps(active_tasks, append_only_mode, allow_append_forms)

        # 添加最终步骤
        self.add_final_steps(active_tasks)

        return {
            "name": "Knowledge Expansion Batch",
            "steps": self.steps
        }

def create_expansion_pipeline_steps(selected_apis, active_tasks, append_only_mode, allow_append_forms):
    """创建拓展流水线步骤（兼容性接口）"""
    builder = ExpansionPipelineBuilder()
    return builder.build(selected_apis, active_tasks, append_only_mode, allow_append_forms)

def render_maintenance_tab():
    """渲染图谱维护标签页"""
    st.header("🕸️ Graph Maintenance")

    # 获取临时数据统计
    maintenance_stats = get_maintenance_stats()

    # 渲染统计指标
    render_maintenance_metrics(maintenance_stats)

    # 渲染临时数据预览
    render_temp_data_preview(maintenance_stats)

    # 渲染维护操作表单
    render_maintenance_form()

def render_maintenance_form():
    """渲染维护操作表单"""
    with st.form("maintenance_form"):
        # 去重参数
        render_deduplication_params()

        # 清理参数
        render_cleaning_params()

        # 临时数据导入参数
        render_temp_data_import_params()

        # 执行按钮
        submitted = st.form_submit_button("🚀 Run Maintenance", type="primary", use_container_width=True)

    # 处理表单提交
    if submitted:
        execute_maintenance_pipeline()

def render_deduplication_params():
    """渲染去重参数"""
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Deduplication")
        st.checkbox("Strict Mode", True)
        st.slider("Similarity", 0.5, 1.0, 0.9)
def render_cleaning_params():
    """渲染清理参数"""
    with st.container():
        st.subheader("Cleaning")
st.checkbox("Remove Isolated Nodes", key="unknown_checkbox_auto_8")
def render_temp_data_import_params():
    """渲染临时数据导入参数"""
    st.subheader("导入 tmp 抽取结果")
    st.checkbox("刷新前先追加 tmp/extracted_events_*.jsonl", value=True)
    st.number_input("最多读取文件数（0=全部）", min_value=0, value=0, step=1)
    st.checkbox("追加旧实体 original_forms（追加模式）", value=True)
def execute_maintenance_pipeline():
    """执行维护流水线"""
    # 这里需要获取表单数据，但由于Streamlit的限制，我们需要在表单内部处理
    # 为了简化，这里使用默认值
    pipeline_def = {
        "name": "Graph Maintenance",
        "steps": [
            {
                "id": "append_tmp_events",
                "tool": "append_tmp_extracted_events",
                "inputs": {
                    "max_files": 0,
                    "allow_append_original_forms": True
                },
                "output": "tmp_append_result"
            },
            {
                "id": "refresh_kg",
                "tool": "refresh_knowledge_graph",
                "inputs": {},
                "output": "status"
            }
        ]
    }
    execute_pipeline(pipeline_def)

    # 清理临时缓存文件
    cleanup_temp_cache()

def cleanup_temp_cache():
    """清理临时缓存文件"""
    try:
        data_dir = ROOT_DIR / "data"
        entities_tmp_file = data_dir / "tmp" / "entities_tmp.json"
        events_tmp_file = data_dir / "tmp" / "abstract_to_event_map_tmp.json"

        for p in [entities_tmp_file, events_tmp_file]:
            if p.exists():
                p.unlink()
        st.cache_data.clear()
        st.success("已清理临时缓存文件")
    except Exception as e:
        st.warning(f"清理缓存失败: {e}")

def get_maintenance_stats():
    """获取维护相关的统计数据"""
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

    @st.cache_data(ttl=60)
    def list_news_files(base: Path, pattern: str):
        files = sorted(base.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(f) for f in files]

    extracted_files = list_extracted_files(extracted_dir)
    deduped_files = list_news_files(deduped_dir, "*.jsonl")
    raw_files = list_news_files(raw_dir, "*.jsonl")

    return {
        'entities_tmp': entities_tmp,
        'events_tmp': events_tmp,
        'extracted_files': extracted_files,
        'deduped_files': deduped_files,
        'raw_files': raw_files
    }

def render_maintenance_metrics(stats):
    """渲染维护统计指标"""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("临时实体（缓存条数）", len(stats['entities_tmp']))
    c2.metric("临时事件（缓存条数）", len(stats['events_tmp']))
    c3.metric("提取结果文件数", len(stats['extracted_files']))
    c4.metric("去重新闻文件数", len(stats['deduped_files']))
    c5.metric("原始新闻文件数", len(stats['raw_files']))

def render_temp_data_preview(stats):
    """渲染临时数据预览"""
    with st.expander("查看临时实体 / 事件示例", expanded=False):
        render_entities_preview(stats['entities_tmp'])
        render_events_preview(stats['events_tmp'])
        render_files_preview("提取结果文件", stats['extracted_files'])
        render_files_preview("去重新闻文件", stats['deduped_files'])
        render_files_preview("原始新闻文件", stats['raw_files'])

def render_entities_preview(entities_tmp):
    """渲染实体预览"""
    if entities_tmp:
        df_ent = pd.DataFrame([
            {
                "name": k,
                "first_seen": v.get("first_seen", ""),
                "sources": ",".join([
                    (s.get("name") or s.get("id") or s.get("url") or str(s))
                    if isinstance(s, dict) else str(s)
                    for s in v.get("sources", [])
                ])[:80],
            }
            for k, v in list(entities_tmp.items())[:50]
        ])
        st.write("临时实体（最多50条预览）")
        st.dataframe(df_ent, use_container_width=True)
    else:
        st.info("暂无临时实体数据")

def render_events_preview(events_tmp):
    """渲染事件预览"""
    if events_tmp:
        df_evt = pd.DataFrame([
            {
                "abstract": k,
                "first_seen": v.get("first_seen", ""),
                "entities": ",".join(v.get("entities", []))[:80]
            }
            for k, v in list(events_tmp.items())[:50]
        ])
        st.write("临时事件（最多50条预览）")
        st.dataframe(df_evt, use_container_width=True)
    else:
        st.info("暂无临时事件数据")

def render_files_preview(title, files):
    """渲染文件列表预览"""
    if files:
        st.write(f"{title}（最新5个）")
        st.table({"path": files[:5]})
    else:
        st.info(f"暂无{title.lower()}")


def render_template_management():
    """渲染Pipeline模板管理"""
    with st.expander("📚 Pipeline模板", expanded=False):
        template_name = st.text_input("模板名称", key="template_name")
        col_save_template, col_load_template = st.columns(2)

        with col_save_template:
            if st.button("💾 保存模板", use_container_width=True, disabled=not template_name or not st.session_state.pipeline_steps, key="pipeline_save_template"):
                save_pipeline_template(template_name)

        with col_load_template:
            render_template_loader()

def save_pipeline_template(template_name):
    """保存Pipeline模板"""
    try:
        template_data = {
            "name": template_name,
            "steps": st.session_state.pipeline_steps.copy(),
            "created_at": datetime.now().isoformat(),
            "tool_count": len(st.session_state.pipeline_steps)
        }

        # 保存到session_state (可以扩展到文件系统)
        if "pipeline_templates" not in st.session_state:
            st.session_state.pipeline_templates = {}

        st.session_state.pipeline_templates[template_name] = template_data
        st.success(f"模板 '{template_name}' 已保存！")
    except Exception as e:
        st.error(f"保存失败: {e}")

def render_template_loader():
    """渲染模板加载器"""
    if "pipeline_templates" in st.session_state and st.session_state.pipeline_templates:
        template_options = list(st.session_state.pipeline_templates.keys())
        selected_template = st.selectbox("选择模板", [""] + template_options, key="load_template")

        if selected_template and st.button("📂 加载模板", use_container_width=True, key="pipeline_load_template"):
            load_pipeline_template(selected_template)
    else:
        st.caption("暂无保存的模板")

def load_pipeline_template(selected_template):
    """加载Pipeline模板"""
    try:
        template_data = st.session_state.pipeline_templates[selected_template]
        st.session_state.pipeline_steps = template_data["steps"].copy()
        st.success(f"模板 '{selected_template}' 已加载 ({template_data['tool_count']} 个步骤)")
        st.rerun()
    except Exception as e:
        st.error(f"加载失败: {e}")

def render_pipeline_toolbar():
    """渲染Pipeline工具栏"""
    c_add, c_clear, c_reorder = st.columns([2, 1, 1])
    with c_add:
        add_new_pipeline_step()

    with c_clear:
        clear_all_pipeline_steps()

    with c_reorder:
        reorder_pipeline_steps()

def add_new_pipeline_step():
    """添加新的Pipeline步骤"""
    tools = FunctionRegistry.get_all_tools()
    selected_tool = st.selectbox("Select Tool", list(tools.keys()), label_visibility="collapsed")
    if st.button("➕ Add Step", use_container_width=True, key="pipeline_add_step"):
        st.session_state.pipeline_steps.append({
            "id": f"step_{len(st.session_state.pipeline_steps) + 1}",
            "tool": selected_tool,
            "inputs": {}
        })
        st.rerun()

def clear_all_pipeline_steps():
    """清空所有Pipeline步骤"""
    if st.button("🧹 Clear All", use_container_width=True, disabled=not st.session_state.pipeline_steps, key="pipeline_clear_all"):
        st.session_state.pipeline_steps = []
        st.success("已清空所有步骤")
        st.rerun()

def reorder_pipeline_steps():
    """重新编号Pipeline步骤"""
    if len(st.session_state.pipeline_steps) > 1:
        if st.button("🔄 Reorder", use_container_width=True, key="pipeline_reorder_steps"):
            # 简单的重新编号
            for i, step in enumerate(st.session_state.pipeline_steps):
                step["id"] = f"step_{i + 1}"
            st.success("步骤已重新编号")
            st.rerun()

def render_pipeline_steps():
    """渲染Pipeline步骤列表"""
    if not st.session_state.pipeline_steps:
        st.info("No steps added. Select a tool to start.")
        return

    tools = FunctionRegistry.get_all_tools()

    for i, step in enumerate(st.session_state.pipeline_steps):
        tool_name = step["tool"]
        tool_meta = tools.get(tool_name, {})

        with st.expander(f"Step {i+1}: {tool_name}", expanded=False):
            render_step_header(i, step)
            render_step_content(i, step, tool_meta)

def render_step_header(i, step):
    """渲染步骤头部（ID和控制按钮）"""
    c_id, c_move, c_del = st.columns([3, 1.5, 1])
    step["id"] = c_id.text_input("ID", step["id"], key=f"id_{i}")

    # 步骤移动按钮
    with c_move:
        render_step_move_buttons(i)

    # 删除按钮
    if c_del.button("🗑️", key=f"del_{i}"):
        st.session_state.pipeline_steps.pop(i)
        st.rerun()

def render_step_move_buttons(i):
    """渲染步骤移动按钮"""
    col_up, col_down = st.columns(2)
    with col_up:
        if st.button("⬆️", key=f"up_{i}", disabled=i==0):
            # 上移步骤
            st.session_state.pipeline_steps[i], st.session_state.pipeline_steps[i-1] = \
            st.session_state.pipeline_steps[i-1], st.session_state.pipeline_steps[i]
            st.rerun()
    with col_down:
        if st.button("⬇️", key=f"down_{i}", disabled=i==len(st.session_state.pipeline_steps)-1):
            # 下移步骤
            st.session_state.pipeline_steps[i], st.session_state.pipeline_steps[i+1] = \
            st.session_state.pipeline_steps[i+1], st.session_state.pipeline_steps[i]
            st.rerun()

def render_step_content(i, step, tool_meta):
    """渲染步骤内容（描述、参数、输出）"""
    st.caption(tool_meta.get("description", ""))

    # 参数编辑区
    params = tool_meta.get("parameters", {})
    if params:
        for p_name, p_info in params.items():
            render_input_field(i, p_name, p_info, step.get("inputs", {}), step)
    else:
        st.info("无参数")

    step["output"] = st.text_input("Output to ($var)", step.get("output", ""), key=f"out_{i}")
def render_custom_builder():
    """渲染自定义Pipeline构建器"""
    st.header("🛠️ Custom Pipeline Builder")

    # Pipeline模板管理
    col_template, _ = st.columns([1, 3])
    with col_template:
        render_template_management()

    col_builder, col_preview = st.columns([1.5, 1])

    with col_builder:
        # 工具栏
        render_pipeline_toolbar()

        # 步骤编辑
        render_pipeline_steps()

    with col_preview:
        # Pipeline历史记录
        if "pipeline_history" not in st.session_state:
            st.session_state.pipeline_history = []

        with st.expander("📚 执行历史", expanded=False):
            if st.session_state.pipeline_history:
                for i, hist in enumerate(reversed(st.session_state.pipeline_history[-5:])):  # 最近5个
                    col_time, col_status = st.columns([2, 1])
                    with col_time:
                        st.caption(f"{hist['timestamp']} - {hist['name']}")
                    with col_status:
                        if hist['status'] == 'success':
                            st.caption("✅ 成功")
                        else:
                            st.caption("❌ 失败")
            else:
                st.caption("暂无执行历史")

        st.subheader("Preview")
        pipeline_def = {"name": "Custom Pipeline", "steps": st.session_state.pipeline_steps}
        st.code(yaml.dump(pipeline_def, sort_keys=False), language="yaml")

        # Pipeline验证
        if st.button("🔍 验证Pipeline", use_container_width=True, disabled=not st.session_state.pipeline_steps, key="pipeline_validate"):
            validation_errors = []

            # 检查步骤完整性
            for i, step in enumerate(st.session_state.pipeline_steps):
                if not step.get("tool"):
                    validation_errors.append(f"步骤 {i+1}: 缺少工具选择")
                if not step.get("id"):
                    validation_errors.append(f"步骤 {i+1}: 缺少步骤ID")

                # 检查必需参数
                tool_meta = tools.get(step.get("tool", ""))
                if tool_meta:
                    params = tool_meta.get("parameters", [])
                    for param in params:
                        if param.get("required", False):
                            param_name = param["name"]
                            if param_name not in step.get("inputs", {}):
                                validation_errors.append(f"步骤 {i+1} ({step.get('tool')}): 缺少必需参数 '{param_name}'")

            if validation_errors:
                st.error("Pipeline验证失败:")
                for error in validation_errors:
                    st.write(f"• {error}")
            else:
                st.success("✅ Pipeline验证通过！所有步骤配置正确。")

        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True, disabled=not st.session_state.pipeline_steps, key="pipeline_run"):
            execute_pipeline(pipeline_def)

def render_snapshots_tab():
    st.header("📸 Knowledge Graph Snapshots")
    st.caption("生成/查看可视化快照（kg_visual.json / kg_visual_timeline.json）")
    if st.button("生成快照", type="primary", key="snapshots_generate"):
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


def get_tool_usage_stats():
    """获取工具使用统计"""
    tool_stats = {}
    if "pipeline_steps" in st.session_state and st.session_state.pipeline_steps:
        for step in st.session_state.pipeline_steps:
            tool_name = step.get("tool", "")
            if tool_name:
                tool_stats[tool_name] = tool_stats.get(tool_name, 0) + 1
    return tool_stats

def render_tool_usage_stats(tool_stats):
    """渲染工具使用统计"""
    if tool_stats:
        st.info(f"📊 当前Pipeline使用了 {len(tool_stats)} 种不同工具，总计 {sum(tool_stats.values())} 个步骤")

        # 工具使用频率
        sorted_tools = sorted(tool_stats.items(), key=lambda x: x[1], reverse=True)
        with st.expander("🔥 当前Pipeline工具使用统计", expanded=False):
            for tool_name, count in sorted_tools:
                st.write(f"• **{tool_name}**: {count} 次使用")

def get_tool_categories():
    """获取工具分类配置"""
    return {
        "Data Fetch": ["fetch", "search", "scrape", "crawl"],
        "Extraction": ["extract", "process", "parse", "llm"],
        "Graph Ops": ["graph", "update", "refresh", "merge", "kg", "node", "edge"],
        "Reporting": ["report", "markdown", "summary", "export"],
        "Utility": ["save", "load", "tmp", "debug", "test"],
    }

def categorize_tools(all_tools):
    """对工具进行分类"""
    CATEGORY_ORDER = get_tool_categories()

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

    return categorized

def render_tool_explorer_tab():
    """渲染工具探索器标签页"""
    st.header("Tool Explorer")
    st.caption("自动发现所有注册工具 · 支持搜索、预览、复制、一键执行")

    # 1. 自动加载所有真实工具（核心！）
    all_tools = FunctionRegistry.get_all_tools()  # <-- 你的真实注册表
    if not all_tools:
        st.warning("未检测到已注册的工具，请检查 FunctionRegistry")
        return

    # 工具使用统计
    tool_stats = get_tool_usage_stats()
    render_tool_usage_stats(tool_stats)

    # 对工具进行分类
    categorized = categorize_tools(all_tools)

    # 智能推荐
    render_smart_recommendations(all_tools, categorized)

    # 搜索和批量操作
    search_query = render_search_and_batch_tools(categorized)

    # 过滤工具
    if search_query:
        categorized = filter_tools_by_search(categorized, search_query)

    # 渲染工具网格
    render_tool_grid(categorized)

def render_smart_recommendations(all_tools, categorized):
    """渲染智能推荐"""
    current_tools = set()
    if "pipeline_steps" in st.session_state:
        current_tools = {step.get("tool") for step in st.session_state.pipeline_steps if step.get("tool")}

    if current_tools:
        # 基于当前Pipeline推荐相关工具
        recommendations = []
        for tool_name, meta in all_tools.items():
            if tool_name not in current_tools:
                category = meta.get("category") or get_category_from_categorized(categorized, tool_name)
                # 推荐相同类别的工具
                current_categories = {get_category_from_categorized(categorized, step.get("tool", "")) for step in st.session_state.pipeline_steps if step.get("tool")}
                if category in current_categories:
                    recommendations.append((tool_name, category))

        if recommendations:
            with st.expander(f"💡 智能推荐 ({len(recommendations)})", expanded=False):
                for tool_name, category in recommendations[:5]:  # 最多显示5个推荐
                    st.write(f"• **{tool_name}** ({category})")

def get_category_from_categorized(categorized, tool_name):
    """从已分类的工具中获取类别"""
    for category, tools in categorized.items():
        if any(name == tool_name for name, _ in tools):
            return category
    return "Other"

def render_search_and_batch_tools(categorized):
    """渲染搜索和批量操作工具栏"""
    col_search, col_batch = st.columns([3, 1])
    with col_search:
        # 3. 搜索框
        search = st.text_input("Search Tools", placeholder="输入工具名或描述关键词...", key="tool_search")

    with col_batch:
        # 批量复制功能
        if st.button("📋 批量复制选中", use_container_width=True, disabled=len(categorized) == 0, key="tool_batch_copy_selected"):
            handle_batch_copy(categorized)

    return search

def handle_batch_copy(categorized):
    """处理批量复制功能"""
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = []

    if st.session_state.selected_tools:
        # 生成批量YAML
        batch_steps = []
        all_tools = FunctionRegistry.get_all_tools()

        for tool_name in st.session_state.selected_tools:
            if tool_name in all_tools:
                step_yaml = {
                    "id": tool_name,
                    "tool": tool_name,
                    "inputs": {},
                    "output": f"{tool_name}_result"
                }
                batch_steps.append(step_yaml)

        if batch_steps:
            batch_yaml = yaml.dump({"steps": batch_steps}, sort_keys=False, allow_unicode=True)
            st.code(batch_yaml, language="yaml")
            st.success(f"已生成 {len(batch_steps)} 个工具的批量配置")
    else:
        st.warning("请先选择要批量复制的工具")

    st.session_state.selected_tools = []  # 重置选择

def filter_tools_by_search(categorized, search):
    """根据搜索条件过滤工具"""
    if not search:
        return categorized

    filtered = {}
    for cat, tools in categorized.items():
        matched = []
        for name, meta in tools:
            if (search.lower() in name.lower() or
                (meta.get("description") and search.lower() in meta.get("description", "").lower())):
                matched.append((name, meta))
        if matched:
            filtered[cat] = matched
    return filtered

def render_tool_grid(categorized):
    """渲染工具网格"""
    # 4. 主渲染区 - 响应式卡片流
    for category, tools in categorized.items():
        with st.expander(f"**{category}** · {len(tools)} tools", expanded=True):
            cols = st.columns(3, gap="medium")  # 每行3个卡片，可改成2或4
            for idx, (tool_name, meta) in enumerate(tools):
                with cols[idx % 3]:
                    render_tool_card(tool_name, meta, idx)

def render_tool_card(tool_name, meta, idx):
    """渲染单个工具卡片"""
    with st.container(border=True):
        # 批量选择复选框
        selected = st.checkbox("", key=f"select_{tool_name}_{idx}", label_visibility="collapsed")
        if selected:
            if "selected_tools" not in st.session_state:
                st.session_state.selected_tools = []
            if tool_name not in st.session_state.selected_tools:
                st.session_state.selected_tools.append(tool_name)
        elif "selected_tools" in st.session_state and tool_name in st.session_state.selected_tools:
            st.session_state.selected_tools.remove(tool_name)

        st.markdown(f"**`{tool_name}`**")

        desc = meta.get("description") or "No description"
        st.caption(desc)

        # 参数表单（可编辑）
        params = meta.get("parameters", {})
        if params:
            render_tool_parameters(tool_name, meta, idx)
        else:
            st.info("No parameters")

        # 底部标签
        render_tool_tags(tool_name, meta)

def render_tool_parameters(tool_name, meta, idx):
    """渲染工具参数表单"""
    with st.form(key=f"form_{tool_name}_{idx}", clear_on_submit=False, border=False):
        inputs = {}
        for p_name, p_info in meta.get("parameters", {}).items():
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
            execute_tool_debug(tool_name, inputs)

        # 一键复制为 Pipeline Step
        if copy_step:
            copy_tool_as_step(tool_name, inputs)

def execute_tool_debug(tool_name, inputs):
    """执行工具调试"""
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

def copy_tool_as_step(tool_name, inputs):
    """复制工具为Pipeline步骤"""
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

def render_tool_tags(tool_name, meta):
    """渲染工具标签"""
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

