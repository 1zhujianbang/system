import streamlit as st
import sys
from pathlib import Path
import yaml

# 添加项目根目录到 path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception as e:
            st.error(f"读取配置失败: {e}")
            return {}
    return {}


st.set_page_config(page_title="System Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ 系统设置")
st.caption("编辑 Agent 并发/限速等核心参数（写入 config/config.yaml）")

cfg = load_config()
agent1 = cfg.get("agent1_config", {}) or {}
agent2 = cfg.get("agent2_config", {}) or {}
agent3 = cfg.get("agent3_config", {}) or {}

tabs = st.tabs(["Agent1", "Agent2", "Agent3"])

with st.form("system_settings"):
    with tabs[0]:
        st.subheader("Agent1（抓取+抽取）")
        a1_workers = st.number_input(
            "max_workers",
            1, 64,
            int(agent1.get("max_workers", 1)),
            help="并发线程数，控制同时处理的新闻数。"
        )
        a1_qps = st.number_input(
            "rate_limit_per_sec",
            0.1, 20.0,
            float(agent1.get("rate_limit_per_sec", 1.0)), 0.1,
            help="LLM 请求速率上限（次/秒），避免超限。"
        )
        a1_dedupe = st.number_input(
            "dedupe_threshold",
            1, 10,
            int(agent1.get("dedupe_threshold", 1)),
            help="数值越小越严格（更少误杀，但可能漏掉近似重复）；越大则更宽松（更多近似被视为重复，但风险误杀不同新闻）。"
        )

    with tabs[1]:
        st.subheader("Agent2（拓展搜索）")
        a2_workers = st.number_input(
            "A2 max_workers",
            1, 64,
            int(agent2.get("max_workers", 1)),
            help="并发处理拓展新闻的任务数。"
        )
        a2_qps = st.number_input(
            "A2 rate_limit_per_sec",
            0.1, 20.0,
            float(agent2.get("rate_limit_per_sec", 1.0)), 0.1,
            help="LLM 抽取速率上限（次/秒）。"
        )

    with tabs[2]:
        st.subheader("Agent3（知识图谱压缩）")
        g3_e_workers = st.number_input(
            "entity_max_workers",
            1, 16,
            int(agent3.get("entity_max_workers", 1)),
            help="实体压缩并行 worker 数。"
        )
        g3_ent_batch = st.number_input(
            "entity_batch_size",
            10, 500,
            int(agent3.get("entity_batch_size", 10)),
            help="单批处理的实体数量。"
        )
        g3_ent_sim = st.number_input(
            "entity_precluster_similarity",
            0.1, 1.0,
            float(agent3.get("entity_precluster_similarity", 0.1)), 0.01,
            help="实体预聚类相似度阈值。"
        )
        g3_ent_limit = st.number_input(
            "entity_precluster_limit",
            10, 2000,
            int(agent3.get("entity_precluster_limit", 10)),
            help="实体预聚类上限（防止批次过大）。"
        )
        g3_ev_workers = st.number_input(
            "event_max_workers",
            1, 16,
            int(agent3.get("event_max_workers", 1)),
            help="事件压缩并行 worker 数。"
        )
        g3_ev_batch = st.number_input(
            "event_batch_size",
            5, 200,
            int(agent3.get("event_batch_size", 5)),
            help="单批处理的事件数量。"
        )
        g3_ev_sim = st.number_input(
            "event_precluster_similarity",
            0.1, 1.0,
            float(agent3.get("event_precluster_similarity", 0.1)), 0.01,
            help="事件预聚类相似度阈值。"
        )
        g3_ev_limit = st.number_input(
            "event_precluster_limit",
            10, 2000,
            int(agent3.get("event_precluster_limit", 10)),
            help="事件预聚类上限（防止批次过大）。"
        )
        g3_rate = st.number_input(
            "rate_limit_per_sec",
            0.1, 20.0,
            float(agent3.get("rate_limit_per_sec", 0.1)), 0.1,
            help="LLM 调用速率上限（次/秒）。"
        )
        g3_bucket_days = st.number_input(
            "event_bucket_days",
            1, 90,
            int(agent3.get("event_bucket_days", 1)),
            help="事件分桶的时间跨度（天）。"
        )
        g3_bucket_overlap = st.number_input(
            "event_bucket_entity_overlap",
            0, 10,
            int(agent3.get("event_bucket_entity_overlap", 0)),
            help="事件分桶间实体重叠阈值。"
        )
        g3_bucket_max = st.number_input(
            "event_bucket_max_size",
            10, 1000,
            int(agent3.get("event_bucket_max_size", 10)),
            help="单桶事件最大条数，防止过大。"
        )
        g3_max_summary = st.number_input(
            "max_summary_chars",
            50, 2000,
            int(agent3.get("max_summary_chars", 50)),
            help="摘要截断长度，避免 prompt 过长。"
        )
        g3_ev_per_entity = st.number_input(
            "entity_evidence_per_entity",
            0, 10,
            int(agent3.get("entity_evidence_per_entity", 0)),
            help="为每个实体采样的事件证据条数。"
        )
        g3_ev_max_chars = st.number_input(
            "entity_evidence_max_chars",
            50, 2000,
            int(agent3.get("entity_evidence_max_chars", 50)),
            help="单条证据的最大字符数。"
        )

    submitted = st.form_submit_button("💾 保存配置", type="primary", use_container_width=True)

    if submitted:
        try:
            cfg["agent1_config"] = {
                "max_workers": int(a1_workers),
                "rate_limit_per_sec": float(a1_qps),
                "dedupe_threshold": int(a1_dedupe),
            }
            cfg["agent2_config"] = {
                "max_workers": int(a2_workers),
                "rate_limit_per_sec": float(a2_qps),
            }
            cfg["agent3_config"] = {
                "entity_batch_size": int(g3_ent_batch),
                "event_batch_size": int(g3_ev_batch),
                "event_bucket_days": int(g3_bucket_days),
                "event_bucket_entity_overlap": int(g3_bucket_overlap),
                "event_bucket_max_size": int(g3_bucket_max),
                "event_precluster_similarity": float(g3_ev_sim),
                "event_precluster_limit": int(g3_ev_limit),
                "entity_precluster_similarity": float(g3_ent_sim),
                "entity_precluster_limit": int(g3_ent_limit),
                "max_summary_chars": int(g3_max_summary),
                "entity_max_workers": int(g3_e_workers),
                "event_max_workers": int(g3_ev_workers),
                "rate_limit_per_sec": float(g3_rate),
                "entity_evidence_per_entity": int(g3_ev_per_entity),
                "entity_evidence_max_chars": int(g3_ev_max_chars),
            }
            CONFIG_PATH.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            st.success("配置已保存到 config/config.yaml")
        except Exception as e:
            st.error(f"保存失败: {e}")


