# src/agents/agent3.py
"""
Agent3: 知识图谱管理代理

该代理负责管理和更新知识图谱，包括实体和事件的去重、合并等。
核心逻辑已重构到 functions/graph_ops.py 中实现解耦合。
"""

from ..functions.graph_ops import KnowledgeGraph
from ..core import ConfigManager, tools
import argparse


def refresh_graph():
    """
    Agent3主流程：刷新知识图谱
    """
    tools.log("🚀 启动 Agent3：知识图谱刷新")
    kg = KnowledgeGraph()
    result = kg.refresh_graph()
    tools.log("✅ Agent3完成：知识图谱刷新完毕")
    return result


def append_only_update_graph(events_list, default_source: str = "auto_pipeline", allow_append_original_forms: bool = True):
    """
    Agent3追加更新：只追加新事件/实体到现有图谱
    """
    tools.log("🚀 启动 Agent3：追加更新知识图谱")
    kg = KnowledgeGraph()
    result = kg.append_only_update(events_list, default_source, allow_append_original_forms)
    tools.log(f"✅ Agent3追加更新完成：新增实体 {result.get('added_entities', 0)}，新增事件 {result.get('added_events', 0)}")
    return result


def build_graph():
    """
    Agent3构建图谱：构建知识图谱
    """
    tools.log("🚀 启动 Agent3：构建知识图谱")
    kg = KnowledgeGraph()
    result = kg.build_graph()
    tools.log("✅ Agent3构建完成")
    return result


def run_agent3():
    """
    同步运行Agent3（供命令行调用）
    """
    args = parse_args()
    if args.action == "refresh":
        refresh_graph()
    elif args.action == "build":
        build_graph()
    else:
        tools.log("❌ 未知操作，请使用 --action refresh 或 --action build")


def parse_args():
    parser = argparse.ArgumentParser(description="Agent3 知识图谱管理")
    parser.add_argument("--action", choices=["refresh", "build"], default="refresh",
                       help="执行操作：refresh(刷新图谱) 或 build(构建图谱)")
    return parser.parse_args()