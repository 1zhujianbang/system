# src/agents/agent1.py
"""
Agent1: 新闻实体提取代理

该代理负责从新闻中提取实体和事件信息。
核心逻辑已重构到 functions/extraction.py 中实现解耦合。
"""

from ..functions.extraction import process_news_pipeline
from ..functions.graph_ops import refresh_knowledge_graph
from ..core import ConfigManager, LoggerManager, tools
import threading
import asyncio


async def process_news_stream(max_workers: int = 3, rate_limit_per_sec: float = 1.0):
    """
    Agent1主流程：处理新闻管道

    Args:
        max_workers: 最大并发数
        rate_limit_per_sec: 每秒速率限制
    """
    tools.log("🚀 启动 Agent1：新闻实体提取管道")

    # 调用functions中的处理逻辑
    result = await process_news_pipeline(
        max_workers=max_workers,
        rate_limit_per_sec=rate_limit_per_sec
    )

    tools.log(f"✅ Agent1完成：处理了 {result.get('processed_count', 0)} 条新闻，来自 {result.get('files_processed', 0)} 个文件")

    # 如果有新数据，触发知识图谱刷新
    if result.get('processed_count', 0) > 0:
        try:
            # 在后台线程中刷新知识图谱
            def refresh_async():
                refresh_knowledge_graph()

            with tools._refresh_lock:
                thread = threading.Thread(target=refresh_async, daemon=True)
                thread.start()
                tools.log("🔄 已启动知识图谱刷新线程")
        except Exception as e:
            tools.log(f"⚠️ 启动知识图谱刷新失败: {e}")
    else:
        tools.log("📭 未处理任何新闻，跳过知识图谱刷新")


def run_agent1():
    """
    同步运行Agent1（供命令行调用）
    """
    import asyncio

    # 获取配置
    config_manager = ConfigManager()
    max_workers = config_manager.get_concurrency_limit("agent1_config")
    rate_limit = config_manager.get_rate_limit("agent1_config")

    # 运行异步处理
    asyncio.run(process_news_stream(
        max_workers=max_workers,
        rate_limit_per_sec=rate_limit
    ))