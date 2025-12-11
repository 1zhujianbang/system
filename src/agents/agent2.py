# src/agents/agent2.py
"""
Agent2: 实体拓展新闻代理

该代理根据现有实体搜索相关新闻并提取新信息。
核心逻辑已重构到 functions/data_fetch.py 中实现解耦合。
"""

import argparse
import asyncio
from ..functions.data_fetch import expand_news_by_recent_entities
from ..core import ConfigManager, tools


async def main(args=None):
    """
    Agent2主函数：根据实体拓展新闻

    Args:
        args: 命令行参数
    """
    tools.log("🚀 启动 Agent2：实体拓展新闻...")

    # 获取配置参数
    entity_limit = args.entity_limit if args else 1
    time_window_days = args.time_window_days if args else 30
    limit_per_entity = args.limit_per_entity if args else 120
    full_search = args.full_search if args else False

    # 获取并发和速率配置
    config_manager = ConfigManager()
    max_workers = config_manager.get_concurrency_limit("agent2_config")
    rate_limit = config_manager.get_rate_limit("agent2_config")

    # 调用functions中的拓展逻辑
    result = await expand_news_by_recent_entities(
        entity_limit=entity_limit,
        time_window_days=time_window_days,
        limit_per_entity=limit_per_entity,
        full_search=full_search,
        rate_limit=rate_limit,
        max_workers=max_workers
    )

    tools.log(f"🎉 Agent2任务完成：处理了 {result.get('processed_count', 0)} 条新闻，搜索到 {result.get('expanded_news_count', 0)} 条相关新闻")


def run_agent2():
    """
    同步运行Agent2（供命令行调用）
    """
    args = parse_args()
    asyncio.run(main(args))


def parse_args():
    parser = argparse.ArgumentParser(description="Agent2 实体拓展新闻")
    parser.add_argument("--keywords", "-k", nargs="+", help="指定实体关键词列表，替代最近实体")
    parser.add_argument("--entity-limit", type=int, default=1, help="从最近实体库选择的数量（未指定关键词时生效）")
    parser.add_argument("--time-window-days", type=int, default=30, help="最近实体时间窗口 / 搜索时间窗口（天）")
    parser.add_argument("--limit-per-entity", type=int, default=120, help="每个实体搜索新闻数量上限")
    parser.add_argument("--full-search", action="store_true", help="是否全面检索至2020年")
    return parser.parse_args()