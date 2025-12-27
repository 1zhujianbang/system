"""
Reporting tools: 报告生成
"""
from typing import Any
from ...infra.registry import register_tool
from ...core import StandardEventPipeline


@register_tool(
    name="generate_markdown_report",
    description="根据事件列表生成 Markdown 格式的简报",
    category="Reporting"
)
async def generate_markdown_report(events_list: Any, title: str = "Market Analysis Report") -> str:
    """
    生成 Markdown 报告
    - 使用标准事件数据管道进行数据处理
    """
    # 使用标准事件数据管道
    pipeline = StandardEventPipeline()
    pipeline_result = await pipeline.execute(events_list)

    # 获取处理后的数据
    events = pipeline_result.get("transformation", [])
    if not events:
        return f"# {title}\n\nNo events found."
        
    lines = [f"# {title}", ""]
    lines.append(f"**Total Events Extracted:** {len(events)}")
    lines.append("")
    
    # 简单的按时间排序
    sorted_events = sorted(events, key=lambda x: x.get('published_at') or "", reverse=True)
    
    for i, ev in enumerate(sorted_events, 1):
        abstract = ev.get('abstract', 'No Title')
        summary = ev.get('event_summary', '')
        entities = ', '.join(ev.get('entities', []))
        source = ev.get('source', 'Unknown')
        ts = ev.get('published_at', 'Unknown Time')
        
        lines.append(f"### {i}. {abstract}")
        lines.append(f"- **时间:** {ts}")
        lines.append(f"- **来源:** {source}")
        lines.append(f"- **实体:** {entities}")
        lines.append(f"- **摘要:** {summary}")
        lines.append("---")
        
    return "\n".join(lines)




