from typing import List, Dict, Any
from ..core.registry import register_tool

@register_tool(
    name="generate_markdown_report",
    description="根据事件列表生成 Markdown 格式的简报",
    category="Reporting"
)
def generate_markdown_report(events_list: Any, title: str = "Market Analysis Report") -> str:
    """
    生成 Markdown 报告
    - 兼容输入为字符串/嵌套列表，尽可能解析成事件字典列表
    """
    def normalize(ev_input: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if ev_input is None:
            return out
        items = ev_input if isinstance(ev_input, list) else [ev_input]
        for it in items:
            if isinstance(it, dict):
                out.append(it)
            elif isinstance(it, list):
                for sub in it:
                    if isinstance(sub, dict):
                        out.append(sub)
                    elif isinstance(sub, str):
                        try:
                            parsed = json.loads(sub)
                            if isinstance(parsed, dict):
                                out.append(parsed)
                            elif isinstance(parsed, list):
                                out.extend([p for p in parsed if isinstance(p, dict)])
                        except Exception:
                            continue
            elif isinstance(it, str):
                try:
                    parsed = json.loads(it)
                    if isinstance(parsed, dict):
                        out.append(parsed)
                    elif isinstance(parsed, list):
                        out.extend([p for p in parsed if isinstance(p, dict)])
                except Exception:
                    continue
        return out

    events = normalize(events_list)
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

