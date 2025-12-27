"""
Pipeline 执行框架（应用层）。

目标：
- 与 Streamlit/UI 解耦，可在 CLI/worker 中复用
- 统一 step 输入解析、工具调用、状态记录
- 提供 hooks 便于 run_store 落盘与实时日志
"""

from .engine import PipelineEngine  # noqa: F401
from .models import PipelineRunState, StepState  # noqa: F401






