"""
Tools（CLI/工作流入口层）薄封装。

原则：
- 这里只做 register_tool 与参数适配
- 真实业务编排在 `src/app/*`
"""

# 导入子模块以触发 @register_tool（按需增加）
from . import snapshots  # noqa: F401






