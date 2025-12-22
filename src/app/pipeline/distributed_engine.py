"""
应用层 - 分布式 Pipeline 执行引擎

实现基于 Celery 的分布式 Pipeline 执行引擎，用于替代原有的单机执行引擎。
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

from ...infra.registry import FunctionRegistry
from .context import PipelineContext
from .models import PipelineRunState, StepState
from .celery_worker import run_pipeline_task, run_step_task


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


Hook = Callable[[PipelineRunState, StepState, PipelineContext], None]


class DistributedPipelineEngine:
    """
    分布式 Pipeline 执行引擎（与 UI 解耦）。

    - 输入解析：支持 "$var" 从 context 注入
    - 工具调用：支持分布式任务处理
    - hooks：每步开始/结束回调（用于 run_store 落盘与观测）
    """

    def __init__(
        self,
        context: PipelineContext,
        *,
        on_step_start: Optional[Hook] = None,
        on_step_end: Optional[Hook] = None,
        use_distributed: bool = True,
    ) -> None:
        self.ctx = context
        self.on_step_start = on_step_start
        self.on_step_end = on_step_end
        self.use_distributed = use_distributed
        self._tools_loaded = False

    def _ensure_tools_loaded(self) -> None:
        """
        确保工具已注册到 FunctionRegistry。
        """
        if self._tools_loaded:
            return
        import importlib
        import traceback
        import sys
        
        errors = []
        
        # 加载业务模块（app/business），确保所有 @register_tool 被执行
        try:
            importlib.import_module("src.app.business")
        except Exception as e:
            errors.append(f"src.app.business: {e}\n{traceback.format_exc()}")
        
        # 加载新入口（interfaces/tools）
        try:
            importlib.import_module("src.interfaces.tools")
        except Exception as e:
            errors.append(f"src.interfaces.tools: {e}\n{traceback.format_exc()}")
        
        self._tools_loaded = True
        
        # 记录加载结果
        if errors:
            error_msg = "Tool loading errors:\n" + "\n".join(errors)
            print(error_msg, file=sys.stderr)

    def _resolve_inputs(self, step: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_tools_loaded()
        mapping = step.get("inputs", {}) or {}
        out: Dict[str, Any] = {}
        for k, v in mapping.items():
            if isinstance(v, str) and v.startswith("$"):
                out[k] = self.ctx.get(v[1:])
            else:
                out[k] = v

        # pydantic 校验（沿用 registry 的自动模型）
        tool_name = str(step.get("tool") or "")
        model = FunctionRegistry.get_input_model(tool_name)
        if model:
            try:
                validated: BaseModel = model(**out)  # type: ignore[misc]
                out = validated.model_dump()
            except Exception as e:
                raise ValueError(f"参数校验失败: {e}") from e
        return out

    async def run_step(self, run_state: Optional[PipelineRunState], step_idx: int, step: Dict[str, Any]) -> Any:
        """
        运行单个步骤（支持分布式执行）
        """
        step_id = str(step.get("id") or step.get("tool") or f"step_{step_idx+1}")
        tool = str(step.get("tool") or "")
        output_key = str(step.get("output") or "").strip()
        retry = int(step.get("retry", 0) or 0)
        continue_on_error = bool(step.get("continue_on_error", False))

        started_at = _utc_now_iso()
        t0 = time.time()
        st = StepState(
            step_idx=step_idx,
            step_id=step_id,
            tool=tool,
            output_key=output_key,
            status="running",
            started_at=started_at,
            ended_at="",
            duration_ms=0.0,
            inputs_resolved=None,
        )

        inputs = self._resolve_inputs(step)
        st.inputs_resolved = inputs
        if self.on_step_start and run_state is not None:
            self.on_step_start(run_state, st, self.ctx)

        # 如果启用分布式执行，将任务发送到 Celery
        if self.use_distributed:
            try:
                # 发送任务到 Celery
                task = run_step_task.delay(
                    run_id=run_state.run_id if run_state else f"step_run_{int(time.time())}",
                    step_idx=step_idx,
                    step=step,
                    context_data=self.ctx.to_dict()
                )
                
                # 等待任务完成
                result = task.get(timeout=300)  # 5分钟超时
                
                if result["status"] == "success":
                    step_result = result.get("result")
                    if output_key:
                        self.ctx.set(output_key, step_result)
                    st.status = "success"
                    st.ended_at = _utc_now_iso()
                    st.duration_ms = (time.time() - t0) * 1000.0
                    if self.on_step_end and run_state is not None:
                        self.on_step_end(run_state, st, self.ctx)
                    return step_result
                else:
                    raise Exception(result.get("error", "Unknown error"))
                    
            except Exception as e:
                st.status = "failed"
                st.error = str(e)
                st.ended_at = _utc_now_iso()
                st.duration_ms = (time.time() - t0) * 1000.0
                if self.on_step_end and run_state is not None:
                    self.on_step_end(run_state, st, self.ctx)
                
                if continue_on_error:
                    self.ctx.log(f"步骤失败但 continue_on_error=True，将继续: {e}", level="WARN", source=step_id)
                    return None
                raise e
        else:
            # 使用本地执行（原有逻辑）
            func = FunctionRegistry.get_tool(tool)
            if not func:
                err = f"找不到工具: {tool}"
                st.status = "failed"
                st.error = err
                st.ended_at = _utc_now_iso()
                st.duration_ms = (time.time() - t0) * 1000.0
                if self.on_step_end and run_state is not None:
                    self.on_step_end(run_state, st, self.ctx)
                raise ValueError(err)

            attempt = 0
            last_exc: Optional[Exception] = None
            while attempt <= retry:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**inputs)
                    else:
                        # 同步函数放线程池，避免阻塞 event loop
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(None, lambda: func(**inputs))
                    if output_key:
                        self.ctx.set(output_key, result)
                    st.status = "success"
                    st.ended_at = _utc_now_iso()
                    st.duration_ms = (time.time() - t0) * 1000.0
                    if self.on_step_end and run_state is not None:
                        self.on_step_end(run_state, st, self.ctx)
                    return result
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    attempt += 1
                    if attempt <= retry:
                        self.ctx.log(f"任务重试 {attempt}/{retry}: {e}", level="WARN", source=step_id)
                        await asyncio.sleep(min(5.0, 1.0 * attempt))

            st.status = "failed"
            st.error = str(last_exc)
            st.ended_at = _utc_now_iso()
            st.duration_ms = (time.time() - t0) * 1000.0
            if self.on_step_end and run_state is not None:
                self.on_step_end(run_state, st, self.ctx)

            if continue_on_error:
                self.ctx.log(f"步骤失败但 continue_on_error=True，将继续: {last_exc}", level="WARN", source=step_id)
                return None
            raise last_exc  # type: ignore[misc]

    async def run_pipeline(
        self,
        *,
        run_id: str,
        project_id: str,
        pipeline_def: Dict[str, Any],
        start_at: int = 0,
    ) -> PipelineContext:
        """
        运行完整 Pipeline（支持分布式执行）
        """
        steps = pipeline_def.get("steps", []) or []
        
        # 如果启用分布式执行，将整个 Pipeline 发送到 Celery
        if self.use_distributed:
            try:
                # 发送任务到 Celery
                task = run_pipeline_task.delay(
                    run_id=run_id,
                    project_id=project_id,
                    pipeline_def=pipeline_def,
                    context_data=self.ctx.to_dict()
                )
                
                # 等待任务完成
                result = task.get(timeout=3600)  # 1小时超时
                
                if result["status"] == "success":
                    # 更新上下文
                    context_data = result.get("context", {})
                    for key, value in context_data.items():
                        self.ctx.set(key, value)
                    return self.ctx
                else:
                    raise Exception(result.get("error", "Unknown error"))
                    
            except Exception as e:
                raise Exception(f"Distributed pipeline execution failed: {str(e)}") from e
        else:
            # 使用本地执行（原有逻辑）
            run_state = PipelineRunState(
                run_id=str(run_id),
                project_id=str(project_id),
                pipeline_name=str(pipeline_def.get("name") or "Pipeline"),
                total_steps=len(steps),
            )

            for i, step in enumerate(steps):
                if i < int(start_at):
                    continue
                if not isinstance(step, dict):
                    raise ValueError(f"非法 step 配置（必须为 dict）：index={i}")
                await self.run_step(run_state, i, step)
            return self.ctx