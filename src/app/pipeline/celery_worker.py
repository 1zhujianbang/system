"""
应用层 - Celery 分布式任务处理

实现基于 Celery 的分布式任务处理系统，用于优化 Pipeline 执行。
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from celery import Celery
from celery.schedules import crontab

from .engine import PipelineEngine
from .context import PipelineContext
from ...infra import get_logger


# 初始化 Celery 应用
celery_app = Celery('marketlens_pipeline')

# 配置 Celery
celery_app.conf.update(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'pipeline.tasks.run_pipeline_task': {'queue': 'pipelines'},
        'pipeline.tasks.run_step_task': {'queue': 'steps'},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


# 日志记录器
logger = get_logger(__name__)


@celery_app.task(bind=True)
def run_pipeline_task(
    self,
    run_id: str,
    project_id: str,
    pipeline_def: Dict[str, Any],
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    执行完整的 Pipeline 任务
    
    Args:
        run_id: 运行ID
        project_id: 项目ID
        pipeline_def: Pipeline定义
        context_data: 上下文数据
        
    Returns:
        Dict: 执行结果
    """
    try:
        logger.info(f"Starting pipeline run {run_id} for project {project_id}")
        
        # 创建 Pipeline 上下文
        ctx = PipelineContext()
        for key, value in context_data.items():
            ctx.set(key, value)
        
        # 创建 Pipeline 引擎
        engine = PipelineEngine(ctx)
        
        # 运行 Pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ctx = loop.run_until_complete(
            engine.run_pipeline(
                run_id=run_id,
                project_id=project_id,
                pipeline_def=pipeline_def
            )
        )
        
        # 返回结果
        result = {
            "status": "success",
            "run_id": run_id,
            "project_id": project_id,
            "context": ctx.to_dict(),
            "completed_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Pipeline run {run_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline run {run_id} failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "run_id": run_id,
            "project_id": project_id,
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True)
def run_step_task(
    self,
    run_id: str,
    step_idx: int,
    step: Dict[str, Any],
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    执行单个 Pipeline 步骤任务
    
    Args:
        run_id: 运行ID
        step_idx: 步骤索引
        step: 步骤定义
        context_data: 上下文数据
        
    Returns:
        Dict: 执行结果
    """
    try:
        logger.info(f"Starting step {step_idx} for run {run_id}")
        
        # 创建 Pipeline 上下文
        ctx = PipelineContext()
        for key, value in context_data.items():
            ctx.set(key, value)
        
        # 创建 Pipeline 引擎
        engine = PipelineEngine(ctx)
        
        # 运行步骤
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            engine.run_step(None, step_idx, step)
        )
        
        # 返回结果
        return {
            "status": "success",
            "run_id": run_id,
            "step_idx": step_idx,
            "result": result,
            "context": ctx.to_dict(),
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Step {step_idx} for run {run_id} failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "run_id": run_id,
            "step_idx": step_idx,
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True)
def health_check_task(self) -> Dict[str, Any]:
    """
    健康检查任务
    
    Returns:
        Dict: 健康状态
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "worker": self.request.hostname
    }


# 定期任务配置
celery_app.conf.beat_schedule = {
    'health-check': {
        'task': 'pipeline.tasks.health_check_task',
        'schedule': crontab(minute='*/5'),  # 每5分钟执行一次
    },
}


# 自动发现任务
celery_app.autodiscover_tasks(['src.app.pipeline'])


if __name__ == '__main__':
    celery_app.start()