"""
DataNormalizer - 数据规范化工具
统一处理各种数据输入格式的转换
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Callable
from .logging import LoggerManager
from .serialization import Serializer


class DataNormalizer:
    """统一数据规范化器"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger: logging.Logger = logger or LoggerManager.get_logger(__name__)

    @staticmethod
    def normalize_event_input(ev_input: Any) -> List[Dict[str, Any]]:
        """
        规范化事件输入，支持多种输入格式

        支持的输入格式：
        - Dict: 单个事件
        - List[Dict]: 多个事件
        - List[str]: JSON字符串或文件路径
        - str: JSON字符串或文件路径

        Args:
            ev_input: 输入数据

        Returns:
            规范化后的事件列表
        """
        out: List[Dict[str, Any]] = []

        if ev_input is None:
            return out

        def load_file(path_str: str) -> List[Dict[str, Any]]:
            """从文件加载数据"""
            file_out = []
            p = Path(path_str)
            if not p.exists():
                return file_out

            try:
                if p.suffix.lower() in [".jsonl", ".json"]:
                    if p.suffix.lower() == ".jsonl":
                        with p.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict):
                                        file_out.append(obj)
                                except Exception:
                                    continue
                    else:
                        parsed = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(parsed, dict):
                            file_out.append(parsed)
                        elif isinstance(parsed, list):
                            file_out.extend([p for p in parsed if isinstance(p, dict)])
            except Exception as e:
                LoggerManager.get_logger(__name__).warning(f"Failed to load file {path_str}: {e}")

            return file_out

        # 处理不同类型的输入
        items = ev_input if isinstance(ev_input, list) else [ev_input]

        for it in items:
            if isinstance(it, dict):
                out.append(it)
            elif isinstance(it, list):
                for sub in it:
                    if isinstance(sub, dict):
                        out.append(sub)
                    elif isinstance(sub, str):
                        # 尝试作为JSON字符串或文件路径
                        try:
                            parsed = json.loads(sub)
                            if isinstance(parsed, dict):
                                out.append(parsed)
                            elif isinstance(parsed, list):
                                out.extend([p for p in parsed if isinstance(p, dict)])
                        except Exception:
                            # 尝试作为文件路径
                            file_data = load_file(sub)
                            out.extend(file_data)
            elif isinstance(it, str):
                try:
                    parsed = json.loads(it)
                    if isinstance(parsed, dict):
                        out.append(parsed)
                    elif isinstance(parsed, list):
                        out.extend([p for p in parsed if isinstance(p, dict)])
                except Exception:
                    # 尝试作为文件路径
                    file_data = load_file(it)
                    out.extend(file_data)

        return out

    @staticmethod
    def validate_event_format(event: Dict) -> bool:
        """
        验证事件数据格式

        Args:
            event: 事件字典

        Returns:
            是否为有效格式
        """
        required_fields = ['abstract', 'entities', 'event_summary']
        return all(field in event for field in required_fields)

    @staticmethod
    def clean_event_data(events: List[Dict]) -> List[Dict]:
        """
        清理和标准化事件数据

        Args:
            events: 事件列表

        Returns:
            清理后的事件列表
        """
        cleaned = []

        for event in events:
            if not isinstance(event, dict):
                continue

            # 确保必需字段存在
            if not DataNormalizer.validate_event_format(event):
                continue

            cleaned_event = {
                'abstract': str(event.get('abstract', '')).strip(),
                'entities': event.get('entities', []) if isinstance(event.get('entities'), list) else [],
                'entities_original': event.get('entities_original', []) if isinstance(event.get('entities_original'), list) else [],
                'event_summary': str(event.get('event_summary', '')).strip(),
                'source': str(event.get('source', 'unknown')).strip(),
                'published_at': event.get('published_at'),
                'news_id': event.get('news_id')
            }

            # 确保entities和entities_original长度一致
            entities = cleaned_event['entities']
            entities_original = cleaned_event['entities_original']

            if len(entities_original) < len(entities):
                # 补充缺失的原始实体
                entities_original.extend([''] * (len(entities) - len(entities_original)))
            elif len(entities_original) > len(entities):
                # 截断多余的原始实体
                entities_original = entities_original[:len(entities)]

            cleaned_event['entities_original'] = entities_original
            cleaned.append(cleaned_event)

        return cleaned

    @staticmethod
    def merge_duplicate_events(events: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """
        合并相似的重复事件（简单实现）

        Args:
            events: 事件列表
            similarity_threshold: 相似度阈值

        Returns:
            去重后的事件列表
        """
        if not events:
            return events

        from difflib import SequenceMatcher

        merged = []

        for event in events:
            is_duplicate = False

            for existing in merged:
                # 简单的摘要相似度比较
                similarity = SequenceMatcher(None,
                    event.get('abstract', ''),
                    existing.get('abstract', '')
                ).ratio()

                if similarity >= similarity_threshold:
                    # 合并实体和来源
                    existing_entities = set(existing.get('entities', []))
                    existing_entities.update(event.get('entities', []))
                    existing['entities'] = list(existing_entities)

                    existing_sources = set(existing.get('sources', []))
                    existing_sources.add(event.get('source', 'unknown'))
                    existing['sources'] = list(existing_sources)

                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(event.copy())

        return merged


# ======================
# 数据管道架构 (TF-03)
# ======================

class DataPipeline:
    """
    标准化数据处理管道
    支持：输入规范化 → 验证 → 转换 → 输出序列化
    """

    def __init__(self, name: str = "default_pipeline"):
        self.name = name
        self.logger = LoggerManager.get_logger(f"DataPipeline.{name}")
        self.stages: List[Dict[str, Any]] = []

    def add_stage(self, name: str, processor: Callable, config: Optional[Dict[str, Any]] = None):
        """添加处理阶段"""
        self.stages.append({
            "name": name,
            "processor": processor,
            "config": config or {}
        })

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        执行完整数据管道

        Args:
            input_data: 原始输入数据

        Returns:
            处理结果字典
        """
        self.logger.info(f"开始执行数据管道: {self.name}")
        current_data = input_data
        results = {}

        for stage in self.stages:
            stage_name = stage["name"]
            processor = stage["processor"]
            config = stage["config"]

            try:
                self.logger.debug(f"执行阶段: {stage_name}")
                if hasattr(processor, '__call__'):
                    # 支持同步和异步处理器
                    import asyncio
                    if asyncio.iscoroutinefunction(processor):
                        current_data = await processor(current_data, **config)
                    else:
                        current_data = processor(current_data, **config)

                results[stage_name] = current_data
                self.logger.debug(f"阶段 {stage_name} 完成")

            except Exception as e:
                self.logger.error(f"阶段 {stage_name} 执行失败: {e}")
                results[stage_name] = {"error": str(e), "stage": stage_name}
                break

        self.logger.info(f"数据管道 {self.name} 执行完成")
        return results


class StandardEventPipeline(DataPipeline):
    """
    标准事件数据处理管道
    预定义的事件数据处理流程
    """

    def __init__(self):
        super().__init__("event_pipeline")
        self._setup_stages()

    def _setup_stages(self):
        """设置标准处理阶段"""
        # 1. 数据规范化
        self.add_stage(
            "normalization",
            self._normalize_data,
            {"normalizer": DataNormalizer()}
        )

        # 2. 数据验证
        self.add_stage(
            "validation",
            self._validate_data
        )

        # 3. 数据转换
        self.add_stage(
            "transformation",
            self._transform_data
        )

        # 4. 数据序列化
        self.add_stage(
            "serialization",
            self._serialize_data,
            {"serializer": Serializer()}
        )

    def _normalize_data(self, data: Any, normalizer: DataNormalizer) -> List[Dict[str, Any]]:
        """数据规范化阶段"""
        return normalizer.normalize_event_input(data)

    def _validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """数据验证阶段"""
        validated = []
        for item in data:
            if DataNormalizer.validate_event_format(item):
                validated.append(item)
            else:
                self.logger.warning(f"数据格式验证失败: {item.get('id', 'unknown')}")
        return validated

    def _transform_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """数据转换阶段"""
        # 标准化时间格式、清理数据等
        transformed = []
        for item in data:
            transformed_item = item.copy()

            # 标准化时间字段
            if 'published_at' in transformed_item:
                # 确保时间格式一致
                pass

            # 清理实体字段
            if 'entities' in transformed_item:
                entities = transformed_item['entities']
                if isinstance(entities, str):
                    transformed_item['entities'] = [entities]
                elif not isinstance(entities, list):
                    transformed_item['entities'] = []

            transformed.append(transformed_item)

        return transformed

    def _serialize_data(self, data: List[Dict[str, Any]], serializer: Serializer) -> str:
        """数据序列化阶段"""
        return serializer.safe_json_dumps(data)


class BatchDataProcessor:
    """
    批量数据处理器
    结合AsyncExecutor进行并发数据处理
    """

    def __init__(self, pipeline: DataPipeline, max_workers: int = 5):
        self.pipeline = pipeline
        self.max_workers = max_workers
        self.logger = LoggerManager.get_logger("BatchDataProcessor")

    async def process_batch(self, batch_data: List[Any]) -> List[Dict[str, Any]]:
        """
        并发处理批量数据

        Args:
            batch_data: 批量输入数据

        Returns:
            处理结果列表
        """
        from ..utils.llm_utils import AsyncExecutor

        self.logger.info(f"开始批量处理 {len(batch_data)} 项数据")

        async def process_single(item):
            try:
                result = await self.pipeline.execute(item)
                return {"success": True, "data": result, "input": item}
            except Exception as e:
                self.logger.error(f"处理失败: {e}")
                return {"success": False, "error": str(e), "input": item}

        async_executor = AsyncExecutor()
        results = await async_executor.run_concurrent_tasks(
            tasks=[lambda i=item: process_single(i) for item in batch_data],
            concurrency=self.max_workers
        )

        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        self.logger.info(f"批量处理完成: {len(successful)} 成功, {len(failed)} 失败")

        return results
