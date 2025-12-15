"""
LLM API调用工具函数

统一处理LLM API调用，减少重复代码。
"""

import asyncio
import threading
import time
import logging
from typing import List, Any, Optional, Callable, Awaitable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar as TTypeVar
    T = TTypeVar('T')
else:
    T = TypeVar('T')

from ..agents.api_client import LLMAPIPool


class AsyncExecutor:
    """统一异步执行器"""

    def __init__(self, max_concurrent: float = float('inf'), logger: Optional[logging.Logger] = None) -> None:
        self.max_concurrent = max_concurrent
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    async def run_concurrent_tasks(
        self,
        tasks: List[Callable[[], Awaitable[Any]]],
        concurrency: int = 6,
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> List[Any]:
        """
        并发执行异步任务

        Args:
            tasks: 异步任务函数列表，每个函数应是 () -> Awaitable[Any]
            concurrency: 最大并发数
            semaphore: 可选的信号量，如果不提供则自动创建

        Returns:
            任务结果列表
        """
        if not semaphore:
            semaphore = asyncio.Semaphore(concurrency)

        async def run_task(task_func):
            async with semaphore:
                try:
                    return await task_func()
                except Exception as e:
                    self.logger.error(f"Task execution failed: {e}")
                    raise

        try:
            results = await asyncio.gather(*[run_task(task) for task in tasks])
            self.logger.debug(f"Successfully executed {len(tasks)} concurrent tasks")
            return results
        except Exception as e:
            self.logger.error(f"Concurrent task execution failed: {e}")
            raise

    async def run_with_timeout(
        self,
        coro: Awaitable[T],
        timeout: float = 60.0
    ) -> T:
        """
        带超时的异步执行

        Args:
            coro: 异步协程
            timeout: 超时时间（秒）

        Returns:
            协程执行结果

        Raises:
            asyncio.TimeoutError: 超时异常
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as e:
            self.logger.error(f"Operation timed out after {timeout} seconds")
            raise e

    def run_threaded_tasks(
        self,
        tasks: List[Any],
        func: Callable,
        max_workers: int = 4
    ) -> List[Any]:
        """
        多线程执行同步任务

        Args:
            tasks: 任务参数列表
            func: 处理函数
            max_workers: 最大工作线程数

        Returns:
            任务结果列表
        """
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(func, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Threaded task failed: {e}")
                    raise

        self.logger.debug(f"Successfully executed {len(tasks)} threaded tasks")
        return results


class RateLimiter:
    """令牌桶限速器"""

    def __init__(self, rate_per_sec: Optional[float] = None, rate_per_second: Optional[float] = None, logger: Optional[logging.Logger] = None) -> None:
        # 支持两种参数名以保持兼容性
        if rate_per_sec is not None:
            self.rate_per_sec = rate_per_sec
        elif rate_per_second is not None:
            self.rate_per_sec = rate_per_second
        else:
            raise ValueError("Either 'rate_per_sec' or 'rate_per_second' must be provided")

        # 为测试兼容性添加rate_per_second属性
        self.rate_per_second = self.rate_per_sec

        self.tokens = self.rate_per_sec  # 初始令牌数
        self._tokens = self.tokens  # 为测试兼容性添加别名
        self.max_tokens = self.rate_per_sec * 2  # 最大令牌数
        self.last_update = time.time()
        self._last_update = self.last_update  # 为测试兼容性添加别名
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """
        同步获取令牌（阻塞）
        """
        while True:
            with self._lock:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(self.max_tokens, self.tokens + time_passed * self.rate_per_sec)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            # 等待足够的时间来获取一个令牌
            time.sleep(1.0 / self.rate_per_sec)

    async def acquire_async(self) -> None:
        """
        异步获取令牌
        """
        while True:
            with self._lock:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(self.max_tokens, self.tokens + time_passed * self.rate_per_sec)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            # 异步等待
            await asyncio.sleep(1.0 / self.rate_per_sec)

    def try_acquire(self) -> bool:
        """
        尝试获取令牌（非阻塞）

        Returns:
            是否成功获取令牌
        """
        with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + time_passed * self.rate_per_sec)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def get_rate_per_sec(self) -> float:
        """获取每秒速率"""
        return self.rate_per_sec

    def set_rate_per_sec(self, rate_per_sec: float):
        """设置每秒速率"""
        with self._lock:
            self.rate_per_sec = rate_per_sec
            self.max_tokens = rate_per_sec * 2


def call_llm_with_retry(
    llm_pool: LLMAPIPool,
    prompt: str,
    max_tokens: int = 4000,
    timeout: int = 60,
    retries: int = 2,
    limiter: Optional[RateLimiter] = None
) -> Optional[str]:
    """
    统一的LLM调用，带重试和限流

    Args:
        llm_pool: LLM API池实例
        prompt: 提示文本
        max_tokens: 最大token数
        timeout: 超时时间（秒）
        retries: 重试次数
        limiter: 速率限制器

    Returns:
        LLM响应文本，失败时返回None
    """
    if llm_pool is None:
        return None

    # 应用速率限制
    if limiter:
        limiter.acquire()

    try:
        return llm_pool.call(
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries
        )
    except Exception:
        return None


def create_extraction_prompt(title: str, content: str, entity_definitions: str) -> str:
    """
    创建实体提取提示

    Args:
        title: 新闻标题
        content: 新闻内容
        entity_definitions: 实体定义文本

    Returns:
        完整的提示文本
    """
    return f"""你是一名专业的金融与法律信息结构化专家。请从以下新闻中提取所有**真实存在的、具有法律人格或行政职能的实体**。

【实体定义】
{entity_definitions}

【任务要求】
1. 判断新闻是否包含一个或多个独立事件，若包含多个事件，则需要将每个事件分开提取。
2. 对每个事件，输出：
   - 一个简洁、客观、无情绪的中文摘要（作为事件唯一标识）
   - 所有符合上述定义的实体（全称优先，避免缩写，若在该事件中存在的修饰词使得这个实体具有特殊意义或特化职能，则使用包含修饰词的实体作为其名称）
   - 所有符合上述定义的实体的原始语言表述
   - 该事件的本质描述（一句话说明"谁对谁做了什么"，尽可能地包含事件的所有必要信息及用于特化修饰的词语，并力求没有任何情绪色彩）

【输出格式】
严格返回 JSON，不要任何额外文本：
{{
  "events": [
    {{
      "abstract": "事件摘要",
      "entities": ["实体1", "实体2"],
      "entities_original": ["原始表述1", "原始表述2"],
      "event_summary": "事件描述"
    }}
  ]
}}

【新闻】
标题：{title}
正文：{content}"""


def create_deduplication_prompt(entities_batch: list, evidence_map: dict) -> str:
    """
    创建实体去重提示

    Args:
        entities_batch: 实体批次列表
        evidence_map: 证据映射

    Returns:
        完整的提示文本
    """
    evidence_lines = []
    for ent, evs in evidence_map.items():
        if evs:
            for ev in evs:
                evidence_lines.append(f"{ent} <= {ev}")

    return f"""你是一名知识图谱专家。任务：仅在有充分证据时认定实体为同一主体。

【实体列表】
{entities_batch}

【证据】
格式: 实体 <= 摘要 | 参与实体 | 描述
{chr(10).join(evidence_lines) if evidence_lines else "（无可用事件，谨慎合并）"}

【要求】
- 主实体优先更学术、更官方、更中文表述（更XX按优先级顺序）
- 主实体只有在中文语境中指代不唯一时采用原表述，否则采用中文表述。
- 主实体必须具有元完备性，即不能缺少任何必要信息，任何语境下的主实体都具备唯一指代性。
- 只输出确定为同一主体的组合；不确定就返回空。
- 优先严格匹配：同名、明显译名、缩写展开。
- 限制一：行使职能的组织、机构与其下辖的更具体职能的组织、机构不可合并。
- 限制二：不同国家/地区的同名机构，不可合并。
- 限制三：**自然人**实体不可跨越/**注册公司**/**政府机构或部门**/**主权国家或明确行政区**/**国际组织**/**重要产品、品牌及其型号**进行合并。
- 限制四：**注册公司**实体不可跨越/**自然人**/**政府机构或部门**/**主权国家或明确行政区**/**国际组织**/**重要产品、品牌及其型号**进行合并。
- 限制五：**政府机构或部门**实体不可跨越/**注册公司**/**自然人**/**主权国家或明确行政区**/**国际组织**/**重要产品、品牌及其型号**进行合并。
- 限制六：**主权国家或明确行政区**实体不可跨越/**注册公司**/**政府机构或部门**/**自然人**/**国际组织**/**重要产品、品牌及其型号**进行合并。
- 限制七：**国际组织**实体不可跨越**注册公司**/**政府机构或部门**/**主权国家或明确行政区**/**自然人**/**重要产品、品牌及其型号**进行合并。
- 限制八：**重要产品、品牌及其型号**实体不可跨越/**注册公司**/**政府机构或部门**/**主权国家或明确行政区**/**自然人**/**国际组织**进行合并。


【输出格式】
严格返回 JSON：
{{
  "duplicate_entities": [
    ["主实体", "别名或重复"],
    ["主实体2", "别名2", "别名3"]
  ]
}}
如果没有重复，返回 {{"duplicate_entities": []}}。只输出JSON。"""


def create_event_deduplication_prompt(events_batch: dict) -> str:
    """
    创建事件去重提示

    Args:
        events_batch: 事件批次字典

    Returns:
        完整的提示文本
    """
    prompt_lines = ["格式: 摘要 | 参与实体 | 事件描述"]
    for abstract, event in events_batch.items():
        entities = event.get('entities', [])
        summary = event.get('event_summary', '')
        prompt_lines.append(f"{abstract} | {', '.join(entities)} | {summary}")

    return f"""你是一名知识图谱专家。任务：仅在描述"同一具体事实"时才视为重复事件。

【事件列表】
{chr(10).join(prompt_lines)}

【任务】
找出语义上高度重叠、描述同一事实的事件。
限制一：事件具有连续发生或一前一后的关系，不得视为重复事件。


【输出格式】
严格返回 JSON：
{{
  "duplicate_events": [
    ["事件摘要1", "事件摘要2"],
    ["事件摘要3", "事件摘要4", "事件摘要5"]
  ]
}}
如果没有重复，返回 {{"duplicate_events": []}}。只输出JSON。"""
