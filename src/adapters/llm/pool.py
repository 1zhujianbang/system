"""
适配器层 - LLM 客户端池

实现 LLM 客户端池，支持多服务轮询、熔断、限速等功能。
"""
from __future__ import annotations

import random
import json
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from ...ports.llm_client import (
    LLMClient, LLMClientPool, LLMProviderType,
    LLMCallConfig, LLMResponse, RateLimiter, CircuitBreaker
)
from ...infra import (
    get_logger, TokenBucketRateLimiter, SimpleCircuitBreaker,
    LLMError, CircuitBreakerOpenError
)


class ClientEntry:
    """客户端条目"""
    def __init__(
        self,
        name: str,
        client: Any,
        model: str,
        provider_type: LLMProviderType,
        service_key: str
    ):
        self.name = name
        self.client = client
        self.model = model
        self.provider_type = provider_type
        self.service_key = service_key


class DefaultLLMPool(LLMClientPool):
    """默认 LLM 客户端池实现"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self.clients: List[ClientEntry] = []
        self._disabled_until: Dict[str, float] = {}
        self._disabled_file: Optional[Path] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breakers: Dict[str, SimpleCircuitBreaker] = {}
        self._lock = threading.Lock()

        # 已移除JSON文件处理逻辑
        pass
        
        # 初始化时自动加载配置的LLM客户端
        self._load_configured_clients()

    def _load_disabled_state(self):
        """加载熔断状态（已移除JSON文件处理）"""
        # 已移除JSON文件处理逻辑，使用内存存储
        pass

    def _save_disabled_state(self):
        """保存熔断状态（已移除JSON文件处理）"""
        # 已移除JSON文件处理逻辑，使用内存存储
        pass

    def _load_configured_clients(self):
        """从配置加载LLM客户端"""
        try:
            import os
            import json
            from dotenv import load_dotenv
            from pathlib import Path
            
            # 加载 .env.local 文件
            project_root = Path(__file__).parent.parent.parent.parent.resolve()
            dotenv_path = project_root / "config" / ".env.local"
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
                self.logger.info(f"[LLM] 已加载 .env.local 文件: {dotenv_path}")
            
            # 从环境变量获取 LLM API 配置
            llm_apis_json = os.getenv("AGENT1_LLM_APIS", "[]")
            try:
                llm_configs = json.loads(llm_apis_json)
                self.logger.info(f"[LLM] 从环境变量加载到 {len(llm_configs)} 个LLM配置")
            except json.JSONDecodeError as e:
                self.logger.error(f"[LLM] 解析 AGENT1_LLM_APIS 失败: {e}")
                return
            
            # 注册配置的客户端
            if isinstance(llm_configs, list):
                for i, config in enumerate(llm_configs):
                    if not isinstance(config, dict):
                        continue
                        
                    # 获取配置项
                    name = config.get("name", f"llm-client-{i}")
                    api_key = config.get("api_key", "")
                    provider_raw = (
                        config.get("provider")
                        or config.get("provider_type")
                        or config.get("providerType")
                        or config.get("type")
                        or ""
                    )
                    provider_str = str(provider_raw or "").strip().lower()
                    if provider_str in ("ollama", "local"):
                        provider_type = LLMProviderType.LOCAL
                    else:
                        try:
                            provider_type = LLMProviderType(provider_str) if provider_str else LLMProviderType.OPENAI
                        except Exception:
                            provider_type = LLMProviderType.OPENAI

                    base_url = config.get("base_url", "")
                    if not isinstance(base_url, str):
                        base_url = ""
                    base_url = base_url.strip()
                    if not base_url:
                        base_url = "http://localhost:11434/v1" if provider_type == LLMProviderType.LOCAL else "https://api.openai.com/v1"

                    model = config.get("model", "gpt-3.5-turbo")
                    enabled = config.get("enabled", True)
                    
                    # 检查是否启用
                    if not enabled:
                        self.logger.info(f"[LLM] 跳过未启用的客户端: {name}")
                        continue
                        
                    # 检查API密钥
                    if not api_key:
                        if provider_type == LLMProviderType.LOCAL:
                            api_key = "ollama"
                        else:
                            self.logger.warning(f"[LLM] 客户端 {name} 缺少API密钥")
                            continue
                    
                    # 注册客户端
                    if self.register_openai_client(name, api_key, base_url, model, provider_type=provider_type):
                        self.logger.info(f"[LLM] 成功注册客户端: {name} (provider: {provider_type.value}, model: {model}, base_url: {base_url})")
                    else:
                        self.logger.error(f"[LLM] 注册OpenAI客户端失败: {name}")
                        
        except Exception as e:
            self.logger.error(f"[LLM] 加载配置客户端失败: {e}", exc_info=True)

    def register(self, name: str, client: LLMClient, priority: int = 0) -> None:
        """注册 LLM 客户端"""
        # 为每个客户端创建熔断器
        self._circuit_breakers[name] = SimpleCircuitBreaker(
            failure_threshold=5,
            recovery_timeout_seconds=60.0
        )
        self.logger.info(f"Registered LLM client: {name}")

    def register_openai_client(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        provider_type: LLMProviderType = LLMProviderType.OPENAI
    ) -> bool:
        """注册 OpenAI 兼容客户端"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)

            entry = ClientEntry(
                name=name,
                client=client,
                model=model,
                provider_type=provider_type,
                service_key=f"llm_{name.lower()}"
            )
            self.clients.append(entry)
            self._circuit_breakers[name] = SimpleCircuitBreaker()

            self.logger.info(f"Registered OpenAI client: {name} (model: {model})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register client {name}: {e}")
            return False

    def get(self, name: str) -> Optional[LLMClient]:
        """获取指定客户端"""
        for entry in self.clients:
            if entry.name == name:
                return self._wrap_client(entry)
        return None

    def add_client(self, client: LLMClient) -> None:
        """添加客户端（实现抽象方法）"""
        # 使用 register 方法的简化版本
        name = getattr(client, 'name', f'client_{len(self.clients)}')
        self._circuit_breakers[name] = SimpleCircuitBreaker()
        self.logger.info(f"Added LLM client: {name}")

    def remove_client(self, provider: LLMProviderType) -> None:
        """移除指定提供商的客户端"""
        with self._lock:
            to_remove = [e for e in self.clients if e.provider_type == provider]
            for entry in to_remove:
                self.clients.remove(entry)
                self._circuit_breakers.pop(entry.name, None)
                self._disabled_until.pop(entry.name, None)
            self._save_disabled_state()

    def get_client(self, provider: Optional[LLMProviderType] = None) -> Optional[LLMClient]:
        """获取客户端（支持按提供商筛选）"""
        available = self._get_available_clients()
        if provider:
            available = [e for e in available if e.provider_type == provider]
        if not available:
            return None
        entry = random.choice(available)
        return self._wrap_client(entry)

    def list_available(self) -> List[LLMProviderType]:
        """列出可用的提供商类型"""
        available = self._get_available_clients()
        providers = set(e.provider_type for e in available)
        return list(providers)

    async def call_async(
        self,
        prompt: str,
        config: Optional[LLMCallConfig] = None,
        preferred_provider: Optional[LLMProviderType] = None,
    ) -> LLMResponse:
        """异步调用 LLM"""
        # 简化实现：在线程池中运行同步方法
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.call(prompt, config))

    def _wrap_client(self, entry: ClientEntry) -> LLMClient:
        """包装客户端为 LLMClient 接口"""
        # 这里返回一个简单的包装器
        return PooledLLMClient(entry, self)

    def list_clients(self) -> List[str]:
        """列出所有客户端"""
        return [entry.name for entry in self.clients]

    def get_available_client(self) -> Optional[LLMClient]:
        """获取可用客户端"""
        available = self._get_available_clients()
        if not available:
            return None
        entry = random.choice(available)
        return self._wrap_client(entry)

    def _get_available_clients(self) -> List[ClientEntry]:
        """获取可用客户端列表"""
        now_ts = time.time()
        available = []
        for entry in self.clients:
            # 检查熔断状态
            if self._disabled_until.get(entry.name, 0) > now_ts:
                continue
            # 检查熔断器状态
            breaker = self._circuit_breakers.get(entry.name)
            if breaker and not breaker.can_call():
                continue
            available.append(entry)
        return available if available else self.clients.copy()

    def set_global_rate_limiter(self, limiter: RateLimiter) -> None:
        """设置全局限速器"""
        self._rate_limiter = limiter

    def set_global_circuit_breaker(self, breaker: CircuitBreaker) -> None:
        """设置全局熔断器（不适用于池，使用单独熔断器）"""
        pass

    def call(
        self,
        prompt: str,
        config: Optional[LLMCallConfig] = None
    ) -> LLMResponse:
        """调用 LLM（自动选择客户端）"""
        config = config or LLMCallConfig()
        max_retries = config.retries or 2

        for attempt in range(max_retries + 1):
            available = self._get_available_clients()
            if not available:
                return LLMResponse(
                    content="",
                    success=False,
                    error="No available LLM clients"
                )

            entry = random.choice(available)
            try:
                self.logger.info(f"[LLM] Trying {entry.name} (attempt {attempt + 1})")

                # 应用限速
                if self._rate_limiter:
                    self._rate_limiter.acquire()

                # 调用 OpenAI 客户端
                response = entry.client.chat.completions.create(
                    model=config.model or entry.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.max_tokens or 1500,
                    timeout=config.timeout_seconds or 55,
                    stream=False
                )

                content = response.choices[0].message.content.strip()

                # 记录成功
                breaker = self._circuit_breakers.get(entry.name)
                if breaker:
                    breaker.record_success()

                return LLMResponse(
                    content=content,
                    provider=entry.provider_type,
                    model=entry.model
                )

            except Exception as e:
                self.logger.error(f"[LLM] {entry.name} failed: {e}", exc_info=True)

                # 记录失败并应用熔断
                self._handle_error(entry.name, e)

                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return LLMResponse(
            content="",
            success=False,
            error="All LLM clients failed"
        )

    def _handle_error(self, name: str, error: Exception):
        """处理错误，应用熔断策略"""
        breaker = self._circuit_breakers.get(name)
        if breaker:
            breaker.record_failure()

        err_s = str(error)
        if "Arrearage" in err_s or "overdue-payment" in err_s or "Access denied" in err_s:
            self._disabled_until[name] = time.time() + 24 * 3600
        elif "401" in err_s or "403" in err_s:
            self._disabled_until[name] = time.time() + 3600
        elif "Connection error" in err_s or "timed out" in err_s:
            self._disabled_until[name] = time.time() + 10 * 60

        self._save_disabled_state()

    def remove_service(self, name: str) -> bool:
        """移除服务"""
        with self._lock:
            self.clients = [e for e in self.clients if e.name != name]
            self._circuit_breakers.pop(name, None)
            self._disabled_until.pop(name, None)
            self._save_disabled_state()
            return True

    def list_services(self) -> List[Dict[str, Any]]:
        """列出所有服务信息"""
        services = []
        for entry in self.clients:
            breaker = self._circuit_breakers.get(entry.name)
            services.append({
                "name": entry.name,
                "model": entry.model,
                "provider": entry.provider_type.name,
                "circuit_state": breaker.state if breaker else "unknown",
                "disabled_until": self._disabled_until.get(entry.name)
            })
        return services


class PooledLLMClient(LLMClient):
    """池化 LLM 客户端包装器"""

    def __init__(self, entry: ClientEntry, pool: DefaultLLMPool):
        self._entry = entry
        self._pool = pool
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None

    @property
    def provider(self) -> LLMProviderType:
        """获取提供商类型"""
        return self._entry.provider_type

    @property
    def available_models(self) -> List[str]:
        """获取可用模型列表"""
        return [self._entry.model]

    def call(self, prompt: str, config: Optional[LLMCallConfig] = None) -> LLMResponse:
        """调用 LLM"""
        config = config or LLMCallConfig()

        # 检查熔断器
        if self._circuit_breaker and not self._circuit_breaker.can_call():
            return LLMResponse(
                content="",
                success=False,
                error="Circuit breaker is open"
            )

        # 应用限速
        if self._rate_limiter:
            self._rate_limiter.acquire()

        try:
            response = self._entry.client.chat.completions.create(
                model=config.model or self._entry.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens or 1500,
                timeout=config.timeout_seconds or 55,
                stream=False
            )

            content = response.choices[0].message.content.strip()

            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            return LLMResponse(
                content=content,
                provider=self._entry.provider_type,
                model=self._entry.model
            )
        except Exception as e:
            self._pool.logger.error(f"[LLM] Pooled client {self._entry.name} failed: {e}", exc_info=True)
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()

            return LLMResponse(
                content="",
                success=False,
                error=str(e)
            )

    def set_rate_limiter(self, limiter: RateLimiter) -> None:
        """设置限速器"""
        self._rate_limiter = limiter

    def set_circuit_breaker(self, breaker: CircuitBreaker) -> None:
        """设置熔断器"""
        self._circuit_breaker = breaker

    async def call_async(
        self,
        prompt: str,
        config: Optional[LLMCallConfig] = None,
    ) -> LLMResponse:
        """异步调用 LLM"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.call(prompt, config))

    def call_with_retry(
        self,
        prompt: str,
        config: Optional[LLMCallConfig] = None,
        max_retries: int = 3,
        on_retry: Optional[Any] = None,
    ) -> LLMResponse:
        """带重试的调用"""
        import time
        last_error = None
        for attempt in range(max_retries + 1):
            response = self.call(prompt, config)
            if response.success:
                return response
            last_error = response.error
            if on_retry and attempt < max_retries:
                on_retry(attempt, Exception(response.error or "Unknown error"))
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        return LLMResponse(content="", success=False, error=last_error)

    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 简单测试：检查客户端是否可用
            return self._entry.client is not None
        except Exception:
            return False


# =============================================================================
# 全局实例
# =============================================================================

_llm_pool: Optional[DefaultLLMPool] = None
_pool_lock = threading.Lock()


def get_llm_pool() -> DefaultLLMPool:
    """获取全局 LLM 池"""
    global _llm_pool
    if _llm_pool is None:
        with _pool_lock:
            if _llm_pool is None:
                _llm_pool = DefaultLLMPool()
    return _llm_pool


def set_llm_pool(pool: DefaultLLMPool) -> None:
    """设置全局 LLM 池"""
    global _llm_pool
    with _pool_lock:
        _llm_pool = pool
