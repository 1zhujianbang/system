import random
import json
import os
import time
from ..utils.tool_function import tools
tools=tools()
from typing import Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from ..core import ConfigManager, get_key_manager
class LLMAPIPool:
    def __init__(self):
        PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
        dotenv_path = PROJECT_ROOT / "config" / ".env.local"
        load_dotenv(dotenv_path)
        self.clients = []
        self._load_clients()
        if not self.clients:
            raise ValueError("[LLM请求] ❌ 未配置任何有效的 LLM API")

    def _load_clients(self):
        """从安全的密钥管理器加载API客户端"""
        try:
            key_manager = get_key_manager()

            # 获取配置管理器来获取服务列表和模型信息
            config_manager = ConfigManager()

            # 首先尝试从配置获取服务列表（不包含密钥）
            services_config = config_manager.get_config_value("llm_services", None, "agent1_config")

            if not services_config:
                # 如果没有配置服务列表，尝试从环境变量获取（向后兼容）
                legacy_config = os.getenv("AGENT1_LLM_APIS")
                if legacy_config:
                    try:
                        legacy_apis = json.loads(legacy_config)
                        services_config = []
                        for cfg in legacy_apis:
                            if cfg.get("enabled", True):
                                # 将旧配置迁移到新系统
                                service_name = f"llm_{cfg['name'].lower()}"
                                api_key = cfg["api_key"]
                                metadata = {
                                    "base_url": cfg["base_url"],
                                    "model": cfg["model"],
                                    "enabled": True
                                }
                                key_manager.store_api_key(service_name, api_key, metadata)
                                services_config.append({
                                    "name": cfg["name"],
                                    "service_key": service_name
                                })
                        tools.log("[LLM请求] ✅ 已迁移旧配置到安全存储")
                    except Exception as e:
                        tools.log(f"[LLM请求] ⚠️ 迁移旧配置失败: {e}")

            if not services_config:
                tools.log("[LLM请求] ❌ 未找到 LLM 服务配置")
                return

            # 从安全的密钥管理器加载每个服务
            for service in services_config:
                service_name = service["name"]
                service_key = service.get("service_key", f"llm_{service_name.lower()}")

                try:
                    # 从密钥管理器获取API密钥
                    api_key = key_manager.get_api_key(service_key)
                    if not api_key:
                        tools.log(f"[LLM请求] ⚠️ 未找到服务 {service_name} 的API密钥")
                        continue

                    # 获取服务元数据
                    key_info = key_manager.get_key_info(service_key)
                    if not key_info:
                        tools.log(f"[LLM请求] ⚠️ 未找到服务 {service_name} 的元数据")
                        continue

                    metadata = key_info.get("metadata", {})
                    base_url = metadata.get("base_url")
                    model = metadata.get("model")

                    # 创建OpenAI客户端
                    client = OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )

                    self.clients.append({
                        "name": service_name,
                        "client": client,
                        "model": model,
                        "service_key": service_key
                    })

                    tools.log(f"[LLM请求] ✅ 已加载服务 {service_name} (模型: {model})")

                except Exception as e:
                    tools.log(f"[LLM请求] ⚠️ 加载服务 {service_name} 失败: {e}")

        except Exception as e:
            tools.log(f"[LLM请求] ❌ 初始化API客户端失败: {e}")

    def add_service(self, name: str, api_key: str, base_url: str,
                   model: str, enabled: bool = True) -> bool:
        """
        添加新的LLM服务

        Args:
            name: 服务显示名称
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            enabled: 是否启用

        Returns:
            是否添加成功
        """
        try:
            key_manager = get_key_manager()
            service_key = f"llm_{name.lower()}"

            metadata = {
                "base_url": base_url,
                "model": model,
                "enabled": enabled,
                "type": "llm"
            }

            key_manager.store_api_key(service_key, api_key, metadata)

            # 重新加载客户端
            self.clients = []
            self._load_clients()

            tools.log(f"[LLM请求] ✅ 已添加服务 {name}")
            return True

        except Exception as e:
            tools.log(f"[LLM请求] ❌ 添加服务 {name} 失败: {e}")
            return False

    def remove_service(self, name: str) -> bool:
        """
        移除LLM服务

        Args:
            name: 服务名称

        Returns:
            是否移除成功
        """
        try:
            key_manager = get_key_manager()
            service_key = f"llm_{name.lower()}"

            if key_manager.delete_api_key(service_key):
                # 重新加载客户端
                self.clients = []
                self._load_clients()
                tools.log(f"[LLM请求] ✅ 已移除服务 {name}")
                return True
            else:
                tools.log(f"[LLM请求] ⚠️ 服务 {name} 不存在")
                return False

        except Exception as e:
            tools.log(f"[LLM请求] ❌ 移除服务 {name} 失败: {e}")
            return False

    def list_services(self) -> List[Dict[str, Any]]:
        """
        列出所有LLM服务（不包含敏感信息）

        Returns:
            服务信息列表
        """
        services = []
        key_manager = get_key_manager()

        for client in self.clients:
            service_key = client["service_key"]
            key_info = key_manager.get_key_info(service_key)

            if key_info:
                services.append({
                    "name": client["name"],
                    "model": client["model"],
                    "base_url": key_info["metadata"].get("base_url"),
                    "enabled": key_info["metadata"].get("enabled", True),
                    "created_at": key_info.get("created_at"),
                    "last_used": key_info.get("last_used")
                })

        return services

    def call(self, prompt: str, max_tokens: int = 1500, timeout: int = 55, retries: int = 2) -> Optional[str]:
        """
        尝试调用 API 池中的服务，直到成功或耗尽重试次数。
        返回 raw LLM content (str)，由调用方解析 JSON。
        """
        available = self.clients.copy()
        if not available:
            return None

        for attempt in range(retries + 1):
            if not available:
                available = self.clients.copy()  # 重置候选池

            # 随机选一个（简单负载均衡），也可改为 round-robin
            choice = random.choice(available)
            name, client, model = choice["name"], choice["client"], choice["model"]

            try:
                tools.log(f"[LLM请求] 尝试 API [{name}] (第 {attempt+1} 次)")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    timeout=timeout,
                    stream=False
                )
                content = response.choices[0].message.content.strip()
                tools.log(f"[LLM请求] ✅ API [{name}] 成功返回")
                return content

            except Exception as e:
                tools.log(f"[LLM请求] ❌ API [{name}] 失败: {e}")
                available.remove(choice)  # 临时剔除故障节点
                if attempt < retries and len(available) == 0:
                    available = self.clients.copy()  # 无可用时重新启用所有

            time.sleep(2 ** attempt)  # 指数退避

        tools.log("[LLM请求] 💥 所有 API 尝试均失败")
        return None