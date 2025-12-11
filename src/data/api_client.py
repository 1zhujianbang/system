import os
import json
from pathlib import Path
from typing import Optional, Any, List, Dict

from dotenv import load_dotenv
from .news_collector import GNewsCollector
from ..core.singleton import SingletonBase

def get_apis_config(config_path: Path = None) -> List[dict]:
    """
    获取默认API配置（硬编码作为默认值）
    """
    return [
        {"name": "GNews-cn", "language": "zh", "country": "cn", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-us", "language": "en", "country": "us", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-fr", "language": "fr", "country": "fr", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-gb", "language": "en", "country": "gb", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-hk", "language": "zh", "country": "hk", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-ru", "language": "ru", "country": "ru", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-ua", "language": "uk", "country": "ua", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-tw", "language": "zh", "country": "tw", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-sg", "language": "en", "country": "sg", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-jp", "language": "ja", "country": "jp", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-br", "language": "pt", "country": "br", "timeout": 30, "enabled": True, "type": "gnews"},
        {"name": "GNews-ar", "language": "es", "country": "ar", "timeout": 30, "enabled": True, "type": "gnews"}
    ]

class DataAPIPool(SingletonBase):
    _collectors = {}

    def _init_singleton(self) -> None:
        """单例初始化"""
        # 加载环境变量（主要用于获取API密钥）
        PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
        dotenv_path = PROJECT_ROOT / "config" / ".env.local"
        load_dotenv(dotenv_path)

        self.configs: List[Dict[str, Any]] = []
        self.api_key_pool: List[str] = []  # API密钥池
        self._load_configs()
        # 调试：打印已加载的数据源配置
        print(f"[数据获取][DataAPIPool] 已加载数据源配置: {[c.get('name') for c in self.configs]}")

    def _load_configs(self):
        """
        加载API配置和API池
        """
        try:
            # 获取默认配置
            print(f"[数据获取][DataAPIPool] 加载默认 API 配置")
            apis = get_apis_config()
            
            # 加载API密钥池
            gnews_apis_pool = os.getenv("GNEWS_APIS_POOL")
            self.api_key_pool = []
            if gnews_apis_pool:
                print(f"[数据获取][DataAPIPool] 加载GNEWS_APIS_POOL环境变量")
                # 移除环境变量值可能存在的首尾单引号
                gnews_apis_pool_clean = gnews_apis_pool.strip("'")
                try:
                    self.api_key_pool = json.loads(gnews_apis_pool_clean)
                    print(f"[数据获取][DataAPIPool] 已加载API池，包含 {len(self.api_key_pool)} 个API密钥")
                except json.JSONDecodeError:
                    print(f"[数据获取][DataAPIPool] Error: GNEWS_APIS_POOL 格式错误，应为 JSON 数组")
            else:
                print(f"[数据获取][DataAPIPool] 警告: 未设置 GNEWS_APIS_POOL")
            
            # 为每个启用的配置分配API密钥
            api_key_index = 0
            for cfg in apis:
                if cfg.get("enabled", True):
                    # 如果是GNews类型且没有api_key，则从API池分配
                    # 根据 type 或 name 判断
                    is_gnews = cfg.get("type") == "gnews" or "GNews" in cfg.get("name", "")
                    
                    if is_gnews and "api_key" not in cfg and self.api_key_pool:
                        cfg["api_key"] = self.api_key_pool[api_key_index % len(self.api_key_pool)]
                        api_key_index += 1
                        # print(f"[数据获取][DataAPIPool] 为 {cfg['name']} 分配API密钥")
                    
                    self.configs.append(cfg)
        except Exception as e:
            print(f"[数据获取] ❌ 解析 API 配置失败: {e}")
            raise

    def get_collector(self, name: str) -> Optional[Any]:
        """根据 name 返回对应的新闻收集器实例（单例）"""
        if name in self._collectors:
            # print(f"[数据获取][DataAPIPool] 复用已创建的 collector: {name}")
            return self._collectors[name]

        # 查找配置
        config = None
        for cfg in self.configs:
            if name == cfg["name"]:
                config = cfg
                break

        if not config:
            raise ValueError(f"[数据获取][DataAPIPool] 未找到名为 '{name}' 的数据源配置")

        # 创建对应 collector
        # print(f"[数据获取][DataAPIPool] 准备创建 collector: {name}")
        
        source_type = config.get("type", "").lower()
        if not source_type:
            if "gnews" in name.lower(): source_type = "gnews"

        if source_type == "gnews":
            api_key = config.get("api_key") or os.getenv("GNEWS_API_KEY", "")
            if not api_key:
                raise ValueError("GNews 数据源需要配置 api_key 或环境变量 GNEWS_API_KEY")

            language = config.get("language", "zh")
            country = config.get("country")

            collector = GNewsCollector(
                api_key=api_key,
                language=language,
                country=country,
                timeout=config.get("timeout", 30)
            )
        else:
            raise NotImplementedError(f"不支持的数据源类型: {source_type} ({name})")

        self._collectors[name] = collector
        return collector
        
    def list_available_sources(self) -> List[str]:
        """
        获取所有可用的数据源名称列表
        
        Returns:
            数据源名称列表
        """
        available_sources = []
        for cfg in self.configs:
            if cfg.get("enabled", True):
                available_sources.append(cfg["name"])
        return available_sources
