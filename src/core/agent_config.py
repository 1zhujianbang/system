"""
统一的Agent配置加载器
避免在各个agent中重复配置加载代码
"""

from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from .config import ConfigManager
from .logging import LoggerManager


class AgentConfigLoader:
    """Agent配置加载器"""
    
    def __init__(self, agent_name: str, config_dir: Path = None):
        """
        初始化配置加载器
        
        Args:
            agent_name: agent名称（如agent1, agent2, agent3）
            config_dir: 配置目录路径
        """
        self.agent_name = agent_name
        self.config_key = f"{agent_name}_config"
        
        # 加载环境变量
        if config_dir:
            dotenv_path = config_dir / ".env.local"
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
        
        self._config_manager = ConfigManager()
        self._logger = LoggerManager.get_logger(__name__)
    
    def get_max_workers(self, default: int = 6) -> int:
        """获取最大并发数"""
        return self._config_manager.get_concurrency_limit(self.config_key) or default
    
    def get_rate_limit(self, default: float = 1.0) -> float:
        """获取速率限制（每秒请求数）"""
        return self._config_manager.get_rate_limit(self.config_key) or default
    
    def get_batch_size(self, batch_type: str = "entity", default: int = 80) -> int:
        """
        获取批处理大小
        
        Args:
            batch_type: 批处理类型（entity或event）
            default: 默认值
        """
        key = f"{batch_type}_batch_size"
        return int(self._config_manager.get_config_value(key, default, self.config_key))
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config_manager.get_config_value(key, default, self.config_key)
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        config_data = self._config_manager.load_multi_file_config()
        return config_data.get(self.config_key, {})
    
    def log_config(self):
        """记录配置信息到日志"""
        config = self.get_all_config()
        self._logger.info(f"[{self.agent_name}] 配置加载:")
        self._logger.info(f"  - max_workers: {self.get_max_workers()}")
        self._logger.info(f"  - rate_limit_per_sec: {self.get_rate_limit()}")
        
        # 记录其他配置
        for key, value in config.items():
            if key not in ['max_workers', 'rate_limit_per_sec']:
                self._logger.info(f"  - {key}: {value}")


def get_agent_config(agent_name: str, config_dir: Path = None) -> AgentConfigLoader:
    """
    快捷函数：获取agent配置加载器
    
    Args:
        agent_name: agent名称
        config_dir: 配置目录
        
    Returns:
        配置加载器实例
    """
    return AgentConfigLoader(agent_name, config_dir)

