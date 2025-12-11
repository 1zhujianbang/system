"""
单元测试：核心配置模块
"""
import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import ConfigManager


class TestConfigManager:
    """配置管理器测试类"""

    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        config_manager = ConfigManager()
        assert config_manager is not None
        assert hasattr(config_manager, 'load_multi_file_config')

    def test_load_multi_file_config(self):
        """测试多文件配置加载"""
        config_manager = ConfigManager()
        config = config_manager.load_multi_file_config()

        # 检查配置结构
        assert isinstance(config, dict)
        assert 'agent1_config' in config
        assert 'agent2_config' in config
        assert 'agent3_config' in config

    def test_get_config_value(self):
        """测试配置值获取"""
        config_manager = ConfigManager()

        # 测试默认值
        value = config_manager.get_config_value('test_key', 'default_value', 'agent1_config')
        assert value == 'default_value'

        # 测试真实配置值
        max_workers = config_manager.get_config_value('max_workers', 1, 'agent1_config')
        assert isinstance(max_workers, int)

    def test_get_concurrency_limit(self):
        """测试并发限制获取"""
        config_manager = ConfigManager()
        limit = config_manager.get_concurrency_limit('agent1_config')
        assert isinstance(limit, int)
        assert limit > 0

    def test_get_rate_limit(self):
        """测试速率限制获取"""
        config_manager = ConfigManager()
        rate = config_manager.get_rate_limit('agent1_config')
        assert isinstance(rate, (int, float))

    def test_config_caching(self):
        """测试配置缓存机制"""
        config_manager = ConfigManager()

        # 第一次加载
        config1 = config_manager.load_multi_file_config()
        # 第二次加载（应该从缓存返回）
        config2 = config_manager.load_multi_file_config()

        assert config1 == config2
        assert config1 is config2  # 应该返回同一个对象


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
