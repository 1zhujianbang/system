"""
Unit tests for enhanced ConfigManager
"""

import pytest
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.config import ConfigManager


class TestConfigManager:
    """测试增强版配置管理器"""

    def setup_method(self):
        """每个测试前的设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        # 创建测试配置
        self.test_config = {
            "max_workers": 5,
            "rate_limit_per_sec": 2.0,
            "entity_batch_size": 50,
            "llm_services": [
                {"name": "openai", "service_key": "llm_openai"},
                {"name": "anthropic", "service_key": "llm_anthropic"}
            ],
            "agent1_config": {
                "max_workers": 3,
                "rate_limit_per_sec": 1.0
            }
        }

    def teardown_method(self):
        """每个测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_config_validation(self):
        """测试配置验证"""
        cm = ConfigManager()

        # 有效的配置
        valid_config = {
            "max_workers": 5,
            "rate_limit_per_sec": 2.0,
            "entity_batch_size": 50
        }
        is_valid, errors = cm.validate_config(valid_config)
        assert is_valid
        assert len(errors) == 0

        # 无效的配置
        invalid_config = {
            "max_workers": 150,  # 超过最大值
            "rate_limit_per_sec": -1.0,  # 小于最小值
            "invalid_field": "test"  # 不在模式中（但不强制要求）
        }
        is_valid, errors = cm.validate_config(invalid_config)
        assert not is_valid
        assert len(errors) >= 2  # 应该有两个错误

    def test_config_override_from_env(self):
        """测试环境变量覆盖"""
        cm = ConfigManager()

        original_config = {
            "max_workers": 5,
            "rate_limit_per_sec": 2.0,
            "debug": False
        }

        # 模拟环境变量
        env_vars = {
            "MAX_WORKERS": "10",
            "DEBUG": "true",
            "INVALID_VAR": "ignored"  # 不在配置中的变量
        }

        with patch.dict(os.environ, env_vars, clear=False):
            overridden = cm.override_from_env(original_config)

        assert overridden["max_workers"] == 10  # 被环境变量覆盖
        assert overridden["rate_limit_per_sec"] == 2.0  # 未被覆盖
        assert overridden["debug"] is True  # 类型转换成功

    def test_config_security_check(self):
        """测试配置安全性检查"""
        cm = ConfigManager()

        # 包含敏感信息的配置
        config_with_secrets = {
            "api_key": "sk-1234567890abcdef",
            "password": "secret123",
            "normal_field": "value",
            "nested": {
                "token": "abc123def456",
                "another_secret": "hidden_value"
            }
        }

        warnings = cm.check_security(config_with_secrets)
        assert len(warnings) >= 3  # 应该检测到敏感信息

        # 正常的配置
        normal_config = {
            "max_workers": 5,
            "timeout": 30
        }

        warnings = cm.check_security(normal_config)
        assert len(warnings) == 0

    def test_config_hash(self):
        """测试配置哈希"""
        # 创建测试配置文件
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)

        cm = ConfigManager()
        cm._config_dir = self.config_dir
        cm._config_path = config_file

        # 获取哈希
        hash1 = cm.get_config_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 哈希长度

        # 修改配置文件后哈希应该改变
        self.test_config["max_workers"] = 10
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)

        hash2 = cm.get_config_hash()
        assert hash1 != hash2

    @patch('src.core.config.Observer')
    def test_hot_reload(self, mock_observer_class):
        """测试热重载功能"""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        cm = ConfigManager()
        cm._config_dir = self.config_dir

        # 启用热重载
        callback = MagicMock()
        cm.enable_hot_reload(callback)

        # 验证观察器被正确配置
        mock_observer_class.assert_called_once()
        mock_observer.schedule.assert_called_once()
        mock_observer.start.assert_called_once()

        # 禁用热重载
        cm.disable_hot_reload()
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()

    def test_config_value_with_validation(self):
        """测试带验证的配置值获取"""
        # 创建测试配置文件
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)

        cm = ConfigManager()
        cm._config_dir = self.config_dir
        cm._config_path = config_file

        # 测试有效的值
        value = cm.get_validated_config_value("max_workers", 3, min_val=1, max_val=10)
        assert value == 5

        # 测试超出范围的值（应该返回默认值）
        value = cm.get_validated_config_value("max_workers", 3, min_val=10, max_val=20)
        assert value == 3  # 返回默认值


class TestConfigManagerIntegration:
    """集成测试"""

    def setup_method(self):
        """集成测试设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """清理"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_multi_file_config_loading(self):
        """测试多文件配置加载"""
        # 创建基础配置
        base_config = {
            "system": {
                "debug": False,
                "log_level": "INFO"
            },
            "max_workers": 4
        }

        # 创建代理配置
        agent_config = {
            "agent1_config": {
                "max_workers": 2,
                "rate_limit_per_sec": 1.0
            },
            "agent2_config": {
                "max_workers": 3,
                "rate_limit_per_sec": 1.5
            }
        }

        # 写入配置文件
        base_file = self.config_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)

        agents_dir = self.config_dir / "agents"
        agents_dir.mkdir(exist_ok=True)
        agent_file = agents_dir / "agents.yaml"
        with open(agent_file, 'w') as f:
            yaml.dump(agent_config, f)

        cm = ConfigManager()
        cm._config_dir = self.config_dir

        # 测试多文件配置加载
        config = cm.load_multi_file_config()

        assert config["system"]["debug"] is False
        assert config["max_workers"] == 4
        assert config["agent1_config"]["max_workers"] == 2
        assert config["agent2_config"]["rate_limit_per_sec"] == 1.5

    def test_config_validation_with_nested(self):
        """测试嵌套配置验证"""
        cm = ConfigManager()

        # 创建包含嵌套结构的配置
        nested_config = {
            "llm_services": [
                {"name": "openai", "service_key": "llm_openai"},
                {"name": "anthropic", "service_key": "llm_anthropic"}
            ],
            "max_workers": 5
        }

        is_valid, errors = cm.validate_config(nested_config)
        assert is_valid
        assert len(errors) == 0

        # 测试无效的嵌套配置
        invalid_nested = {
            "llm_services": [
                {"invalid_field": "test"},  # 缺少必需的name字段
                {"name": 123}  # name字段类型错误
            ]
        }

        is_valid, errors = cm.validate_config(invalid_nested)
        assert not is_valid
        assert len(errors) >= 2  # 应该有多个错误


if __name__ == "__main__":
    pytest.main([__file__])