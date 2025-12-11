"""
Unit tests for KeyManager - Secure API Key Management
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.key_manager import KeyManager, get_key_manager


class TestKeyManager:
    """测试密钥管理器"""

    def setup_method(self):
        """每个测试前的设置"""
        # 使用临时文件作为密钥存储
        self.temp_dir = Path(tempfile.mkdtemp())
        self.key_store_path = self.temp_dir / ".key_store.enc"

        # 使用固定的主密钥进行测试
        self.master_key = "test_master_key_12345678901234567890123456789012"

        # 预先创建空的存储文件，确保目录存在
        self.key_store_path.parent.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """每个测试后的清理"""
        # 清理临时文件
        if self.key_store_path.exists():
            self.key_store_path.unlink()
        self.temp_dir.rmdir()

    def test_key_manager_initialization(self):
        """测试密钥管理器初始化"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            # 检查是否创建了存储文件
            assert self.key_store_path.exists()
            assert km.fernet is not None

    def test_store_and_get_api_key(self):
        """测试存储和获取API密钥"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            # 确保存储文件已创建
            assert self.key_store_path.exists()

            # 存储API密钥
            service_name = "test_openai"
            api_key = "sk-test1234567890abcdef"
            metadata = {"model": "gpt-4", "base_url": "https://api.openai.com/v1"}

            km.store_api_key(service_name, api_key, metadata)

            # 获取API密钥
            retrieved_key = km.get_api_key(service_name)
            assert retrieved_key == api_key

            # 检查元数据
            key_info = km.get_key_info(service_name)
            assert key_info is not None
            assert key_info["metadata"]["model"] == "gpt-4"
            assert "encrypted_key" not in key_info  # 确保不返回加密密钥

    def test_delete_api_key(self):
        """测试删除API密钥"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            # 存储然后删除
            service_name = "test_service"
            api_key = "test_key_123"

            km.store_api_key(service_name, api_key)
            assert km.get_api_key(service_name) == api_key

            # 删除
            result = km.delete_api_key(service_name)
            assert result is True

            # 确认已删除
            assert km.get_api_key(service_name) is None

    def test_list_services(self):
        """测试列出服务"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            # 存储多个服务
            services = {
                "openai": "sk-openai123",
                "anthropic": "sk-anthropic456",
                "google": "sk-google789"
            }

            for name, key in services.items():
                km.store_api_key(name, key)

            # 列出服务
            service_list = km.list_services()
            assert len(service_list) == 3
            assert "openai" in service_list
            assert "anthropic" in service_list
            assert "google" in service_list

    def test_rotate_master_key(self):
        """测试轮换主密钥"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            # 存储测试密钥
            service_name = "test_rotate"
            api_key = "original_key_123"

            km.store_api_key(service_name, api_key)
            assert km.get_api_key(service_name) == api_key

            # 轮换主密钥
            new_master_key = "new_master_key_12345678901234567890123456789012"
            result = km.rotate_master_key(new_master_key)
            assert result is True

            # 验证密钥仍然可以正确解密
            retrieved_key = km.get_api_key(service_name)
            assert retrieved_key == api_key

    def test_health_check(self):
        """测试健康检查"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            health = km.health_check()
            assert health["status"] == "healthy"
            assert "service_count" in health
            assert "store_path" in health
            assert health["master_key_configured"] is True

    def test_invalid_master_key(self):
        """测试无效的主密钥"""
        # 使用错误的密钥文件内容
        invalid_data = b"invalid_encrypted_data"
        self.key_store_path.write_bytes(invalid_data)

        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            # 应该能够处理损坏的数据并创建新的存储
            health = km.health_check()
            assert health["status"] == "healthy"

    def test_get_nonexistent_key(self):
        """测试获取不存在的密钥"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            result = km.get_api_key("nonexistent_service")
            assert result is None

    def test_get_key_info_nonexistent(self):
        """测试获取不存在密钥的信息"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            result = km.get_key_info("nonexistent_service")
            assert result is None

    @patch('src.core.key_manager.KeyManager._save_master_key')
    @patch('src.core.key_manager.KeyManager._generate_master_key')
    def test_master_key_generation(self, mock_generate, mock_save):
        """测试主密钥生成"""
        mock_generate.return_value = "generated_master_key_1234567890"

        with patch.dict('os.environ', {}, clear=True):
            # 模拟没有环境变量和密钥文件的情况
            with patch('pathlib.Path.exists', return_value=False):
                km = KeyManager(key_store_path=self.key_store_path)

                mock_generate.assert_called_once()
                mock_save.assert_called_once_with("generated_master_key_1234567890")


class TestKeyManagerIntegration:
    """集成测试"""

    def setup_method(self):
        """集成测试设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.key_store_path = self.temp_dir / ".key_store.enc"
        self.master_key = "integration_test_master_key_123456789012"

    def teardown_method(self):
        """清理"""
        if self.key_store_path.exists():
            self.key_store_path.unlink()
        self.temp_dir.rmdir()

    def test_full_workflow(self):
        """测试完整的工作流程"""
        with patch('src.core.key_manager.KeyManager._get_master_key', return_value=self.master_key):
            km = KeyManager(key_store_path=self.key_store_path)

            # 1. 存储多个API密钥
            services = {
                "openai_gpt4": "sk-gpt4-abcdef123456",
                "anthropic_claude": "sk-ant-abcdef123456",
                "google_gemini": "sk-ggl-abcdef123456"
            }

            metadata_list = {
                "openai_gpt4": {"model": "gpt-4", "provider": "openai"},
                "anthropic_claude": {"model": "claude-3", "provider": "anthropic"},
                "google_gemini": {"model": "gemini-pro", "provider": "google"}
            }

            for name, key in services.items():
                km.store_api_key(name, key, metadata_list[name])

            # 2. 验证所有密钥都可以正确获取
            for name, expected_key in services.items():
                retrieved = km.get_api_key(name)
                assert retrieved == expected_key

            # 3. 验证元数据
            for name, expected_metadata in metadata_list.items():
                info = km.get_key_info(name)
                assert info is not None
                for k, v in expected_metadata.items():
                    assert info["metadata"][k] == v

            # 4. 测试服务列表
            service_list = km.list_services()
            assert len(service_list) == 3
            assert set(service_list) == set(services.keys())

            # 5. 删除一个服务
            km.delete_api_key("anthropic_claude")
            assert km.get_api_key("anthropic_claude") is None
            assert km.get_api_key("openai_gpt4") == services["openai_gpt4"]

            # 6. 健康检查
            health = km.health_check()
            assert health["status"] == "healthy"
            assert health["service_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
