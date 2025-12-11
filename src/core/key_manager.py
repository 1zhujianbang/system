"""
Secure Key Manager - API密钥的安全管理
提供加密存储、密钥轮换和访问控制功能
"""

import os
import json
import base64
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Optional, Any, List
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from .logging import LoggerManager
from .singleton import singleton


@singleton
class KeyManager:
    """安全的API密钥管理器"""

    def __init__(self, key_store_path: Optional[Path] = None, master_key: Optional[str] = None):
        """
        初始化密钥管理器

        Args:
            key_store_path: 密钥存储文件路径
            master_key: 主密钥，如果不提供则从环境变量获取
        """
        self.logger = LoggerManager.get_logger(__name__)

        # 默认存储路径
        if key_store_path is None:
            from ..utils.tool_function import tools
            key_store_path = tools.CONFIG_DIR / ".key_store.enc"

        self.key_store_path = key_store_path
        self.master_key = master_key or self._get_master_key()
        self.fernet = self._derive_key(self.master_key)

        # 密钥缓存
        self._key_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False

        # 初始化存储文件（必须在fernet之后）
        self._ensure_key_store()

    def _get_master_key(self) -> str:
        """获取主密钥"""
        # 优先从环境变量获取
        master_key = os.getenv('NEWS_AGENT_MASTER_KEY')

        if not master_key:
            # 如果没有环境变量，尝试从文件读取
            key_file = Path.home() / '.news_agent' / 'master_key'
            if key_file.exists():
                try:
                    master_key = key_file.read_text().strip()
                except Exception as e:
                    self.logger.warning(f"Failed to read master key from file: {e}")

        if not master_key:
            # 生成新的主密钥
            self.logger.warning("No master key found, generating new one")
            master_key = self._generate_master_key()
            self._save_master_key(master_key)

        return master_key

    def _generate_master_key(self) -> str:
        """生成新的主密钥"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def _save_master_key(self, master_key: str):
        """保存主密钥到文件"""
        key_dir = Path.home() / '.news_agent'
        key_dir.mkdir(parents=True, exist_ok=True)
        key_file = key_dir / 'master_key'
        key_file.write_text(master_key)
        key_file.chmod(0o600)  # 只允许所有者读写

    def _derive_key(self, master_key: str) -> Fernet:
        """从主密钥派生加密密钥"""
        # 使用PBKDF2派生密钥
        salt = b'news_agent_salt_2024'  # 固定盐值
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    def _ensure_key_store(self):
        """确保密钥存储文件存在"""
        try:
            if not self.key_store_path.exists():
                self.key_store_path.parent.mkdir(parents=True, exist_ok=True)
                # 创建空的加密存储
                empty_data = {"keys": {}, "metadata": {}}
                encrypted_data = self.fernet.encrypt(json.dumps(empty_data).encode())
                self.key_store_path.write_bytes(encrypted_data)
                # 在Windows上设置文件权限（如果可能）
                try:
                    self.key_store_path.chmod(0o600)
                except OSError:
                    # Windows可能不支持chmod，忽略这个错误
                    pass
        except Exception as e:
            self.logger.warning(f"Failed to ensure key store: {e}")

    def _load_key_store(self) -> Dict[str, Any]:
        """加载密钥存储"""
        try:
            encrypted_data = self.key_store_path.read_bytes()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except (InvalidToken, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load key store: {e}")
            # 返回空的存储结构
            return {"keys": {}, "metadata": {}}

    def _save_key_store(self, data: Dict[str, Any]):
        """保存密钥存储"""
        try:
            encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
            self.key_store_path.write_bytes(encrypted_data)
        except Exception as e:
            self.logger.error(f"Failed to save key store: {e}")

    def store_api_key(self, service_name: str, api_key: str, metadata: Optional[Dict[str, Any]] = None):
        """
        存储API密钥

        Args:
            service_name: 服务名称，如 'openai', 'gnews'
            api_key: API密钥
            metadata: 元数据，如使用限制、过期时间等
        """
        data = self._load_key_store()

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]  # 短哈希用于标识

        data["keys"][service_name] = {
            "key_hash": key_hash,
            "encrypted_key": self.fernet.encrypt(api_key.encode()).decode(),
            "metadata": metadata or {},
            "created_at": self._get_timestamp(),
            "last_used": None
        }

        self._save_key_store(data)
        self._key_cache[service_name] = data["keys"][service_name]

        self.logger.info(f"API key for {service_name} stored securely")

    def get_api_key(self, service_name: str) -> Optional[str]:
        """
        获取API密钥

        Args:
            service_name: 服务名称

        Returns:
            解密的API密钥，如果不存在返回None
        """
        if service_name in self._key_cache:
            cached = self._key_cache[service_name]
        else:
            data = self._load_key_store()
            cached = data.get("keys", {}).get(service_name)
            if cached:
                self._key_cache[service_name] = cached

        if not cached:
            return None

        try:
            # 解密密钥
            encrypted_key = cached["encrypted_key"]
            decrypted_key = self.fernet.decrypt(encrypted_key.encode()).decode()

            # 更新最后使用时间
            self._update_last_used(service_name)

            return decrypted_key
        except (InvalidToken, KeyError) as e:
            self.logger.error(f"Failed to decrypt API key for {service_name}: {e}")
            return None

    def _update_last_used(self, service_name: str):
        """更新密钥最后使用时间"""
        data = self._load_key_store()
        if service_name in data.get("keys", {}):
            data["keys"][service_name]["last_used"] = self._get_timestamp()
            self._save_key_store(data)
            # 更新缓存
            if service_name in self._key_cache:
                self._key_cache[service_name]["last_used"] = data["keys"][service_name]["last_used"]

    def delete_api_key(self, service_name: str) -> bool:
        """删除API密钥"""
        data = self._load_key_store()
        if service_name in data.get("keys", {}):
            del data["keys"][service_name]
            self._save_key_store(data)
            self._key_cache.pop(service_name, None)
            self.logger.info(f"API key for {service_name} deleted")
            return True
        return False

    def list_services(self) -> List[str]:
        """列出所有存储的服务名称"""
        data = self._load_key_store()
        return list(data.get("keys", {}).keys())

    def get_key_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取密钥信息（不包含实际密钥）"""
        data = self._load_key_store()
        key_info = data.get("keys", {}).get(service_name)
        if key_info:
            # 返回不包含实际密钥的信息
            info = key_info.copy()
            info.pop("encrypted_key", None)
            return info
        return None

    def rotate_master_key(self, new_master_key: Optional[str] = None) -> bool:
        """
        轮换主密钥

        Args:
            new_master_key: 新的主密钥，如果不提供则生成新的

        Returns:
            是否成功轮换
        """
        try:
            # 加载所有现有密钥
            old_data = self._load_key_store()

            # 生成新的主密钥
            if not new_master_key:
                new_master_key = self._generate_master_key()

            # 创建新的Fernet实例
            new_fernet = self._derive_key(new_master_key)

            # 重新加密所有密钥
            new_data = {"keys": {}, "metadata": old_data.get("metadata", {})}

            for service_name, key_data in old_data.get("keys", {}).items():
                # 解密旧密钥
                old_encrypted = key_data["encrypted_key"]
                decrypted_key = self.fernet.decrypt(old_encrypted.encode()).decode()

                # 用新密钥重新加密
                new_encrypted = new_fernet.encrypt(decrypted_key.encode()).decode()

                new_data["keys"][service_name] = key_data.copy()
                new_data["keys"][service_name]["encrypted_key"] = new_encrypted

            # 保存新数据
            self.fernet = new_fernet
            self.master_key = new_master_key
            self._save_key_store(new_data)
            self._save_master_key(new_master_key)

            # 清空缓存
            self._key_cache.clear()

            self.logger.info("Master key rotated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to rotate master key: {e}")
            return False

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试加密/解密功能
            test_data = "test_key_123"
            encrypted = self.fernet.encrypt(test_data.encode())
            decrypted = self.fernet.decrypt(encrypted).decode()

            if decrypted != test_data:
                return {"status": "error", "message": "Encryption/decryption test failed"}

            # 检查存储文件
            if not self.key_store_path.exists():
                return {"status": "error", "message": "Key store file does not exist"}

            # 加载存储检查
            data = self._load_key_store()
            service_count = len(data.get("keys", {}))

            return {
                "status": "healthy",
                "service_count": service_count,
                "store_path": str(self.key_store_path),
                "master_key_configured": bool(self.master_key)
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}


# 全局实例
def get_key_manager() -> KeyManager:
    """获取密钥管理器实例"""
    return KeyManager()


# 便捷函数
def store_api_key(service_name: str, api_key: str, metadata: Optional[Dict[str, Any]] = None):
    """存储API密钥的便捷函数"""
    get_key_manager().store_api_key(service_name, api_key, metadata)


def get_api_key(service_name: str) -> Optional[str]:
    """获取API密钥的便捷函数"""
    return get_key_manager().get_api_key(service_name)
