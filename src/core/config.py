"""
ConfigManager - 增强版配置管理器
提供统一的配置读取、缓存、验证和安全功能
"""

import time
import os
import yaml
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .logging import LoggerManager
from .singleton import singleton


class ConfigManager:
    """增强版配置管理器"""

    def __init__(self, cache_ttl: int = 300, logger=None):
        self._cache_ttl = cache_ttl
        self._config_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        # 延迟导入以避免循环依赖
        from ..utils.tool_function import tools
        self._config_dir = tools.CONFIG_DIR
        self._config_path = self._config_dir / "config.yaml"  # 向后兼容
        self._logger = logger

    def get_config_value(self, key_path: str, default=None, agent_config: str = None) -> Any:
        """
        获取配置值，支持点分隔路径访问

        Args:
            key_path: 配置路径，如 "max_workers" 或 "agent1_config.max_workers"
            default: 默认值
            agent_config: 代理配置前缀，如 "agent1_config"

        Returns:
            配置值或默认值
        """
        cache_key = f"{agent_config}_{key_path}" if agent_config else key_path

        # 检查缓存是否过期
        if cache_key in self._cache_timestamps:
            if time.time() - self._cache_timestamps[cache_key] > self._cache_ttl:
                self.invalidate_cache(cache_key)

        # 缓存未命中或已过期
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = self._load_config_value(key_path, default, agent_config)
            self._cache_timestamps[cache_key] = time.time()

        return self._config_cache[cache_key]

    def _load_config_value(self, key_path: str, default: Any, agent_config: str = None) -> Any:
        """实际加载配置值的逻辑"""
        try:
            # 优先使用多文件配置加载
            config_data = self.load_multi_file_config()

            # 如果多文件配置为空，尝试单一文件
            if not config_data:
                if not self._config_path.exists():
                    if self._logger:
                        self._logger.warning(f"配置文件不存在: {self._config_path}")
                    return default

                with open(self._config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}

            # 如果指定了代理配置，获取对应section
            if agent_config:
                agent_cfg = config_data.get(agent_config, {}) or {}
                if not isinstance(agent_cfg, dict):
                    return default
                config_data = agent_cfg

            # 解析点分隔路径
            value = config_data
            for part in key_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                    if value is None:
                        return default
                else:
                    return default

            return value if value is not None else default

        except Exception as e:
            if self._logger:
                self._logger.error(f"加载配置失败 {key_path}: {e}")
            return default

    def invalidate_cache(self, key: str = None):
        """使缓存失效"""
        if key:
            self._config_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
            if self._logger:
                self._logger.debug(f"Invalidated cache for key: {key}")
        else:
            self._config_cache.clear()
            self._cache_timestamps.clear()
            if self._logger:
                self._logger.debug("Invalidated all config cache")

    def reload_config(self):
        """强制重新加载所有配置"""
        self.invalidate_cache()
        if self._logger:
            self._logger.info("Configuration reloaded")

    def load_multi_file_config(self) -> Dict[str, Any]:
        """
        加载多文件配置结构

        加载顺序:
        1. base.yaml (基础配置)
        2. agents/*.yaml (各代理配置)
        3. functions/*.yaml (功能配置，如有)

        Returns:
            合并后的配置字典
        """
        merged_config: Dict[str, Any] = {}

        # 1. 加载基础配置
        base_config = self._load_yaml_file(self._config_dir / "base.yaml")
        if base_config:
            merged_config.update(base_config)

        # 2. 加载代理配置
        agents_dir = self._config_dir / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.yaml"):
                agent_config = self._load_yaml_file(agent_file)
                if agent_config:
                    merged_config.update(agent_config)

        # 3. 加载功能配置 (如果将来需要)
        functions_dir = self._config_dir / "functions"
        if functions_dir.exists():
            for func_file in functions_dir.glob("*.yaml"):
                func_config = self._load_yaml_file(func_file)
                if func_config:
                    merged_config.update(func_config)

        # 4. 向后兼容：如果没有多文件配置，尝试加载单一config.yaml
        if not merged_config and self._config_path.exists():
            single_config = self._load_yaml_file(self._config_path)
            if single_config:
                merged_config.update(single_config)

        return merged_config

    def _load_yaml_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """加载单个YAML文件"""
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if self._logger:
                    self._logger.debug(f"Loaded config from {file_path.name}")
                return config_data if isinstance(config_data, dict) else {}
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to load config from {file_path}: {e}")
            return {}

    def get_concurrency_limit(self, agent_config: str = "agent1_config") -> int:
        """获取并发限制"""
        return int(self.get_config_value("max_workers", 6, agent_config))

    def get_rate_limit(self, agent_config: str = "agent1_config") -> float:
        """获取速率限制"""
        return float(self.get_config_value("rate_limit_per_sec", 20.0, agent_config))

    def get_entity_batch_size(self, agent_config: str = "agent3_config") -> int:
        """获取实体批处理大小"""
        return int(self.get_config_value("entity_batch_size", 80, agent_config))

    def get_event_batch_size(self, agent_config: str = "agent3_config") -> int:
        """获取事件批处理大小"""
        return int(self.get_config_value("event_batch_size", 15, agent_config))

    def validate_numeric_range(self, value: Union[int, float], min_val: Optional[Union[int, float]] = None,
                              max_val: Optional[Union[int, float]] = None, field_name: str = "value") -> bool:
        """
        验证数值范围

        Args:
            value: 要验证的值
            min_val: 最小值（包含）
            max_val: 最大值（包含）
            field_name: 字段名称，用于错误信息

        Returns:
            验证是否通过
        """
        if min_val is not None and value < min_val:
            if self._logger:
                self._logger.warning(f"配置验证失败: {field_name}={value} 小于最小值 {min_val}")
            return False

        if max_val is not None and value > max_val:
            if self._logger:
                self._logger.warning(f"配置验证失败: {field_name}={value} 大于最大值 {max_val}")
            return False

        return True

    def validate_config_consistency(self, config_data: Dict[str, Any]) -> List[str]:
        """
        验证配置一致性

        Args:
            config_data: 配置数据

        Returns:
            验证错误列表，空列表表示验证通过
        """
        errors = []

        # 验证agent配置一致性
        for agent in ["agent1_config", "agent2_config", "agent3_config"]:
            if agent in config_data:
                agent_config = config_data[agent]

                # 验证max_workers范围
                if "max_workers" in agent_config:
                    workers = agent_config["max_workers"]
                    if not self.validate_numeric_range(workers, 1, 20, f"{agent}.max_workers"):
                        errors.append(f"{agent}.max_workers 必须在1-20之间")

                # 验证rate_limit范围
                if "rate_limit_per_sec" in agent_config:
                    rate_limit = agent_config["rate_limit_per_sec"]
                    if not self.validate_numeric_range(rate_limit, 0.1, 100.0, f"{agent}.rate_limit_per_sec"):
                        errors.append(f"{agent}.rate_limit_per_sec 必须在0.1-100.0之间")

        # 验证agent3特殊配置
        if "agent3_config" in config_data:
            agent3_config = config_data["agent3_config"]

            # 验证批处理大小
            for field in ["entity_batch_size", "event_batch_size"]:
                if field in agent3_config:
                    batch_size = agent3_config[field]
                    if not self.validate_numeric_range(batch_size, 1, 200, f"agent3_config.{field}"):
                        errors.append(f"agent3_config.{field} 必须在1-200之间")

            # 验证相似度阈值
            for field in ["entity_precluster_similarity", "event_precluster_similarity"]:
                if field in agent3_config:
                    similarity = agent3_config[field]
                    if not self.validate_numeric_range(similarity, 0.0, 1.0, f"agent3_config.{field}"):
                        errors.append(f"agent3_config.{field} 必须在0.0-1.0之间")

        return errors

    def load_and_validate_config(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        加载并验证配置

        Returns:
            (配置数据, 验证错误列表)
        """
        config_data = self.load_multi_file_config()
        validation_errors = self.validate_config_consistency(config_data)

        if validation_errors and self._logger:
            self._logger.warning(f"配置验证发现 {len(validation_errors)} 个问题:")
            for error in validation_errors:
                self._logger.warning(f"  - {error}")

        return config_data, validation_errors

    def get_validated_config_value(self, key_path: str, default: Any = None,
                                 agent_config: str = None, min_val: Optional[Union[int, float]] = None,
                                 max_val: Optional[Union[int, float]] = None) -> Any:
        """
        获取并验证配置值

        Args:
            key_path: 配置路径
            default: 默认值
            agent_config: 代理配置前缀
            min_val: 最小值
            max_val: 最大值

        Returns:
            验证后的配置值
        """
        value = self.get_config_value(key_path, default, agent_config)

        if value is not None and (min_val is not None or max_val is not None):
            field_name = f"{agent_config}.{key_path}" if agent_config else key_path
            if not self.validate_numeric_range(value, min_val, max_val, field_name):
                if self._logger:
                    self._logger.warning(f"使用默认值 {default} 替代无效配置值 {value}")
                return default

        return value

    def validate_config(self, config_data: Dict[str, Any], schema: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        验证配置数据

        Args:
            config_data: 配置数据
            schema: 验证模式，如果为None则使用默认模式

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        if schema is None:
            schema = self._get_default_schema()

        def validate_section(section_data: Dict[str, Any], section_schema: Dict[str, Any], path: str = ""):
            for key, rules in section_schema.items():
                full_path = f"{path}.{key}" if path else key

                if key not in section_data:
                    if rules.get("required", False):
                        errors.append(f"Missing required configuration: {full_path}")
                    continue

                value = section_data[key]
                expected_type = rules.get("type")

                # 类型检查
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Invalid type for {full_path}: expected {expected_type.__name__}, got {type(value).__name__}")

                # 范围检查
                if "min" in rules and isinstance(value, (int, float)):
                    if value < rules["min"]:
                        errors.append(f"Value too small for {full_path}: {value} < {rules['min']}")

                if "max" in rules and isinstance(value, (int, float)):
                    if value > rules["max"]:
                        errors.append(f"Value too large for {full_path}: {value} > {rules['max']}")

                # 枚举检查
                if "choices" in rules and value not in rules["choices"]:
                    errors.append(f"Invalid value for {full_path}: {value} not in {rules['choices']}")

                # 递归验证嵌套配置
                if "nested" in rules:
                    if isinstance(value, dict):
                        validate_section(value, rules["nested"], full_path)
                    elif isinstance(value, list) and rules["nested"].get("*"):
                        # 处理列表类型，验证每个元素
                        item_schema = rules["nested"]["*"]
                        for i, item in enumerate(value):
                            item_path = f"{full_path}[{i}]"
                            if isinstance(item, dict):
                                validate_section(item, item_schema["nested"], item_path)

        validate_section(config_data, schema)
        return len(errors) == 0, errors

    def _get_default_schema(self) -> Dict[str, Any]:
        """获取默认配置验证模式"""
        return {
            "max_workers": {
                "type": int,
                "min": 1,
                "max": 100,
                "required": False
            },
            "rate_limit_per_sec": {
                "type": (int, float),
                "min": 0.1,
                "max": 1000.0,
                "required": False
            },
            "entity_batch_size": {
                "type": int,
                "min": 10,
                "max": 1000,
                "required": False
            },
            "event_batch_size": {
                "type": int,
                "min": 5,
                "max": 500,
                "required": False
            },
            "llm_services": {
                "type": list,
                "required": False,
                "nested": {
                    "*": {  # 任意键名
                        "type": dict,
                        "nested": {
                            "name": {"type": str, "required": True},
                            "service_key": {"type": str, "required": False}
                        }
                    }
                }
            }
        }

    def enable_hot_reload(self, callback: Optional[Callable] = None):
        """
        启用配置热重载

        Args:
            callback: 配置变更时的回调函数
        """
        if hasattr(self, '_hot_reload_enabled') and self._hot_reload_enabled:
            return

        self._hot_reload_enabled = True
        self._hot_reload_callback = callback

        # 创建文件监控器
        event_handler = ConfigFileHandler(self._config_dir, self._on_config_changed)
        self._observer = Observer()
        self._observer.schedule(event_handler, str(self._config_dir), recursive=True)
        self._observer.start()

        if self._logger:
            self._logger.info("Configuration hot reload enabled")

    def disable_hot_reload(self):
        """禁用配置热重载"""
        if hasattr(self, '_observer'):
            self._observer.stop()
            self._observer.join()
            self._hot_reload_enabled = False
            if self._logger:
                self._logger.info("Configuration hot reload disabled")

    def _on_config_changed(self, event):
        """配置文件变更处理"""
        if event.is_directory:
            return

        # 检查是否是配置文件
        config_files = ['config.yaml', 'base.yaml'] + [f for f in os.listdir(self._config_dir) if f.endswith('.yaml')]
        if Path(event.src_path).name in config_files:
            if self._logger:
                self._logger.info(f"Configuration file changed: {event.src_path}")

            # 清空缓存
            self.invalidate_cache()

            # 调用回调函数
            if self._hot_reload_callback:
                try:
                    self._hot_reload_callback()
                except Exception as e:
                    if self._logger:
                        self._logger.error(f"Hot reload callback failed: {e}")

    def get_config_hash(self) -> str:
        """
        获取配置文件的哈希值，用于检测配置变更

        Returns:
            配置文件的SHA256哈希
        """
        hasher = hashlib.sha256()

        # 收集所有配置文件
        config_files = []
        if self._config_path.exists():
            config_files.append(self._config_path)

        # 多文件配置
        base_config = self._config_dir / "base.yaml"
        if base_config.exists():
            config_files.append(base_config)

        agents_dir = self._config_dir / "agents"
        if agents_dir.exists():
            config_files.extend(agents_dir.glob("*.yaml"))

        # 计算哈希
        for config_file in sorted(config_files):
            try:
                with open(config_file, 'rb') as f:
                    hasher.update(f.read())
            except Exception:
                continue

        return hasher.hexdigest()

    def override_from_env(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从环境变量覆盖配置

        Args:
            config_data: 原始配置数据

        Returns:
            覆盖后的配置数据
        """
        def override_recursive(data: Any, prefix: str = "") -> Any:
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    env_key = f"{prefix}{key}".upper() if prefix else key.upper()
                    env_value = os.getenv(env_key)

                    if env_value is not None:
                        # 尝试转换类型
                        if isinstance(value, bool):
                            result[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                        elif isinstance(value, int):
                            try:
                                result[key] = int(env_value)
                            except ValueError:
                                result[key] = value
                        elif isinstance(value, float):
                            try:
                                result[key] = float(env_value)
                            except ValueError:
                                result[key] = value
                        else:
                            result[key] = env_value
                        if self._logger:
                            self._logger.debug(f"Config override from env: {env_key} = {result[key]}")
                    else:
                        result[key] = override_recursive(value, f"{env_key}_")
                return result
            else:
                return data

        return override_recursive(config_data)

    def check_security(self, config_data: Dict[str, Any]) -> List[str]:
        """
        检查配置安全性

        Args:
            config_data: 配置数据

        Returns:
            安全警告列表
        """
        warnings = []

        def check_recursive(data: Any, path: str = ""):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_path = f"{path}.{key}" if path else key

                    # 检查敏感信息
                    sensitive_keys = ['password', 'secret', 'key', 'token', 'api_key', 'apikey']
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        if isinstance(value, str) and len(value) > 10:
                            warnings.append(f"Potential sensitive data found: {full_path}")

                    # 递归检查
                    check_recursive(value, full_path)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    check_recursive(item, f"{path}[{i}]")

        check_recursive(config_data)
        return warnings

    def migrate_config(self, from_version: str, to_version: str) -> bool:
        """
        配置迁移（未来扩展）

        Args:
            from_version: 源版本
            to_version: 目标版本

        Returns:
            迁移是否成功
        """
        # TODO: 实现配置迁移逻辑
        if self._logger:
            self._logger.info(f"Config migration from {from_version} to {to_version} not implemented yet")
        return False


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变更处理器"""

    def __init__(self, config_dir: Path, callback: Callable):
        self.config_dir = config_dir
        self.callback = callback
        self._last_event_time = 0
        self._debounce_seconds = 1.0  # 防抖时间，避免频繁触发

    def on_modified(self, event):
        current_time = time.time()
        if current_time - self._last_event_time > self._debounce_seconds:
            self._last_event_time = current_time
            self.callback(event)

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        self.on_modified(event)
