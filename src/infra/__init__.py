"""
基础设施层（Infrastructure Layer）。

提供通用基础能力：
- Clock: 统一时间管理
- IdFactory: ID 生成与规范化
- Migration: Schema 版本管理
- Retry: 重试策略
- Rate Limiting: 限速
- Circuit Breaker: 熔断
"""

from .common import (
    # Clock
    Clock,
    SystemClock,
    MockClock,
    get_clock,
    set_clock,
    utc_now,
    utc_now_iso,
    parse_iso,
    # IdFactory
    IdFactory,
    # Migration
    MigrationRecord,
    MigrationManager,
    # Retry
    RetryStrategy,
    RetryConfig,
    retry_with_backoff,
    retry_async,
    # Rate Limiter
    TokenBucketRateLimiter,
    # Circuit Breaker
    SimpleCircuitBreaker,
)

# Logging
from .logging import (
    LoggerManager,
    get_logger,
    set_log_level,
)

# Config
from .config import (
    ConfigManager,
    get_config_manager,
    set_config_manager,
    get_config_value,
    ConfigFileHandler,
)

# Key Manager
from .key_manager import (
    KeyManager,
    get_key_manager,
    store_api_key,
    get_api_key,
)

# Cache
from .cache import (
    MemoryCache,
    FileCache,
    SmartCache,
    get_global_cache,
    set_global_cache,
    cached_operation,
)

# Exceptions
from .exceptions import (
    NewsAgentException,
    ConfigError,
    ValidationError,
    NetworkError,
    APIError,
    ProcessingError,
    FileOperationError,
    ConcurrencyError,
    LLMError,
    CircuitBreakerOpenError,
    RateLimitExceededError,
    StoreError,
    MigrationError,
    handle_errors,
    handle_async_errors,
    ErrorHandler,
)

# Serialization
from .serialization import (
    Serializer,
    extract_json_from_llm_response,
    safe_json_loads,
    format_json_for_llm,
)

# File Utils
from .file_utils import (
    ensure_dir,
    ensure_dirs,
    safe_unlink,
    safe_unlink_multiple,
    generate_timestamp,
    get_file_size_mb,
    cleanup_temp_files,
    read_json_sync,
    write_json_sync,
    AsyncFileOperations,
    AsyncFileLock,
)

__all__ = [
    # Clock
    "Clock",
    "SystemClock",
    "MockClock",
    "get_clock",
    "set_clock",
    "utc_now",
    "utc_now_iso",
    "parse_iso",
    # IdFactory
    "IdFactory",
    # Migration
    "MigrationRecord",
    "MigrationManager",
    # Retry
    "RetryStrategy",
    "RetryConfig",
    "retry_with_backoff",
    "retry_async",
    # Rate Limiter
    "TokenBucketRateLimiter",
    # Circuit Breaker
    "SimpleCircuitBreaker",
    # Logging
    "LoggerManager",
    "get_logger",
    "set_log_level",
    # Config
    "ConfigManager",
    "get_config_manager",
    "set_config_manager",
    "get_config_value",
    # Cache
    "MemoryCache",
    "FileCache",
    "SmartCache",
    "get_global_cache",
    "set_global_cache",
    "cached_operation",
    # Exceptions
    "NewsAgentException",
    "ConfigError",
    "ValidationError",
    "NetworkError",
    "APIError",
    "ProcessingError",
    "FileOperationError",
    "ConcurrencyError",
    "LLMError",
    "CircuitBreakerOpenError",
    "RateLimitExceededError",
    "StoreError",
    "MigrationError",
    "handle_errors",
    "handle_async_errors",
    "ErrorHandler",
    # Serialization
    "Serializer",
    "extract_json_from_llm_response",
    "safe_json_loads",
    "format_json_for_llm",
    # File Utils
    "ensure_dir",
    "ensure_dirs",
    "safe_unlink",
    "safe_unlink_multiple",
    "generate_timestamp",
    "get_file_size_mb",
    "cleanup_temp_files",
    "read_json_sync",
    "write_json_sync",
    "AsyncFileOperations",
    "AsyncFileLock",
]



