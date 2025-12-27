"""
端口层（Ports Layer）。

定义外部依赖的抽象接口，由 Adapters 层实现。

包含：
- llm_client: LLM 客户端接口（调用/限速/熔断）
- store: 存储仓储接口（实体/事件/关系/审查）
- snapshot: 快照/运行记录/日志接口
- kg_read_store: 图谱只读仓储（兼容现有）
"""

# LLM Client
from .llm_client import (
    LLMProviderType,
    LLMCallConfig,
    LLMResponse,
    RateLimiter,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    LLMClient,
    LLMClientPool,
)

# Store
from .store import (
    KGReadStore,
    EntityStore,
    EventStore,
    RelationStore,
    ParticipantStore,
    EventEdgeStore,
    MentionStore,
    ReviewStore,
    UnifiedStore,
)

# Snapshot / Run / Log
from .snapshot import (
    GraphSnapshotType,
    SnapshotMeta,
    SnapshotNode,
    SnapshotEdge,
    Snapshot,
    SnapshotParams,
    SnapshotWriter,
    SnapshotReader,
    RunStatus,
    RunRecord,
    RunStore,
    LogLevel,
    LogEntry,
    LogStore,
)

# Extraction (NewsSource, EntityExtractor, EventExtractor, Deduplicator)
from .extraction import (
    NewsSourceType,
    NewsItem,
    FetchConfig,
    FetchResult,
    NewsSource,
    NewsSourcePool,
    ExtractedEntity,
    EntityExtractionResult,
    EntityExtractor,
    ExtractedEvent,
    EventExtractionResult,
    EventExtractor,
    DeduplicationResult,
    Deduplicator,
    TextDeduplicator,
)

__all__ = [
    # LLM
    "LLMProviderType",
    "LLMCallConfig",
    "LLMResponse",
    "RateLimiter",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "LLMClient",
    "LLMClientPool",
    # Store
    "KGReadStore",
    "EntityStore",
    "EventStore",
    "RelationStore",
    "ParticipantStore",
    "EventEdgeStore",
    "MentionStore",
    "ReviewStore",
    "UnifiedStore",
    # Snapshot
    "GraphSnapshotType",
    "SnapshotMeta",
    "SnapshotNode",
    "SnapshotEdge",
    "Snapshot",
    "SnapshotParams",
    "SnapshotWriter",
    "SnapshotReader",
    "RunStatus",
    "RunRecord",
    "RunStore",
    "LogLevel",
    "LogEntry",
    "LogStore",
    # Extraction
    "NewsSourceType",
    "NewsItem",
    "FetchConfig",
    "FetchResult",
    "NewsSource",
    "NewsSourcePool",
    "ExtractedEntity",
    "EntityExtractionResult",
    "EntityExtractor",
    "ExtractedEvent",
    "EventExtractionResult",
    "EventExtractor",
    "DeduplicationResult",
    "Deduplicator",
    "TextDeduplicator",
]



