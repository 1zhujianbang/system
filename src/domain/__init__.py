"""
领域层（Domain Layer）。

包含：
- models: 核心领域模型（EntityMention/EventMention/EntityCanonical/EventCanonical/RelationTriple/EventEdge等）
- rules: 业务规则接口（CandidateGenerator/Adjudicator/Applier/MentionResolver）

原则：
- 纯业务规则，不做 IO
- 所有关系/边强制带 time 字段
- 使用 dataclass 定义不可变值对象
"""

from .models import (
    # Value Objects
    SourceRef,
    TimePrecision,
    # Mention 模型
    EntityMention,
    EventMention,
    # Canonical 模型
    EntityCanonical,
    EventCanonical,
    # 关系模型
    RelationTriple,
    EventEdge,
    EventEdgeType,
    Participant,
    # Review 模型
    ReviewTask,
    ReviewTaskStatus,
    ReviewTaskType,
    MergeDecision,
    MergeDecisionType,
)

from .data_pipeline import (
    DataNormalizer,
    DataPipeline,
    StandardEventPipeline,
    BatchDataProcessor,
)

from .data_ops import (
    merge_entity_data,
    merge_event_data,
    validate_entity_data,
    validate_event_data,
    cleanup_duplicate_entities,
    cleanup_duplicate_events,
    backup_data_file,
    restore_from_backup,
)

from .rules import (
    # 候选生成
    CandidateReason,
    EntityMergeCandidatePair,
    EventMergeCandidatePair,
    CandidateGenerator,
    # 裁决
    MergeVerdict,
    EntityMergeVerdict,
    EventMergeVerdict,
    Adjudicator,
    # 应用
    MergeAction,
    EdgeCreationAction,
    Applier,
    # Resolution
    ResolutionResult,
    MentionResolver,
    # 纯函数
    normalize_entity_name,
    compute_name_similarity,
    select_canonical_name,
    select_canonical_event,
    merge_entity_sources,
    merge_original_forms,
    validate_time_constraint,
)

__all__ = [
    # Models
    "SourceRef",
    "TimePrecision",
    "EntityMention",
    "EventMention",
    "EntityCanonical",
    "EventCanonical",
    "RelationTriple",
    "EventEdge",
    "EventEdgeType",
    "Participant",
    "ReviewTask",
    "ReviewTaskStatus",
    "ReviewTaskType",
    "MergeDecision",
    "MergeDecisionType",
    # Rules
    "CandidateReason",
    "EntityMergeCandidatePair",
    "EventMergeCandidatePair",
    "CandidateGenerator",
    "MergeVerdict",
    "EntityMergeVerdict",
    "EventMergeVerdict",
    "Adjudicator",
    "MergeAction",
    "EdgeCreationAction",
    "Applier",
    "ResolutionResult",
    "MentionResolver",
    # Pure functions
    "normalize_entity_name",
    "compute_name_similarity",
    "select_canonical_name",
    "select_canonical_event",
    "merge_entity_sources",
    "merge_original_forms",
    "validate_time_constraint",
    # Data Pipeline
    "DataNormalizer",
    "DataPipeline",
    "StandardEventPipeline",
    "BatchDataProcessor",
    # Data Operations
    "merge_entity_data",
    "merge_event_data",
    "validate_entity_data",
    "validate_event_data",
    "cleanup_duplicate_entities",
    "cleanup_duplicate_events",
    "backup_data_file",
    "restore_from_backup",
]




