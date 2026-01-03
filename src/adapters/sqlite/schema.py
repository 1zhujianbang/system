"""
SQLite Schema 定义与迁移。

包含：
- 表结构定义（DDL）
- Schema 版本管理
- 迁移脚本
"""
from __future__ import annotations

from typing import List

# 当前 Schema 版本
SCHEMA_VERSION = "5"

# =============================================================================
# 核心表结构（V3）
# =============================================================================

CORE_TABLES_DDL = """
-- Meta 表：存储全局元数据
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 实体表
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    original_forms_json TEXT NOT NULL
);

-- 事件表
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    abstract TEXT NOT NULL UNIQUE,
    event_summary TEXT NOT NULL,
    event_types_json TEXT NOT NULL,
    event_start_time TEXT NOT NULL,
    event_start_time_text TEXT NOT NULL,
    event_start_time_precision TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    sources_json TEXT NOT NULL
);

-- 参与关系表（实体参与事件，强制带 time）
CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    roles_json TEXT NOT NULL,
    time TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    UNIQUE(event_id, entity_id),
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

-- 关系表（实体-实体三元组，强制带 time）
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    subject_entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL,
    relation_kind TEXT NOT NULL DEFAULT '',
    time TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    UNIQUE(event_id, subject_entity_id, predicate, object_entity_id),
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(subject_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY(object_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_participants_time ON participants(time);
CREATE INDEX IF NOT EXISTS idx_relations_time ON relations(time);
CREATE INDEX IF NOT EXISTS idx_events_first_seen ON events(first_seen);
"""

# =============================================================================
# Mention-first 表结构（审计层）
# =============================================================================

MENTION_TABLES_DDL = """
-- 实体提及表（先落 mention，再 resolve 到 canonical）
CREATE TABLE IF NOT EXISTS entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mention_id TEXT NOT NULL UNIQUE,
    name_text TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    source_json TEXT NOT NULL,
    resolved_entity_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_reported_at ON entity_mentions(reported_at);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_resolved ON entity_mentions(resolved_entity_id);

-- 事件提及表
CREATE TABLE IF NOT EXISTS event_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mention_id TEXT NOT NULL UNIQUE,
    abstract_text TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    source_json TEXT NOT NULL,
    resolved_event_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_mentions_reported_at ON event_mentions(reported_at);
CREATE INDEX IF NOT EXISTS idx_event_mentions_resolved ON event_mentions(resolved_event_id);
"""

# =============================================================================
# Review 表结构（审查任务与决策）
# =============================================================================

REVIEW_TABLES_DDL = """
-- 审查任务队列
CREATE TABLE IF NOT EXISTS review_tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    input_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 50,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT NOT NULL DEFAULT '',
    output_json TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_review_tasks_status ON review_tasks(status);
CREATE INDEX IF NOT EXISTS idx_review_tasks_priority ON review_tasks(priority);

-- 合并决策记录
CREATE TABLE IF NOT EXISTS merge_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    input_hash TEXT NOT NULL UNIQUE,
    output_json TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_merge_decisions_type ON merge_decisions(type);
"""

# =============================================================================
# Alias & Redirect 表结构（实体/事件收敛）
# =============================================================================

ALIAS_TABLES_DDL = """
-- 实体别名
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    decision_input_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);

-- 实体重定向（合并记录）
CREATE TABLE IF NOT EXISTS entity_redirects (
    from_entity_id TEXT PRIMARY KEY,
    to_entity_id TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    decision_input_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(to_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

-- 事件别名
CREATE TABLE IF NOT EXISTS event_aliases (
    abstract TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    decision_input_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_event_aliases_event ON event_aliases(event_id);

-- 事件重定向
CREATE TABLE IF NOT EXISTS event_redirects (
    from_event_id TEXT PRIMARY KEY,
    to_event_id TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    decision_input_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(from_event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(to_event_id) REFERENCES events(event_id) ON DELETE CASCADE
);
"""

# =============================================================================
# 事件演化边表结构
# =============================================================================

EVENT_EDGE_TABLES_DDL = """
-- 事件-事件演化边（强制带 time）
CREATE TABLE IF NOT EXISTS event_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_event_id TEXT NOT NULL,
    to_event_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    time TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    decision_input_hash TEXT NOT NULL DEFAULT '',
    UNIQUE(from_event_id, to_event_id, edge_type),
    FOREIGN KEY(from_event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(to_event_id) REFERENCES events(event_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_event_edges_time ON event_edges(time);
"""

# =============================================================================
# Canonical 主名称映射表结构（entity/event 显示名与可检索主键）
# =============================================================================

MAIN_NAME_TABLES_DDL = """
-- 实体主名称（展示/检索用，不影响 entity_id 的稳定性）
CREATE TABLE IF NOT EXISTS entity_main_names (
    entity_id TEXT PRIMARY KEY,
    main_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_entity_main_names_main_name ON entity_main_names(main_name);

-- 事件主摘要（展示/检索用，不影响 event_id 的稳定性）
CREATE TABLE IF NOT EXISTS event_main_abstracts (
    event_id TEXT PRIMARY KEY,
    main_abstract TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_event_main_abstracts_main_abstract ON event_main_abstracts(main_abstract);
"""

# =============================================================================
# Schema 迁移表
# =============================================================================

MIGRATION_TABLE_DDL = """
-- Schema 迁移记录
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 1
);
"""

# =============================================================================
# 完整初始化 DDL
# =============================================================================


def get_full_schema_ddl() -> str:
    """获取完整的 Schema DDL"""
    return "\n".join([
        CORE_TABLES_DDL,
        MENTION_TABLES_DDL,
        REVIEW_TABLES_DDL,
        ALIAS_TABLES_DDL,
        EVENT_EDGE_TABLES_DDL,
        MAIN_NAME_TABLES_DDL,
        MIGRATION_TABLE_DDL,
    ])


# =============================================================================
# 迁移定义
# =============================================================================


class Migration:
    """迁移定义"""
    def __init__(self, version: str, description: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql


# 迁移列表（按版本顺序）
MIGRATIONS: List[Migration] = [
    Migration(
        version="1",
        description="Initial schema with entities, events, participants, relations",
        up_sql=CORE_TABLES_DDL,
    ),
    Migration(
        version="2",
        description="Add mention-first tables and review infrastructure",
        up_sql=MENTION_TABLES_DDL + REVIEW_TABLES_DDL + ALIAS_TABLES_DDL + EVENT_EDGE_TABLES_DDL,
    ),
    Migration(
        version="3",
        description="Add schema_migrations table for version tracking",
        up_sql=MIGRATION_TABLE_DDL,
    ),
    Migration(
        version="4",
        description="Add relation_kind to relations",
        up_sql="ALTER TABLE relations ADD COLUMN relation_kind TEXT NOT NULL DEFAULT '';",
    ),
    Migration(
        version="5",
        description="Add entity/event main name mapping tables",
        up_sql=MAIN_NAME_TABLES_DDL,
    ),
]


def get_migrations_since(current_version: str) -> List[Migration]:
    """获取指定版本之后的所有迁移"""
    if not current_version:
        return MIGRATIONS
    try:
        current_num = int(current_version)
    except ValueError:
        current_num = 0
    return [m for m in MIGRATIONS if int(m.version) > current_num]
