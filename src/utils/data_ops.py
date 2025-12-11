"""
统一数据操作工具

合并了数据持久化、实体更新、事件映射等核心数据操作功能。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Set
from .file_utils import ensure_dir
from .data_utils import safe_save_data, load_json_data
from ..utils.tool_function import tools


# 重新导出核心数据操作函数
def write_jsonl_file(file_path: Path, data: List[Dict[str, Any]], ensure_ascii: bool = False) -> None:
    """写入JSONL格式文件"""
    from .data_utils import write_jsonl_file as _write_jsonl_file
    _write_jsonl_file(file_path, data, ensure_ascii)


def read_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """读取JSONL格式文件"""
    from .data_utils import read_jsonl_file as _read_jsonl_file
    return _read_jsonl_file(file_path)


def write_json_file(file_path: Path, data: Any, ensure_ascii: bool = False, indent: int = 2) -> None:
    """写入JSON格式文件"""
    from .data_utils import write_json_file as _write_json_file
    _write_json_file(file_path, data, ensure_ascii, indent)


def read_json_file(file_path: Path) -> Any:
    """读取JSON格式文件"""
    from .data_utils import read_json_file as _read_json_file
    return _read_json_file(file_path)


def sanitize_datetime_fields(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """处理datetime字段的序列化"""
    from .data_utils import sanitize_datetime_fields as _sanitize_datetime_fields
    return _sanitize_datetime_fields(data)


def create_temp_file_path(base_dir: Path, prefix: str = "", suffix: str = "", extension: str = "jsonl") -> Path:
    """创建临时文件路径"""
    from .data_utils import create_temp_file_path as _create_temp_file_path
    return _create_temp_file_path(base_dir, prefix, suffix, extension)


def update_entities(entities: List[str], entities_original: List[str], source: str, published_at: Optional[str] = None) -> bool:
    """
    更新实体库，支持增量更新和数据合并

    Args:
        entities: 实体名称列表
        entities_original: 实体原始表述列表（与entities一一对应）
        source: 数据来源
        published_at: 发布时间戳

    Returns:
        是否有数据更新
    """
    from .data_utils import update_entities as _update_entities
    return _update_entities(entities, entities_original, source, published_at)


def update_abstract_map(extracted_list: List[Dict[str, Any]], source: str, published_at: Optional[str] = None) -> bool:
    """
    更新事件映射，支持增量更新和数据合并

    Args:
        extracted_list: 提取的事件列表
        source: 数据来源
        published_at: 发布时间戳

    Returns:
        是否有数据更新
    """
    from .data_utils import update_abstract_map as _update_abstract_map
    return _update_abstract_map(extracted_list, source, published_at)


# 高级数据操作函数
def merge_entity_data(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    合并实体数据到目标字典

    Args:
        target: 目标实体数据
        source: 来源实体数据
    """
    if not source:
        return

    # 合并sources
    target_sources = set(target.get('sources', []))
    source_sources = source.get('sources', [])
    if isinstance(source_sources, list):
        target_sources.update(source_sources)
    target['sources'] = list(target_sources)

    # 合并original_forms
    target_forms = set(target.get('original_forms', []))
    source_forms = source.get('original_forms', [])
    if isinstance(source_forms, list):
        target_forms.update(source_forms)
    target['original_forms'] = list(target_forms)

    # 更新时间戳（取最新时间）
    target_ts = target.get('first_seen')
    source_ts = source.get('first_seen')
    if source_ts and (not target_ts or source_ts > target_ts):
        target['first_seen'] = source_ts


def merge_event_data(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    合并事件数据到目标字典

    Args:
        target: 目标事件数据
        source: 来源事件数据
    """
    if not source:
        return

    # 合并sources
    target_sources = set(target.get('sources', []))
    source_sources = source.get('sources', [])
    if isinstance(source_sources, list):
        target_sources.update(source_sources)
    target['sources'] = list(target_sources)

    # 合并entities
    target_entities = set(target.get('entities', []))
    source_entities = source.get('entities', [])
    if isinstance(source_entities, list):
        target_entities.update(source_entities)
    target['entities'] = list(target_entities)

    # 更新时间戳（取最新时间）
    target_ts = target.get('first_seen')
    source_ts = source.get('first_seen')
    if source_ts and (not target_ts or source_ts > target_ts):
        target['first_seen'] = source_ts

    # 更新事件摘要（保留更详细的）
    target_summary = target.get('event_summary', '')
    source_summary = source.get('event_summary', '')
    if source_summary and len(source_summary) > len(target_summary):
        target['event_summary'] = source_summary


def validate_entity_data(entity: Dict[str, Any]) -> List[str]:
    """
    验证实体数据结构

    Args:
        entity: 实体数据字典

    Returns:
        验证错误列表
    """
    errors = []
    required_fields = ['first_seen', 'sources', 'original_forms']

    for field in required_fields:
        if field not in entity:
            errors.append(f"缺少必需字段: {field}")

    if 'sources' in entity and not isinstance(entity['sources'], list):
        errors.append("sources字段必须是列表")

    if 'original_forms' in entity and not isinstance(entity['original_forms'], list):
        errors.append("original_forms字段必须是列表")

    return errors


def validate_event_data(event: Dict[str, Any]) -> List[str]:
    """
    验证事件数据结构

    Args:
        event: 事件数据字典

    Returns:
        验证错误列表
    """
    errors = []
    required_fields = ['entities', 'event_summary', 'sources', 'first_seen']

    for field in required_fields:
        if field not in event:
            errors.append(f"缺少必需字段: {field}")

    if 'entities' in event and not isinstance(event['entities'], list):
        errors.append("entities字段必须是列表")

    if 'sources' in event and not isinstance(event['sources'], list):
        errors.append("sources字段必须是列表")

    return errors


def cleanup_duplicate_entities(entities_dict: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    清理实体库中的重复实体

    Args:
        entities_dict: 实体字典

    Returns:
        清理统计信息
    """
    stats = {'removed': 0, 'merged': 0}

    # 这里可以实现更复杂的去重逻辑
    # 目前只是占位符，实际实现会比较复杂

    return stats


def cleanup_duplicate_events(events_dict: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    清理事件映射中的重复事件

    Args:
        events_dict: 事件字典

    Returns:
        清理统计信息
    """
    stats = {'removed': 0, 'merged': 0}

    # 这里可以实现更复杂的事件去重逻辑
    # 目前只是占位符，实际实现会比较复杂

    return stats


def backup_data_file(file_path: Path, backup_suffix: str = ".bak") -> Optional[Path]:
    """
    备份数据文件

    Args:
        file_path: 原始文件路径
        backup_suffix: 备份文件后缀

    Returns:
        备份文件路径，如果备份失败则返回None
    """
    if not file_path.exists():
        return None

    backup_path = file_path.with_suffix(f"{file_path.suffix}{backup_suffix}")
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        tools.log(f"备份文件失败 {file_path}: {e}")
        return None


def restore_from_backup(backup_path: Path, target_path: Path) -> bool:
    """
    从备份文件恢复

    Args:
        backup_path: 备份文件路径
        target_path: 目标文件路径

    Returns:
        恢复是否成功
    """
    if not backup_path.exists():
        return False

    try:
        import shutil
        shutil.copy2(backup_path, target_path)
        return True
    except Exception as e:
        tools.log(f"从备份恢复失败 {backup_path} -> {target_path}: {e}")
        return False
