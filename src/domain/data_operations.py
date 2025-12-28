"""
领域层 - 数据持久化操作

统一处理数据序列化和文件写入操作。
从 utils/data_utils.py 迁移。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import os

from ..infra.file_utils import ensure_dir
from ..infra.paths import tools as Tools

# 实例化tools
_tools = Tools()

def _kg_store_backend() -> str:
    v = str(os.getenv("KG_STORE_BACKEND") or "").strip().lower()
    return v or "sqlite"


def write_jsonl_file(file_path: Path, data: List[Dict[str, Any]], ensure_ascii: bool = False) -> None:
    """
    写入JSONL格式文件

    Args:
        file_path: 输出文件路径
        data: 要写入的数据列表
        ensure_ascii: 是否确保ASCII编码
    """
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")


def append_jsonl_file(file_path: Path, data: List[Dict[str, Any]], ensure_ascii: bool = False) -> None:
    """
    追加到JSONL格式文件

    Args:
        file_path: 输出文件路径
        data: 要追加的数据列表
        ensure_ascii: 是否确保ASCII编码
    """
    ensure_dir(file_path.parent)

    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")


def read_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    读取JSONL格式文件

    Args:
        file_path: 文件路径

    Returns:
        解析后的数据列表
    """
    if not file_path.exists():
        return []

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 跳过无效行
    return data


def sanitize_datetime_fields(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    处理datetime字段的序列化，将datetime对象转换为ISO格式字符串

    Args:
        data: 输入数据（字典或字典列表）

    Returns:
        处理后的数据
    """
    if isinstance(data, list):
        return [sanitize_datetime_fields(item) for item in data]
    elif isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            else:
                result[k] = v
        return result
    else:
        return data


def write_json_file(file_path: Path, data: Any, ensure_ascii: bool = False, indent: int = 2) -> None:
    """
    写入JSON格式文件

    Args:
        file_path: 输出文件路径
        data: 要写入的数据
        ensure_ascii: 是否确保ASCII编码
        indent: 缩进空格数
    """
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def read_json_file(file_path: Path) -> Any:
    """
    读取JSON格式文件

    Args:
        file_path: 文件路径

    Returns:
        解析后的数据
    """
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_temp_file_path(base_dir: Path, prefix: str = "", suffix: str = "", extension: str = "jsonl") -> Path:
    """
    创建临时文件路径

    Args:
        base_dir: 基础目录
        prefix: 文件名前缀
        suffix: 文件名后缀
        extension: 文件扩展名

    Returns:
        临时文件路径
    """
    from ..infra.file_utils import generate_timestamp

    timestamp = generate_timestamp()
    filename_parts = []

    if prefix:
        filename_parts.append(prefix)
    filename_parts.append(timestamp)
    if suffix:
        filename_parts.append(suffix)

    filename = "_".join(filename_parts) + f".{extension}"
    return base_dir / filename


def safe_save_data(data: Dict[str, Any], main_file: Path, tmp_file: Optional[Path] = None, indent: int = 2) -> bool:
    """
    安全的JSON数据保存，支持缓存同步

    Args:
        data: 要保存的数据
        main_file: 主文件路径
        tmp_file: 缓存文件路径（可选）
        indent: JSON缩进

    Returns:
        保存是否成功
    """
    try:
        # 确保目录存在
        ensure_dir(main_file.parent)

        # 直接写入主文件
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        # 同步写入缓存文件
        if tmp_file:
            try:
                ensure_dir(tmp_file.parent)
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=indent)
            except Exception:
                # 缓存文件写入失败不影响主流程
                pass

        return True
    except Exception as e:
        _tools.log(f"数据保存失败 {main_file}: {e}")
        return False


def load_entities_from_sqlite() -> Dict[str, Any]:
    """
    从SQLite加载实体数据

    Returns:
        实体数据字典
    """
    try:
        from src.adapters.sqlite.store import get_store
        store = get_store()
        return store.export_entities_json()
    except Exception as e:
        _tools.log(f"从SQLite加载实体数据失败: {e}")
        return {}

def load_events_from_sqlite() -> Dict[str, Any]:
    """
    从SQLite加载事件数据

    Returns:
        事件数据字典
    """
    try:
        from src.adapters.sqlite.store import get_store
        store = get_store()
        return store.export_abstract_map_json()
    except Exception as e:
        _tools.log(f"从SQLite加载事件数据失败: {e}")
        return {}


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
    # 参数验证
    if len(entities) != len(entities_original):
        _tools.log("实体列表和原始表述列表长度不匹配")
        return False

    backend = _kg_store_backend()
    wrote_any = False

    if backend in {"neo4j", "dual"}:
        try:
            from src.adapters.graph_store.neo4j_adapter import get_neo4j_store

            get_neo4j_store().upsert_entities(entities, entities_original, source=source, reported_at=published_at)
            wrote_any = True
        except Exception as e:
            _tools.log(f"Neo4j实体写入失败: {e}")

    if backend in {"sqlite", "dual"} or not wrote_any:
        try:
            from src.adapters.sqlite.store import get_store

            store = get_store()
            store.upsert_entities(entities, entities_original, source=source, reported_at=published_at)
            wrote_any = True
        except Exception as e:
            _tools.log(f"SQLite实体写入失败: {e}")

    return wrote_any

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
    backend = _kg_store_backend()
    wrote_any = False

    if backend in {"neo4j", "dual"}:
        try:
            from src.adapters.graph_store.neo4j_adapter import get_neo4j_store

            get_neo4j_store().upsert_events(extracted_list, source=source, reported_at=published_at)
            wrote_any = True
        except Exception as e:
            _tools.log(f"Neo4j事件写入失败: {e}")

    if backend in {"sqlite", "dual"} or not wrote_any:
        try:
            from src.adapters.sqlite.store import get_store

            store = get_store()
            store.upsert_events(extracted_list, source=source, reported_at=published_at)
            wrote_any = True
        except Exception as e:
            _tools.log(f"SQLite事件写入失败: {e}")

    return wrote_any

# =============================================================================
# 高级数据操作函数
# =============================================================================

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
    errors: List[str] = []
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
    errors: List[str] = []
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
    stats: Dict[str, int] = {'removed': 0, 'merged': 0}
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
    stats: Dict[str, int] = {'removed': 0, 'merged': 0}
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
        _tools.log(f"备份文件失败 {file_path}: {e}")
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
        _tools.log(f"从备份恢复失败 {backup_path} -> {target_path}: {e}")
        return False
