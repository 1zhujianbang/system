"""
数据持久化工具函数

统一处理数据序列化和文件写入操作。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from ..utils.file_utils import ensure_dir
from ..utils.tool_function import tools as Tools

# 实例化tools
_tools = Tools()


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
    from ..utils.file_utils import generate_timestamp

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


def load_json_data(file_path: Path) -> Dict[str, Any]:
    """
    安全的JSON数据加载

    Args:
        file_path: 文件路径

    Returns:
        加载的数据字典
    """
    if not file_path.exists():
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        _tools.log(f"数据加载失败 {file_path}: {e}")
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

    # 获取时间戳
    now = datetime.now(timezone.utc).isoformat()
    base_ts = published_at or now

    # 加载现有数据
    existing = load_json_data(_tools.ENTITIES_FILE)
    needs_update = False

    # 处理每个实体
    for ent, ent_original in zip(entities, entities_original):
        if not ent or not ent_original:
            continue

        if ent not in existing:
            # 新实体
            existing[ent] = {
                "first_seen": base_ts,
                "sources": [source],
                "original_forms": [ent_original]
            }
            needs_update = True
        else:
            # 现有实体 - 更新信息
            entity_data = existing[ent]

            # 更新时间戳（取最早时间）
            old_ts = entity_data.get("first_seen")
            if old_ts and base_ts and base_ts < old_ts:
                entity_data["first_seen"] = base_ts
                needs_update = True

            # 更新来源
            if source not in entity_data.get("sources", []):
                entity_data.setdefault("sources", []).append(source)
                needs_update = True

            # 更新原始表述
            original_forms = entity_data.setdefault("original_forms", [])
            if ent_original not in original_forms:
                original_forms.append(ent_original)
                needs_update = True

    # 保存数据
    if needs_update:
        success = safe_save_data(existing, _tools.ENTITIES_FILE, _tools.ENTITIES_TMP_FILE, indent=2)
        if success:
            _tools.log(f"实体库已更新，共 {len(existing)} 个实体")
        return success

    return False


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
    # 获取时间戳
    now = datetime.now(timezone.utc).isoformat()
    base_ts = published_at or now

    # 加载现有数据
    abstract_map = load_json_data(_tools.ABSTRACT_MAP_FILE)
    needs_update = False

    for item in extracted_list:
        if not isinstance(item, dict):
            continue

        key = item.get("abstract", "").strip()
        if not key:
            continue

        entities = item.get("entities", [])
        event_summary = item.get("event_summary", "")

        if key not in abstract_map:
            # 新事件
            abstract_map[key] = {
                "entities": entities,
                "event_summary": event_summary,
                "sources": [source],
                "first_seen": base_ts
            }
            needs_update = True
        else:
            # 现有事件 - 合并信息
            event_data = abstract_map[key]

            # 更新时间戳（取最早时间）
            old_ts = event_data.get("first_seen")
            if old_ts and base_ts and base_ts < old_ts:
                event_data["first_seen"] = base_ts
                needs_update = True

            # 更新来源
            if source not in event_data.get("sources", []):
                event_data.setdefault("sources", []).append(source)
                needs_update = True

            # 合并实体（去重）
            existing_entities = event_data.get("entities", [])
            for ent in entities:
                if ent not in existing_entities:
                    existing_entities.append(ent)
                    needs_update = True

            # 更新事件摘要（如果有新的且不同的摘要）
            existing_summary = event_data.get("event_summary", "")
            if event_summary and event_summary != existing_summary:
                if existing_summary:
                    # 合并摘要
                    event_data["event_summary"] = existing_summary + "; " + event_summary
                else:
                    event_data["event_summary"] = event_summary
                needs_update = True

    # 保存数据
    if needs_update:
        success = safe_save_data(abstract_map, _tools.ABSTRACT_MAP_FILE, _tools.ABSTRACT_TMP_FILE, indent=2)
        if success:
            _tools.log(f"事件映射已更新，共 {len(abstract_map)} 个事件")
        return success

    return False
