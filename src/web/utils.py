import streamlit as st
import json
import pandas as pd
import os
from pathlib import Path
from src.web.config import DATA_DIR
# API配置现在通过ConfigManager获取

# 模板存储目录
TEMPLATES_DIR = DATA_DIR / "config" / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# 临时原始新闻目录（Agent1 处理入口）
RAW_NEWS_TMP_DIR = DATA_DIR / "tmp" / "raw_news"

def _kg_store_backend() -> str:
    v = str(os.getenv("KG_STORE_BACKEND") or "").strip().lower()
    return v or "sqlite"

def load_entities():
    """
    从配置的知识库存储读取实体数据（无缓存）
    """
    backend = _kg_store_backend()
    if backend in {"neo4j"}:
        try:
            from src.adapters.graph_store.neo4j_adapter import get_neo4j_store

            data = get_neo4j_store().export_entities_json()
            if isinstance(data, dict):
                return data
        except Exception as e:
            st.error(f"Error loading entities from Neo4j: {e}")

    try:
        from src.adapters.sqlite.store import get_store

        store = get_store()
        data = store.export_entities_json()
        if isinstance(data, dict):
            return data
    except Exception as e:
        st.error(f"Error loading entities from SQLite: {e}")
        return {}
    return {}

def load_events():
    """
    从配置的知识库存储读取事件数据（无缓存）
    """
    backend = _kg_store_backend()
    if backend in {"neo4j"}:
        try:
            from src.adapters.graph_store.neo4j_adapter import get_neo4j_store

            data = get_neo4j_store().export_abstract_map_json()
            if isinstance(data, dict):
                return data
        except Exception as e:
            st.error(f"Error loading events from Neo4j: {e}")

    try:
        from src.adapters.sqlite.store import get_store

        store = get_store()
        data = store.export_abstract_map_json()
        if isinstance(data, dict):
            return data
    except Exception as e:
        st.error(f"Error loading events from SQLite: {e}")
        return {}
    return {}

def get_raw_news_files():
    """
    直接获取原始新闻文件列表（无缓存）
    仅使用 tmp/raw_news，并按修改时间倒序。
    """
    files = []
    if RAW_NEWS_TMP_DIR.exists():
        files.extend(RAW_NEWS_TMP_DIR.glob("*.jsonl"))
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

def load_raw_news_file(file_path: Path):
    """
    直接加载单个新闻文件内容（无缓存）
    """
    news_items = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    news_items.append(json.loads(line))
                except:
                    continue
    return news_items

def save_pipeline_template(name: str, steps: list):
    """保存流水线模板"""
    file_path = TEMPLATES_DIR / f"{name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False)
    return file_path

def load_pipeline_templates():
    """加载所有可用模板"""
    templates = {}
    if TEMPLATES_DIR.exists():
        for f in TEMPLATES_DIR.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    templates[f.stem] = json.load(fp)
            except:
                continue
    return templates

def get_default_api_sources_df():
    """
    返回默认的 API 源配置 DataFrame (使用默认配置)
    """
    try:
        from src.core import ConfigManager
        config_manager = ConfigManager()
        apis_config = config_manager.get_config_value("llm_apis", "[]", "agent1_config")
        data = json.loads(apis_config) if apis_config else []
    except Exception as e:
        # Fallback if backend config fails
        data = []

    if not data:
        # 扩展到20个以上的新闻源
        data = [
            {"name": "GNews-cn", "language": "zh", "country": "cn", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-us", "language": "en", "country": "us", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-hk", "language": "zh", "country": "hk", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-tw", "language": "zh", "country": "tw", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-sg", "language": "en", "country": "sg", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-gb", "language": "en", "country": "gb", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-au", "language": "en", "country": "au", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-ca", "language": "en", "country": "ca", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-fr", "language": "fr", "country": "fr", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-de", "language": "de", "country": "de", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-jp", "language": "ja", "country": "jp", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-kr", "language": "ko", "country": "kr", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-ru", "language": "ru", "country": "ru", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-ua", "language": "uk", "country": "ua", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-br", "language": "pt", "country": "br", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-ar", "language": "es", "country": "ar", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-mx", "language": "es", "country": "mx", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-es", "language": "es", "country": "es", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-it", "language": "it", "country": "it", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-in", "language": "en", "country": "in", "timeout": 30, "enabled": True, "type": "gnews"}
        ]
        
    return pd.DataFrame(data)
