import streamlit as st
import json
import pandas as pd
from pathlib import Path
from src.web.config import ENTITIES_FILE, EVENTS_FILE, DATA_DIR
from src.data.api_client import get_apis_config

# 模板存储目录
TEMPLATES_DIR = DATA_DIR / "config" / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# 临时原始新闻目录（Agent1 处理入口）
RAW_NEWS_TMP_DIR = DATA_DIR / "tmp" / "raw_news"

@st.cache_data(ttl=60)
def load_entities():
    """
    加载实体数据，缓存 60 秒
    """
    if ENTITIES_FILE.exists():
        try:
            with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading entities: {e}")
            return {}
    return {}

@st.cache_data(ttl=60)
def load_events():
    """
    加载事件数据，缓存 60 秒
    """
    if EVENTS_FILE.exists():
        try:
            with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading events: {e}")
            return {}
    return {}

@st.cache_data(ttl=300)
def get_raw_news_files():
    """
    获取原始新闻文件列表，缓存 300 秒 (5分钟)
    仅使用 tmp/raw_news，并按修改时间倒序。
    """
    files = []
    if RAW_NEWS_TMP_DIR.exists():
        files.extend(RAW_NEWS_TMP_DIR.glob("*.jsonl"))
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

@st.cache_data(ttl=60)
def load_raw_news_file(file_path: Path):
    """
    加载单个新闻文件内容，缓存 60 秒
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
    返回默认的 API 源配置 DataFrame (直接从后端配置加载)
    """
    try:
        data = get_apis_config()
    except Exception as e:
        # Fallback if backend config fails
        data = []

    if not data:
         data = [
            {"name": "GNews-cn", "language": "zh", "country": "cn", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-us", "language": "en", "country": "us", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-fr", "language": "fr", "country": "fr", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-gb", "language": "en", "country": "gb", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-hk", "language": "zh", "country": "hk", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-ru", "language": "ru", "country": "ru", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-ua", "language": "uk", "country": "ua", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-tw", "language": "zh", "country": "tw", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-sg", "language": "en", "country": "sg", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-jp", "language": "ja", "country": "jp", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-br", "language": "pt", "country": "br", "timeout": 30, "enabled": True, "type": "gnews"},
            {"name": "GNews-ar", "language": "es", "country": "ar", "timeout": 30, "enabled": True, "type": "gnews"}
        ]
        
    return pd.DataFrame(data)
