from pathlib import Path

# 获取项目根目录 (假设当前文件在 src/web/config.py)
# src/web/config.py -> src/web -> src -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_NEWS_DIR = DATA_DIR / "raw_news"
LOGS_DIR = DATA_DIR / "logs"

# 关键数据文件
ENTITIES_FILE = DATA_DIR / "entities.json"
EVENTS_FILE = DATA_DIR / "abstract_to_event_map.json"

# UI 配置
PAGE_TITLE_PREFIX = "新闻智能体系统 - "

