# src/agents/agent1.py
"""
智能体1：流式新闻去重 + LLM驱动的真实世界实体与事件提取器

核心原则：
- 实体 = 能签署合同、被起诉、发布公告、拥有银行账户的主体
  （自然人、公司、政府机构、国家、地区、国际组织）
- 排除：代币名称、技术术语、抽象概念、情绪词、泛称
- 提取即自动写入 entities.json，无需人工审核
- 每个事件生成唯一摘要，并关联实体与事件描述
"""

import os
import sys
import json
import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv

# ======================
# 路径与配置
# ======================

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"
RAW_NEWS_DIR = DATA_DIR / "raw_news"
DEDUPED_NEWS_DIR = DATA_DIR / "deduped_news"
LOG_FILE = DATA_DIR / "logs" / "agent1.log"

# 确保目录存在
for d in [DATA_DIR, RAW_NEWS_DIR, DEDUPED_NEWS_DIR, DATA_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

# 数据文件
ENTITIES_FILE = DATA_DIR / "entities.json"
ABSTRACT_MAP_FILE = DATA_DIR / "abstract_to_event_map.json"
PROCESSED_IDS_FILE = DATA_DIR / "processed_ids.txt"
STOP_WORDS_FILE = DATA_DIR / "stop_words.txt"

# 加载环境变量
load_dotenv(CONFIG_DIR / ".env.local")
LLM_MODEL = os.getenv("AGENT1_LLM_MODEL", "deepseek-chat")
DEDUPE_THRESHOLD = int(os.getenv("AGENT1_DEDUPE_THRESHOLD", "3"))

# ======================
# 工具函数
# ======================

def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_stop_words() -> Set[str]:
    stop_words = set()
    if STOP_WORDS_FILE.exists():
        with open(STOP_WORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith("#"):
                    stop_words.add(word)
    return stop_words

STOP_WORDS = load_stop_words()

def is_valid_entity(entity: str) -> bool:
    word = entity.strip()
    if not word or len(word) < 2:
        return False
    if word in STOP_WORDS:
        return False
    if word.isdigit() or re.match(r'^[0-9+\-\.%$€¥]+$', word):
        return False
    if not any(c.isalnum() for c in word):
        return False
    # 排除常见无效模式
    if re.search(
        r'(上涨|下跌|暴涨|暴跌|利好|利空|市场|投资者|用户|社区|'
        r'协议|链|代币|币种|项目|平台|技术|系统|机制|概念|'
        r'行情|趋势|信号|策略|模型|算法|框架|生态)$',
        word
    ):
        return False
    return True

def simhash(text: str, bits=64) -> int:
    text = re.sub(r'\s+', ' ', text.lower())
    tokens = text.split()
    v = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            bit = (h >> i) & 1
            if bit:
                v[i] += 1
            else:
                v[i] -= 1
    hash_val = 0
    for i in range(bits):
        if v[i] > 0:
            hash_val |= (1 << i)
    return hash_val

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

# ======================
# 新闻去重器
# ======================

class NewsDeduplicator:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.seen_hashes: Set[int] = set()

    def is_duplicate(self, text: str) -> bool:
        h = simhash(text)
        for seen_h in self.seen_hashes:
            if hamming_distance(h, seen_h) <= self.threshold:
                return True
        self.seen_hashes.add(h)
        return False

    def dedupe_file(self, input_path: Path, output_path: Path):
        log(f"🔍 去重中: {input_path.name}")
        seen_ids = set()
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    seen_ids.add(item["id"])
        
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "a", encoding="utf-8") as fout:
            for line in fin:
                try:
                    news = json.loads(line)
                    if news["id"] in seen_ids:
                        continue
                    raw_text = (news.get("title", "") + " " + news.get("content", "")).strip()
                    if not raw_text:
                        continue
                    if self.is_duplicate(raw_text):
                        continue
                    fout.write(line)
                    seen_ids.add(news["id"])
                except Exception as e:
                    log(f"⚠️ 跳过无效行: {e}")

# ======================
# LLM 结构化提取器（含精准提示词）
# ======================

def llm_extract_events(title: str, content: str) -> List[Dict]:
    try:
        from openai import OpenAI
    except ImportError:
        log("❌ openai 未安装，跳过 LLM 提取")
        return []

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_URL")
    if not api_key:
        log("❌ API_KEY 未设置")
        return []

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    prompt = f"""你是一名专业的金融与法律信息结构化专家。请从以下新闻中提取所有**真实存在的、具有法律人格或行政职能的实体**。

【实体定义】
✅ 必须满足以下任一条件：
- 是自然人（如 Elon Musk、Cathie Wood）
- 是注册公司（如 Binance、Coinbase、Tesla）
- 是政府机构或部门（如 美国证券交易委员会、中国人民银行、欧盟委员会）
- 是主权国家或明确行政区（如 美国、新加坡、加利福尼亚州、香港特别行政区）
- 是国际组织（如 国际货币基金组织、联合国）

❌ 以下内容**不得**视为实体：
- 抽象概念（如 “去中心化”、“流动性”、“市场情绪”）
- 技术术语（如 “智能合约”、“零知识证明”、“PoS”）
- 代币/资产名称（如 “BTC”、“以太坊”、“Solana”）——除非指代其基金会或开发公司（如 “以太坊基金会”）
- 泛称（如 “投资者”、“监管机构”、“某交易所”）
- 情绪/行情描述（如 “牛市”、“暴跌”、“利好”）

【任务要求】
1. 判断新闻是否包含一个或多个独立事件。
2. 对每个事件，输出：
   - 一个简洁、客观、无情绪的中文摘要（作为事件唯一标识）
   - 所有符合上述定义的实体（全称优先，避免缩写）
   - 该事件的本质描述（一句话说明“谁对谁做了什么”）

【输出格式】
严格返回 JSON，不要任何额外文本：
{{
  "events": [
    {{
      "abstract": "美国证券交易委员会推迟对比特币ETF的最终决定",
      "entities": ["美国证券交易委员会", "VanEck"],
      "event_summary": "监管机构延长了对某资产管理公司比特币ETF申请的审查期"
    }}
  ]
}}

【新闻】
标题：{title}
正文：{content}"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            timeout=600,
            stream=False
        )
        content = response.choices[0].message.content.strip()

        # 清理 Markdown 包裹
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].split("```")[0]
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```")[0]

        data = json.loads(content)
        events = data.get("events", [])
        result = []
        for item in events:
            abstract = item.get("abstract", "").strip()
            entities = [e for e in item.get("entities", []) if is_valid_entity(e)]
            summary = item.get("event_summary", "").strip()
            if abstract and entities and summary:
                result.append({
                    "abstract": abstract,
                    "entities": entities,
                    "event_summary": summary
                })
        return result
    except Exception as e:
        log(f"❌ LLM 提取失败: {e}")
        return []

# ======================
# 自动更新知识库
# ======================

def update_entities(entities: List[str], source: str):
    """自动写入主实体库"""
    now = datetime.now(timezone.utc).isoformat()
    existing = {}
    if ENTITIES_FILE.exists():
        with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    
    for ent in entities:
        if ent not in existing:
            existing[ent] = {
                "first_seen": now,
                "sources": [source]
            }
        else:
            if source not in existing[ent]["sources"]:
                existing[ent]["sources"].append(source)
    
    with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

def update_abstract_map(extracted_list: List[Dict], source: str):
    abstract_map = {}
    if ABSTRACT_MAP_FILE.exists():
        with open(ABSTRACT_MAP_FILE, "r", encoding="utf-8") as f:
            abstract_map = json.load(f)
    
    now = datetime.now(timezone.utc).isoformat()
    for item in extracted_list:
        key = item["abstract"]
        if key not in abstract_map:
            abstract_map[key] = {
                "entities": item["entities"],
                "event_summary": item["event_summary"],
                "sources": [source],
                "first_seen": now
            }
        else:
            s_set = set(abstract_map[key]["sources"])
            s_set.add(source)
            abstract_map[key]["sources"] = sorted(s_set)
    
    with open(ABSTRACT_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(abstract_map, f, ensure_ascii=False, indent=2)

# ======================
# 主处理流程
# ======================

def get_unprocessed_news_files() -> List[Path]:
    processed_ids = set()
    if PROCESSED_IDS_FILE.exists():
        with open(PROCESSED_IDS_FILE, "r") as f:
            processed_ids = set(line.strip() for line in f if line.strip())
    
    unprocessed = []
    for raw_file in sorted(RAW_NEWS_DIR.glob("*.jsonl")):
        deduped_file = DEDUPED_NEWS_DIR / f"{raw_file.stem}_deduped.jsonl"
        if not deduped_file.exists():
            deduper = NewsDeduplicator(threshold=DEDUPE_THRESHOLD)
            deduper.dedupe_file(raw_file, deduped_file)
        unprocessed.append(deduped_file)
    return unprocessed

def process_news_stream():
    log("🚀 启动 Agent1：流式事件与真实实体提取...")
    files = get_unprocessed_news_files()
    if not files:
        log("📭 无可处理新闻文件")
        return

    processed_ids = set()
    if PROCESSED_IDS_FILE.exists():
        with open(PROCESSED_IDS_FILE, "r") as f:
            processed_ids = set(line.strip() for line in f if line.strip())

    total_processed = 0
    with open(PROCESSED_IDS_FILE, "a") as id_log:
        for file_path in files:
            log(f"📄 处理文件: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        news = json.loads(line)
                        news_id = news["id"]
                        if news_id in processed_ids:
                            continue

                        title = news.get("title", "")
                        content = news.get("content", "")
                        source = news.get("source", "unknown")

                        extracted = llm_extract_events(title, content)
                        # 防止 API 限流
                        time.sleep(5)   
                        if extracted:
                            all_entities = []
                            for ev in extracted:
                                all_entities.extend(ev["entities"])
                            if all_entities:
                                update_entities(all_entities, source)
                                update_abstract_map(extracted, source)
                                total_processed += 1

                        id_log.write(news_id + "\n")
                        processed_ids.add(news_id)

                    except Exception as e:
                        log(f"⚠️ 处理单条新闻失败: {e}")
                        continue

             
    log(f"✅ 完成！共处理 {total_processed} 条含有效实体的新闻")

# ======================
# 入口
# ======================

if __name__ == "__main__":
    process_news_stream()