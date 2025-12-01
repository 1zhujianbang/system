# src/agents/agent1.py
"""
智能体1：新闻关联词与事件类型提取器（支持人工审核开关）
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
import re
import warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ENTITIES_FILE = DATA_DIR / "crypto_entities.json"
PENDING_ENTITIES_FILE = DATA_DIR / "pending_entities.json"
STOP_WORDS_FILE = DATA_DIR / "stop_words.txt"

def load_stop_words() -> Set[str]:
    """从统一停用词文件加载（支持注释和空行）"""
    stop_words = set()
    if STOP_WORDS_FILE.exists():
        with open(STOP_WORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith("#"):
                    stop_words.add(word)
    return stop_words

STOP_WORDS = load_stop_words()

def is_valid_candidate(entity: str) -> bool:
    """判断一个候选词是否值得进入待审核池"""
    word = entity.strip()
    if not word:
        return False
    if len(word) == 1:
        return False
    if word in STOP_WORDS:
        return False
    if word.isdigit():
        return False
    if re.match(r'^[0-9+\-\.%]+$', word):  # 纯数字/符号
        return False
    # 排除纯标点或特殊字符
    if not any(c.isalnum() for c in word):
        return False
    return True

EVENT_KEYWORDS = {
    "regulation": ["监管", "合规", "SEC", "罚款", "禁令", "牌照", "法律"],
    "hack": ["黑客", "被盗", "漏洞", "攻击", "安全事件"],
    "listing": ["上线", "上架", "交易对", "支持"],
    "partnership": ["合作", "战略合作", "联盟", "集成"],
    "upgrade": ["升级", "主网", "硬分叉", "技术更新"],
    "market": ["暴跌", "暴涨", "行情", "市值", "价格"],
    "adoption": ["采用", "支付", "集成到", "企业采用"]
}

def load_crypto_entities() -> Set[str]:
    if not ENTITIES_FILE.exists():
        default_data = {
            "crypto_assets": ["BTC", "ETH", "SOL", "USDT", "比特币", "以太坊"],
            "organizations": ["Binance", "SEC", "美联储"],
            "concepts": ["ETF", "减半", "DeFi"]
        }
        DATA_DIR.mkdir(exist_ok=True)
        with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)
        print(f"🆕 首次运行：已创建默认实体库 {ENTITIES_FILE}")

    with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_entities = set()
    for group in data.values():
        all_entities.update(group)
    return all_entities

def save_pending_entities(candidates: Set[str]):
    """保存通过初筛的候选实体"""
    if not candidates:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if PENDING_ENTITIES_FILE.exists():
        with open(PENDING_ENTITIES_FILE, "r", encoding="utf-8") as f:
            pending = json.load(f)
    else:
        pending = {}

    now = datetime.now(timezone.utc).isoformat()
    for ent in candidates:
        if ent not in pending:
            pending[ent] = {
                "first_seen": now,
                "status": "pending",
                "source_contexts": []
            }

    with open(PENDING_ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)

    print(f"📝 初筛后新增 {len(candidates)} 个候选实体到待审核文件")

def approve_pending_entities():
    """
    交互式审核待批准实体。
    支持：
      1. 加入停用词库（追加到 stop_words.txt）
      2. 加入已有分类（编号选择）
      3. 创建新分类
    """
    if not PENDING_ENTITIES_FILE.exists():
        print("📭 待审核文件不存在")
        return

    with open(PENDING_ENTITIES_FILE, "r", encoding="utf-8") as f:
        pending = json.load(f)

    pending_entities = {
        ent: info for ent, info in pending.items()
        if info.get("status") == "pending"
    }

    if not pending_entities:
        print("✅ 所有待审实体已处理完毕！")
        return

    # 加载主知识库
    with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    categories = list(main_data.keys())
    total = len(pending_entities)

    # 加载当前停用词（用于去重）
    current_stop_words = load_stop_words()
    new_stop_words_to_add = set()
    approved_updates = {}  # entity -> target_category or "__STOP__"

    for i, (entity, info) in enumerate(pending_entities.items(), 1):
        print(f"\n{'='*50}")
        print(f"🔍 [{i}/{total}] 审核实体: '{entity}'")
        print("[1] 加入停用词库（永久忽略）")
        print("[2] 加入已有分类")
        print("[3] 创建新分类")
        print("[q] 退出审核")

        choice = input("请选择 (1/2/3/q): ").strip().lower()
        if choice == 'q':
            print("⏹️ 审核已退出。")
            break
        elif choice == '1':
            if entity in current_stop_words:
                print(f"ℹ️ '{entity}' 已在停用词库中")
            else:
                new_stop_words_to_add.add(entity)
                approved_updates[entity] = "__STOP__"
                print(f"🗑️ '{entity}' 将被加入停用词库")
        elif choice == '2':
            print("\n已有分类:")
            for idx, cat in enumerate(categories, 1):
                print(f"  [{idx}] {cat}")
            while True:
                try:
                    sel = input("请选择编号: ").strip()
                    if not sel:
                        continue
                    idx = int(sel) - 1
                    if 0 <= idx < len(categories):
                        target_cat = categories[idx]
                        approved_updates[entity] = target_cat
                        print(f"✅ '{entity}' 将加入分类 '{target_cat}'")
                        break
                    else:
                        print("⚠️ 编号超出范围，请重试")
                except ValueError:
                    print("⚠️ 请输入有效数字")
        elif choice == '3':
            while True:
                new_cat = input("请输入新分类名（如 protocols）: ").strip()
                if new_cat and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', new_cat):
                    if new_cat not in main_data:
                        main_data[new_cat] = []
                        categories.append(new_cat)
                    approved_updates[entity] = new_cat
                    print(f"🆕 创建新分类 '{new_cat}' 并添加 '{entity}'")
                    break
                else:
                    print("⚠️ 分类名需为合法 Python 标识符（字母、数字、下划线，不能以数字开头）")
        else:
            print("⚠️ 无效选项，跳过此实体")
            continue

    if not approved_updates:
        print("\nℹ️ 未做任何修改")
        return

    # --- 更新主知识库（非停用词项）---
    for entity, target in approved_updates.items():
        if target != "__STOP__":
            if entity not in main_data[target]:
                main_data[target].append(entity)

    # 去重 + 排序
    for key in main_data:
        main_data[key] = sorted(list(set(main_data[key])))

    with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)

    # --- 追加新停用词到文件 ---
    if new_stop_words_to_add:
        with open(STOP_WORDS_FILE, "a", encoding="utf-8") as f:
            for word in sorted(new_stop_words_to_add):
                f.write("\n" + word)
        print(f"\n💾 已将 {len(new_stop_words_to_add)} 个新词追加到停用词库: {STOP_WORDS_FILE}")

    # --- 清理 pending 文件 ---
    remaining = {ent: info for ent, info in pending.items() if ent not in approved_updates}
    with open(PENDING_ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 审核完成！共处理 {len(approved_updates)} 个实体。")
    print(f"   - 主知识库已更新: {ENTITIES_FILE}")
    print(f"   - 待审文件剩余 {len(remaining)} 项")

def extract_entities_from_text(text: str, known_entities: Set[str]) -> List[str]:
    if not isinstance(text, str):
        return []
    found = set()
    for entity in known_entities:
        if entity in text:
            found.add(entity)
    return sorted(found)

def classify_event_type(title: str, content: str) -> Optional[str]:
    full_text = (title + " " + content) if isinstance(content, str) else title
    if not isinstance(full_text, str):
        return None
    scores = {}
    for event_type, keywords in EVENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in full_text)
        if score > 0:
            scores[event_type] = score
    return max(scores, key=scores.get) if scores else None

class Agent1EntityExtractor:
    def __init__(self, auto_update: bool = False):
        """
        :param auto_update: 是否自动将新实体写入主知识库（否则写入 pending 文件）
        """
        self.auto_update = auto_update
        self.known_entities = load_crypto_entities()

    def discover_new_entities(self, df: pd.DataFrame, min_freq: int = 2) -> Set[str]:
        from collections import Counter
        candidate_counter = Counter()

        for _, row in df.iterrows():
            text = f"{row['title']} {row.get('content', '')}"
            if not isinstance(text, str):
                continue

            # $WIF
            for match in re.findall(r'\$[A-Za-z0-9]{2,10}', text):
                candidate_counter[match.upper().lstrip('$')] += 1

            # 大写代币符号
            for match in re.findall(r'\b[A-Z]{3,6}\b', text):
                if match not in {"USD", "API", "NFT", "ETF", "SEC", "OKX"}:
                    candidate_counter[match] += 1

            # 中文项目名
            for match in re.findall(r'[\u4e00-\u9fa5]{2,4}', text):
                candidate_counter[match] += 1

        valid_candidates = {
            ent for ent, cnt in candidate_counter.items()
            if cnt >= min_freq and ent not in self.known_entities and is_valid_candidate(ent)
        }
        return valid_candidates

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        required_cols = ['title', 'content']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"输入DataFrame缺少必要列: {col}")

        result_df = df.copy()

        result_df['entities'] = result_df.apply(
            lambda row: extract_entities_from_text(
                str(row['title']) + " " + str(row.get('content', '')),
                self.known_entities
            ),
            axis=1
        )

        result_df['event_type'] = result_df.apply(
            lambda row: classify_event_type(
                str(row['title']), str(row.get('content', ''))
            ),
            axis=1
        )

        # 🔑 关键：根据 auto_update 决定如何处理新实体
        new_entities = self.discover_new_entities(df, min_freq=2)
        if new_entities:
            if self.auto_update:
                self._save_entities_to_main(new_entities)
            else:
                save_pending_entities(new_entities)

        print(f"🧠 智能体1处理完成：共处理 {len(result_df)} 条新闻")
        print(f"   - 平均每条新闻提取 {result_df['entities'].apply(len).mean():.1f} 个实体")
        print(f"   - 识别出 {result_df['event_type'].notna().sum()} 条带事件类型的新闻")
        
        return result_df

    def _save_entities_to_main(self, new_entities: Set[str]):
        """将新实体合并到主知识库（仅当 auto_update=True 时调用）"""
        with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ent in new_entities:
            if ent.isupper() and len(ent) <= 6:
                data["crypto_assets"].append(ent)
            elif any(kw in ent for kw in ["交易所", "币", "Coin"]):
                data["crypto_assets"].append(ent)
            else:
                data["concepts"].append(ent)

        for key in data:
            data[key] = sorted(list(set(data[key])))

        with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 自动新增 {len(new_entities)} 个实体到主知识库: {sorted(new_entities)}")

if __name__ == "__main__":
    approve_pending_entities()