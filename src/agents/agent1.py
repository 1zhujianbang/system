# src/agents/agent1.py
"""
智能体1：新闻关联词与事件类型提取器（支持人工审核开关）
"""
from dotenv import load_dotenv
import os
import time
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
EVENT_KEYWORDS_FILE = DATA_DIR / "event_keywords.json"
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

def load_event_keywords() -> Dict[str, List[str]]:
    """加载事件关键词库"""
    if EVENT_KEYWORDS_FILE.exists():
        with open(EVENT_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

EVENT_KEYWORDS = load_event_keywords()

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

def _add_concept_to_event_keywords(entity: str, event_keywords: dict) -> dict:
    """
    交互式询问是否将 actions 类实体加入事件关键词库
    返回更新后的 event_keywords 字典
    """
    print(f"\n💡 检测到 '{entity}' 被加入 'actions'，是否也作为事件关键词？")
    print("[1] 加入现有事件类型")
    print("[2] 创建新事件类型")
    print("[3] 不加入事件关键词库")

    while True:
        choice = input("请选择 (1/2/3): ").strip()
        if choice == "3":
            return event_keywords
        elif choice == "1":
            print("\n现有事件类型:")
            event_types = list(event_keywords.keys())
            for i, et in enumerate(event_types, 1):
                print(f"  [{i}] {et} → {', '.join(event_keywords[et][:3])}...")
            try:
                idx = int(input("选择编号: ").strip()) - 1
                if 0 <= idx < len(event_types):
                    target_type = event_types[idx]
                    if entity not in event_keywords[target_type]:
                        event_keywords[target_type].append(entity)
                        print(f"✅ '{entity}' 已加入事件类型 '{target_type}'")
                    else:
                        print(f"ℹ️ '{entity}' 已在 '{target_type}' 中")
                    return event_keywords
                else:
                    print("⚠️ 编号超出范围")
            except ValueError:
                print("⚠️ 请输入有效数字")
        elif choice == "2":
            while True:
                new_type = input("输入新事件类型名称（如 'governance'）: ").strip()
                if new_type and re.match(r'^[a-z_][a-z0-9_]*$', new_type):
                    if new_type in event_keywords:
                        print(f"⚠️ 事件类型 '{new_type}' 已存在")
                        continue
                    event_keywords[new_type] = [entity]
                    print(f"🆕 创建新事件类型 '{new_type}' 并添加关键词 '{entity}'")
                    return event_keywords
                else:
                    print("⚠️ 事件类型名需为小写字母、数字、下划线，且不能以数字开头")
        else:
            print("⚠️ 无效选项，请重试")

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
                        if target_cat == "actions":
                            # 只有当 EVENT_KEYWORDS_FILE 存在或可加载时才处理
                            try:
                                with open(EVENT_KEYWORDS_FILE, "r", encoding="utf-8") as f:
                                    current_event_kw = json.load(f)
                            except Exception:
                                current_event_kw = {}

                            updated_event_kw = _add_concept_to_event_keywords(entity, current_event_kw)

                            # 如果有修改，立即保存回文件
                            if updated_event_kw != current_event_kw:
                                with open(EVENT_KEYWORDS_FILE, "w", encoding="utf-8") as f:
                                    json.dump(updated_event_kw, f, ensure_ascii=False, indent=2)
                                print(f"💾 事件关键词库已更新: {EVENT_KEYWORDS_FILE}")
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
    EVENT_KEYWORDS = load_event_keywords()
    scores = {}
    for event_type, keywords in EVENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in full_text)
        if score > 0:
            scores[event_type] = score
    return max(scores, key=scores.get) if scores else None

def load_entity_categories() -> Dict[str, Set[str]]:
    """加载完整的实体分类字典（用于外部模块如 TradingAgent 使用）"""
    if not ENTITIES_FILE.exists():
        load_crypto_entities()  # 触发初始化
    
    with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return {category: set(entities) for category, entities in data.items()}

class Agent1EntityExtractor:
    def __init__(self, auto_update: bool = False):
        """
        :param auto_update: 是否自动将新实体写入主知识库（否则写入 pending 文件）
        """
        self.auto_update = auto_update
        self.known_entities = load_crypto_entities()

    def discover_new_entities(
        self, 
        df: pd.DataFrame, 
        min_freq: int = 2
    ) -> tuple[Set[str], Dict[str, List[str]]]:  # ← 返回两个值
        from collections import Counter
        candidate_counter = Counter()
        # 用于收集上下文：entity -> [title1, title2, ...]
        context_map: Dict[str, List[str]] = {}

        for _, row in df.iterrows():
            title = str(row['title'])
            content = str(row.get('content', ''))
            text = f"{title} {content}"

            if not isinstance(text, str):
                continue

            found_in_row = set()

            # $WIF
            for match in re.findall(r'\$[A-Za-z0-9]{2,10}', text):
                ent = match.upper().lstrip('$')
                if len(ent) >= 2:
                    found_in_row.add(ent)

            # 大写代币符号
            for match in re.findall(r'\b[A-Z]{3,6}\b', text):
                if match not in {"USD", "API", "NFT", "ETF", "SEC", "OKX"}:
                    found_in_row.add(match)

            # 中文项目名
            for match in re.findall(r'[\u4e00-\u9fa5]{2,4}', text):
                found_in_row.add(match)

            # 更新计数 & 上下文
            for ent in found_in_row:
                candidate_counter[ent] += 1
                if ent not in context_map:
                    context_map[ent] = []
                context_map[ent].append(title)  # 或者存整个 text？

        # 筛选有效候选
        valid_candidates = {
            ent for ent, cnt in candidate_counter.items()
            if cnt >= min_freq 
            and ent not in self.known_entities 
            and is_valid_candidate(ent) 
            and ent not in EVENT_KEYWORDS
        }

        # 只保留 valid_candidates 的上下文
        filtered_context_map = {
            ent: context_map[ent] for ent in valid_candidates if ent in context_map
        }

        return valid_candidates, filtered_context_map  # ✅ 返回两个值

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        required_cols = ['title', 'content']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"输入DataFrame缺少必要列: {col}")

        result_df = df.copy()

        # 1. 初筛实体（基于规则/词典）
        result_df['entities'] = result_df.apply(
            lambda row: extract_entities_from_text(
                str(row['title']) + " " + str(row.get('content', '')),
                self.known_entities
            ),
            axis=1
        )

        # 2. 事件类型分类
        result_df['event_type'] = result_df.apply(
            lambda row: classify_event_type(
                str(row['title']), str(row.get('content', ''))
            ),
            axis=1
        )

        # 🔑 3. 发现新实体并收集上下文（用于 LLM 二筛）
        new_entities, context_map = self.discover_new_entities(result_df, min_freq=2)

         # 🤖 4. LLM 二筛（仅非自动模式）
        final_valid_entities = set(self.known_entities)
        if new_entities and not self.auto_update:
            filtered_new, _ = llm_second_pass_filter(new_entities, context_map)
            final_valid_entities.update(filtered_new)
            save_pending_entities(filtered_new)

        elif new_entities and self.auto_update:
            # 自动模式：使用 LLM 的分类结果
            filtered_new, category_map = llm_second_pass_filter(new_entities, context_map)
            final_valid_entities.update(filtered_new)
            self._save_entities_to_main(filtered_new, category_map)

        # ✅ 5. 【关键】用最终有效实体过滤每条新闻的 entities 列
        result_df['entities'] = result_df['entities'].apply(
            lambda ents: [e for e in ents if e in final_valid_entities]
        )

        print(f"🧠 智能体1处理完成：共处理 {len(result_df)} 条新闻")
        print(f"   - 平均每条新闻提取 {result_df['entities'].apply(len).mean():.1f} 个实体")
        print(f"   - 识别出 {result_df['event_type'].notna().sum()} 条带事件类型的新闻")
        
        return result_df

    def _save_entities_to_main(self, new_entities: Set[str], category_map: Dict[str, str]):
        """将新实体按 LLM 预测的类别合并到主知识库"""
        with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 确保所有目标类别存在
        for cat in set(category_map.values()):
            if cat not in data:
                data[cat] = []

        # 按类别添加
        for ent in new_entities:
            cat = category_map.get(ent, "concepts")
            if cat not in data:
                cat = "concepts"
            if ent not in data[cat]:
                data[cat].append(ent)

        # 去重 + 排序
        for key in data:
            data[key] = sorted(list(set(data[key])))

        with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 自动新增 {len(new_entities)} 个实体到主知识库（按 LLM 分类）:")
        for ent in sorted(new_entities):
            print(f"   - '{ent}' → {category_map.get(ent, 'concepts')}")



def llm_second_pass_filter(
    candidates: Set[str], 
    context_map: Dict[str, List[str]]
) -> tuple[Set[str], Dict[str, str]]:  # ← 返回 (有效实体集合, 实体→类别映射)
    """
    使用 DeepSeek API 对初筛候选实体进行二次过滤（批量模式）。
    - 一次性发送所有候选实体
    - 输出格式：{"entity1": {"is_valid": true, "category": "..."}, ...}
    - 若未设置 API Key 或调用失败，则跳过 LLM 过滤，返回原集合（安全降级）
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️ openai 库未安装，跳过 LLM 二筛")
        return candidates, {e: "concepts" for e in candidates}

    # 🔑 加载 API Key
    AGENT_DIR = Path(__file__).parent
    ENV_PATH = AGENT_DIR / ".env.local"
    load_dotenv(ENV_PATH, override=True)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️ 未设置 DEEPSEEK_API_KEY 环境变量，跳过 LLM 二筛")
        return candidates

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # 🧠 构造 entitie 字符串
    sorted_entities = sorted(candidates)
    entities_str = ", ".join(f'"{e}"' for e in sorted_entities)
    contexts_lines = []
    for entity in sorted_entities:
        ctxs = context_map.get(entity, [])
        ctx_str = "\n".join(f"- {ctx}" for ctx in ctxs[:3])  # 最多3条上下文
        contexts_lines.append(f"【{entity}】\n{ctx_str or '（无上下文）'}")
    contexts_str = "\n\n".join(contexts_lines)
    # 💬 提示词
    prompt = f"""你是一个专业的加密货币与区块链领域分析师。请严格评估以下词语是否为有效的领域实体。

**有效实体包括**：
- 加密资产（如 BTC、以太坊、SOL、$WIF）
- 项目/协议（如 Uniswap、Arbitrum、Base链）
- 组织/公司（如 Binance、Coinbase、SEC）
- 技术概念（如 减半、空投、质押、MEV）
- 人名/昵称 （如 麻吉大哥、CZ）
- 行为 （如 分红、合作）
- 事件类型关键词（如 分叉、黑客攻击、监管处罚）

**无效内容包括**：
- 普通动词/形容词（如 上涨、暴跌、利好、宣布）
- 时间词（如 今天、昨日）
- 泛泛词汇（如 市场、投资者、消息）
- 纯数字或符号

词语: {entities_str}
出现上下文: {contexts_str}

请仅输出一个 JSON 对象，格式如下：
{{
  "entity_name1": {{
    "is_valid": true,
    "category": "crypto_assets|organizations|concepts|persons|actions|events|other"
  }},
  "entity_name2": {{
    "is_valid": false,
    "category": "other"
  }}
}}

不要解释，不要额外文本。"""

    total = len(candidates)
    print(f"🤖 启动 DeepSeek LLM 批量二筛：共 {total} 个候选实体")

    try:
        response = client.chat.completions.create(
            # model="deepseek-chat",
            # max_tokens=8192,
            model="deepseek-reasoner",
            extra_body={"thinking": {"type": "enabled"}},
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64000,
            timeout=600,
            stream=False
        )
        content = response.choices[0].message.content.strip()

        # 🧹 清理 Markdown 包裹
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].split("```")[0]
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```")[0]

        # 📦 解析 JSON
        result_dict = json.loads(content)
        print(f"  [DEBUG] DeepSeek 返回原始结果（前3项）: {dict(list(result_dict.items())[:3])}")

        valid_entities = set()
        invalid_entities = set()
        for entity in sorted_entities:
            entry = result_dict.get(entity)
            if isinstance(entry, dict):
                is_valid = entry.get("is_valid")
                if is_valid is True:
                    valid_entities.add(entity)
                    print(f"  ✅ '{entity}' → 有效 ({entry.get('category')})")
                else:
                    invalid_entities.add(entity)
                    print(f"  ❌ '{entity}' → 无效")
            else:
                # LLM 格式错误，但为安全起见保留（或可选择丢弃）
                print(f"  ⚠️ '{entity}' 格式异常，保留（安全策略）")
                valid_entities.add(entity)

        # 检查是否有实体未被 LLM 返回
        missing_entities = set(sorted_entities) - set(result_dict.keys())
        if missing_entities:
            print(f"  ⚠️ LLM 未返回 {len(missing_entities)} 个实体，自动保留: {sorted(missing_entities)}")
            valid_entities.update(missing_entities)

        if invalid_entities:
            # 加载当前停用词（用于去重）
            current_stop_words = load_stop_words()
            new_invalid_words = invalid_entities - current_stop_words

            if new_invalid_words:
                STOP_WORDS_FILE.parent.mkdir(exist_ok=True)
                with open(STOP_WORDS_FILE, "a", encoding="utf-8") as f:
                    for word in sorted(new_invalid_words):
                        f.write("\n" + word)
                print(f"🧹 已将 {len(new_invalid_words)} 个无效词追加到停用词库: {STOP_WORDS_FILE}")
                print(f"   新增词: {sorted(new_invalid_words)}")
            else:
                print("ℹ️ 无效词均已存在于停用词库，无需更新")

        category_map = {}
        for entity in sorted_entities:
            entry = result_dict.get(entity)
            if isinstance(entry, dict) and entry.get("is_valid") is True:
                cat = entry.get("category", "concepts")  # 默认 fallback
                # 确保 category 是主知识库中已有的 key，否则归入 concepts
                if cat not in ["crypto_assets", "organizations", "concepts", "persons", "actions", "events"]:
                    cat = "concepts"
                category_map[entity] = cat

        # 对 missing_entities，也给默认类别（比如 concepts）
        for ent in missing_entities:
            category_map[ent] = "concepts"

        print(f"✅ DeepSeek LLM 二筛完成：{len(valid_entities)}/{total} 个实体通过")
        return valid_entities, category_map 

    except Exception as e:
        print(f"❌ DeepSeek 批量调用失败: {e}")
        print("⚠️ 安全降级：保留所有候选实体，类别设为 concepts")
        return candidates, {e: "concepts" for e in candidates}

if __name__ == "__main__":
    approve_pending_entities()