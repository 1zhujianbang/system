import json
import time
import threading
from datetime import datetime
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from ..utils.tool_function import tools
from .api_client import LLMAPIPool

try:
    import yaml
except ImportError:
    yaml = None


class RateLimiter:
    """简单线程安全令牌桶，控制全局QPS"""
    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.time()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = max(self._next, now) + self.interval

class KnowledgeGraph:
    """
    压缩知识图谱系统，用于管理实体和事件，支持重复检测和更新。
    """
    
    def __init__(self):
        self.tools = tools()
        self.entities_file = self.tools.ENTITIES_FILE
        self.events_file = self.tools.EVENTS_FILE
        self.abstract_map_file = self.tools.ABSTRACT_MAP_FILE
        self.entities_tmp_file = self.tools.ENTITIES_TMP_FILE
        self.abstract_tmp_file = self.tools.ABSTRACT_TMP_FILE
        self.kg_file = self.tools.KNOWLEDGE_GRAPH_FILE
        self.merge_rules_file = self.tools.CONFIG_DIR / "entity_merge_rules.json" # 规则文件路径
        self.merge_rules = {} # 内存中的规则缓存
        self.settings = self._load_agent3_settings()
        self.graph = {
            "entities": {},  # 实体ID -> 实体信息
            "events": {},   # 事件摘要 -> 事件信息
            "edges": []     # 边列表，连接实体和事件
        }
        self.llm_pool = None  # 延迟初始化
        self._tmp_loaded = []  # 记录已加载的tmp文件，刷新完成后清理
        self._load_merge_rules() # 初始化时加载规则
        
    def _init_llm_pool(self):
        """初始化LLM API池"""
        if self.llm_pool is None:
            try:
                self.llm_pool = LLMAPIPool()
                self.tools.log("[知识图谱] LLM API池初始化成功")
            except Exception as e:
                self.tools.log(f"[知识图谱] ❌ 初始化LLM API池失败: {e}")
                self.llm_pool = None
    
    def load_data(self) -> bool:
        """从文件加载实体和事件数据"""
        try:
            if self.entities_file.exists():
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    self.graph['entities'] = json.load(f)
            else:
                self.graph['entities'] = {}
                
            if self.abstract_map_file.exists():
                with open(self.abstract_map_file, 'r', encoding='utf-8') as f:
                    abstract_map = json.load(f)
                    # 转换abstract_map为事件格式
                    self.graph['events'] = {
                        abstract: {
                            "abstract": abstract,
                            "entities": data["entities"],
                            "event_summary": data.get("event_summary", ""),
                            "sources": data.get("sources", []),
                            "first_seen": data.get("first_seen", "")
                        }
                        for abstract, data in abstract_map.items()
                    }
            else:
                self.graph['events'] = {}
            
            # 额外加载 tmp 新增数据（未合并的新增实体/事件）
            self._load_tmp_entities()
            self._load_tmp_events()
                
            self._build_edges()
            self.tools.log(f"[知识图谱] 数据加载成功: {len(self.graph['entities'])} 实体, {len(self.graph['events'])} 事件")
            return True
        except Exception as e:
            self.tools.log(f"[知识图谱] ❌ 加载数据失败: {e}")
            return False
    
    def _build_edges(self):
        """构建实体和事件之间的边"""
        self.graph['edges'] = []
        for abstract, event in self.graph['events'].items():
            for entity in event.get('entities', []):
                if entity in self.graph['entities']:
                    self.graph['edges'].append({
                        "from": entity,
                        "to": abstract,
                        "type": "involved_in"
                    })
    
    def build_graph(self) -> bool:
        """构建知识图谱"""
        return self.load_data()
    
    def _load_merge_rules(self):
        """加载实体合并规则"""
        if self.merge_rules_file.exists():
            try:
                with open(self.merge_rules_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.merge_rules = data.get("merge_rules", {})
                self.tools.log(f"[知识图谱] 已加载 {len(self.merge_rules)} 条实体合并规则")
            except Exception as e:
                self.tools.log(f"[知识图谱] ⚠️ 加载合并规则失败: {e}")
        else:
            self.merge_rules = {}

    def _load_tmp_entities(self):
        """加载并合并 tmp 实体数据"""
        if self.entities_tmp_file.exists():
            try:
                with open(self.entities_tmp_file, "r", encoding="utf-8") as f:
                    tmp_entities = json.load(f)
                    for name, data in tmp_entities.items():
                        if name in self.graph['entities']:
                            self._merge_entity_record(self.graph['entities'][name], data)
                        else:
                            self.graph['entities'][name] = data
                self._tmp_loaded.append(self.entities_tmp_file)
                self.tools.log(f"[知识图谱] 已加载 tmp 实体 {len(tmp_entities)} 条")
            except Exception as e:
                self.tools.log(f"[知识图谱] ⚠️ 加载 tmp 实体失败: {e}")

    def _load_tmp_events(self):
        """加载并合并 tmp 事件数据"""
        if self.abstract_tmp_file.exists():
            try:
                with open(self.abstract_tmp_file, "r", encoding="utf-8") as f:
                    tmp_events = json.load(f)
                    for abstract, data in tmp_events.items():
                        if abstract in self.graph['events']:
                            self._merge_event_record(self.graph['events'][abstract], data)
                        else:
                            self.graph['events'][abstract] = {
                                "abstract": abstract,
                                "entities": data.get("entities", []),
                                "event_summary": data.get("event_summary", ""),
                                "sources": data.get("sources", []),
                                "first_seen": data.get("first_seen", "")
                            }
                self._tmp_loaded.append(self.abstract_tmp_file)
                self.tools.log(f"[知识图谱] 已加载 tmp 事件 {len(tmp_events)} 条")
            except Exception as e:
                self.tools.log(f"[知识图谱] ⚠️ 加载 tmp 事件失败: {e}")

    def _cleanup_tmp_files(self):
        """刷新完成后清理已加载的 tmp 文件"""
        for path in self._tmp_loaded:
            try:
                path.unlink(missing_ok=True)
                self.tools.log(f"[知识图谱] 🗑️ 已清理 tmp 文件: {path}")
            except Exception as e:
                self.tools.log(f"[知识图谱] ⚠️ 无法删除 tmp 文件 {path}: {e}")
        self._tmp_loaded = []

    def _save_merge_rules(self):
        """保存实体合并规则"""
        try:
            data = {
                "merge_rules": self.merge_rules,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            with open(self.merge_rules_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.tools.log(f"[知识图谱] 已保存合并规则库 (共 {len(self.merge_rules)} 条)")
        except Exception as e:
            self.tools.log(f"[知识图谱] ❌ 保存合并规则失败: {e}")

    def _merge_entity_record(self, target: Dict[str, Any], source: Dict[str, Any]):
        """将source实体信息合并到target，不删除节点"""
        if not source:
            return
        # sources
        primary_sources = set()
        for s in target.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        for s in source.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        target['sources'] = list(primary_sources)

        # original_forms
        primary_forms = set()
        for f in target.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)
        for f in source.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)
        target['original_forms'] = list(primary_forms)

        # first_seen 取最早
        primary_first = target.get('first_seen', '')
        source_first = source.get('first_seen', '')
        if source_first and (not primary_first or source_first < primary_first):
            target['first_seen'] = source_first

    def _merge_event_record(self, target: Dict[str, Any], source: Dict[str, Any]):
        """将source事件信息合并到target，不删除节点"""
        if not source:
            return
        # sources
        primary_sources = set()
        for s in target.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        for s in source.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
        target['sources'] = list(primary_sources)

        # entities union
        ents = set(target.get('entities', []))
        ents.update(source.get('entities', []))
        target['entities'] = list(ents)

        # first_seen 取最早
        primary_first = target.get('first_seen', '')
        source_first = source.get('first_seen', '')
        if source_first and (not primary_first or source_first < primary_first):
            target['first_seen'] = source_first

        # event_summary 若缺失则补
        if not target.get('event_summary') and source.get('event_summary'):
            target['event_summary'] = source['event_summary']

    def _load_agent3_settings(self) -> Dict[str, Any]:
        """
        加载agent3相关配置，若无配置则使用默认值。
        """
        defaults = {
            "entity_batch_size": 80,
            "event_batch_size": 15,
            "event_bucket_days": 7,
            "event_bucket_entity_overlap": 1,
            "event_bucket_max_size": 40,
            "event_precluster_similarity": 0.82,
            "event_precluster_limit": 300,
            "entity_precluster_similarity": 0.93,
            "entity_precluster_limit": 500,
            "max_summary_chars": 360,
            "entity_max_workers": 3,
            "event_max_workers": 3,
            "rate_limit_per_sec": 1.0,
            "entity_evidence_per_entity": 2,
            "entity_evidence_max_chars": 400,
        }
        config_file = self.tools.CONFIG_DIR / "config.yaml"
        if yaml and config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    cfg = data.get("agent3_config", {})
                    if isinstance(cfg, dict):
                        for k, v in cfg.items():
                            if k in defaults:
                                defaults[k] = v
            except Exception as e:
                self.tools.log(f"[知识图谱] ⚠️ 加载agent3配置失败，使用默认值: {e}")
        else:
            if not yaml:
                self.tools.log("[知识图谱] ⚠️ 未安装 PyYAML，使用默认配置")
        return defaults

    def _string_similarity(self, a: str, b: str) -> float:
        """字符串相似度(0-1)"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _is_chinese(self, text: str) -> bool:
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _entity_type(self, name: str) -> str:
        """
        轻量类型判定：geo/org/person/product/unknown
        仅用于阻断跨类型合并，宁可 unknown。
        """
        n = name or ""
        geo_kw = ["市", "省", "州", "县", "区", "镇", "乡", "郡", "岛", "府", "道", "自治", "共和国", "王国", "特区", "自治区"]
        org_kw = ["公司", "集团", "银行", "政府", "部", "局", "署", "院", "厅", "行", "党", "机构", "法院", "检察院", "委员会", "组织", "联盟", "理事会", "协会", "基金会", "大学", "学院", "学校", "工厂", "厂", "报", "电视", "新闻", "日报", "晚报", "周报", "社"]
        product_kw = ["系列", "版", "型", "型号", "Pro", "Ultra"]
        if any(k in n for k in geo_kw):
            return "geo"
        if any(k in n for k in org_kw):
            return "org"
        if " " in n or "·" in n:
            return "person"
        if any(k in n for k in product_kw):
            return "product"
        return "unknown"

    def _valid_entity_group(self, group: List[str]) -> bool:
        """跨类型合并拦截：若混合 geo/org/person/product 则拒绝"""
        types = set()
        for name in group:
            t = self._entity_type(name)
            if t != "unknown":
                types.add(t)
        # 如果检测到多种已知类型则视为高风险
        if len(types) > 1:
            self.tools.log(f"[知识图谱] ⚠️ 跨类型合并被阻止: {group} | types={types}")
            return False
        return True

    def _collect_entity_evidence(self, entities_batch: List[str]) -> Dict[str, List[str]]:
        """
        为实体批次收集相关事件摘要+实体，减少幻觉。
        """
        per_entity = int(self.settings.get("entity_evidence_per_entity", 2))
        max_chars = int(self.settings.get("entity_evidence_max_chars", 400))
        evidence: Dict[str, List[str]] = {e: [] for e in entities_batch}
        for abstract, event in self.graph['events'].items():
            ents = event.get('entities', [])
            summary = self._trim_text(event.get('event_summary', "") or abstract, max_chars)
            for e in ents:
                if e in evidence and len(evidence[e]) < per_entity:
                    evidence[e].append(f"{abstract} | {', '.join(ents)} | {summary}")
        return evidence

    def _trim_text(self, text: str, max_chars: int) -> str:
        """控制文本长度，避免prompt过长"""
        if not text or max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _parse_time(self, ts: str) -> float:
        """尽量解析时间戳，失败返回0"""
        if not ts:
            return 0
        try:
            # 支持ISO字符串
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            except Exception:
                return 0

    def _bucket_events_by_time_and_entity(
        self,
        window_days: int,
        min_entity_overlap: int,
        max_bucket_size: int
    ) -> List[Dict[str, Any]]:
        """
        按时间窗口与实体交集分桶事件，减少跨期/跨主体混杂。
        """
        events_items = list(self.graph["events"].items())
        window_sec = window_days * 86400

        # 预排序，时间缺失放末尾
        def _sort_key(item):
            ts = self._parse_time(item[1].get("first_seen", ""))
            return ts if ts > 0 else float("inf")

        events_items.sort(key=_sort_key)

        buckets: List[Dict[str, Any]] = []
        for abstract, event in events_items:
            entities = set(event.get("entities", []))
            ts = self._parse_time(event.get("first_seen", ""))
            placed = False
            for bucket in buckets:
                if len(bucket["keys"]) >= max_bucket_size:
                    continue
                # 时间窗口判定
                if bucket["min_time"] and ts and ts < bucket["min_time"] - window_sec:
                    continue
                if bucket["max_time"] and ts and ts > bucket["max_time"] + window_sec:
                    continue
                # 实体交集判定
                if min_entity_overlap > 0:
                    if not entities or len(bucket["entities"].intersection(entities)) < min_entity_overlap:
                        continue
                # 命中，加入桶
                bucket["keys"].append(abstract)
                bucket["entities"].update(entities)
                if ts:
                    bucket["min_time"] = min(bucket["min_time"] or ts, ts)
                    bucket["max_time"] = max(bucket["max_time"] or ts, ts)
                placed = True
                break
            if not placed:
                buckets.append({
                    "keys": [abstract],
                    "entities": set(entities),
                    "min_time": ts if ts else None,
                    "max_time": ts if ts else None
                })
        self.tools.log(f"[知识图谱] 事件分桶完成，共 {len(buckets)} 个桶")
        return buckets

    def _precluster_entities_by_string(self, entities: List[str], threshold: float, limit: int) -> List[List[str]]:
        """
        基于字符串相似度的轻量预聚类，避免LLM过量输入。
        """
        if len(entities) == 0 or len(entities) > limit:
            return []
        res = []
        used = set()
        for i, ent in enumerate(entities):
            if ent in used:
                continue
            group = [ent]
            used.add(ent)
            for other in entities[i+1:]:
                if other in used:
                    continue
                if self._string_similarity(ent, other) >= threshold:
                    group.append(other)
                    used.add(other)
            if len(group) > 1:
                res.append(group)
        if res:
            self.tools.log(f"[知识图谱] 本地实体预聚类发现 {len(res)} 组可能重复")
        return res

    def _precluster_events_by_string(
        self,
        events_map: Dict[str, Dict[str, Any]],
        keys: List[str],
        threshold: float,
        limit: int,
        max_summary_chars: int
    ) -> List[List[str]]:
        """
        同桶事件的字符串近似聚类，减少LLM负担。
        """
        if len(keys) == 0 or len(keys) > limit:
            return []

        def norm_text(k: str) -> str:
            evt = events_map.get(k, {})
            summary = self._trim_text(evt.get("event_summary", "") or "", max_summary_chars)
            return (k + " " + summary).lower()

        texts = {k: norm_text(k) for k in keys}
        res = []
        used = set()
        for i, key in enumerate(keys):
            if key in used:
                continue
            base = texts.get(key, "")
            group = [key]
            used.add(key)
            for other in keys[i+1:]:
                if other in used:
                    continue
                if self._string_similarity(base, texts.get(other, "")) >= threshold:
                    group.append(other)
                    used.add(other)
            if len(group) > 1:
                res.append(group)
        if res:
            self.tools.log(f"[知识图谱] 本地事件预聚类发现 {len(res)} 组可能重复")
        return res

    def _apply_merge_rules(self) -> bool:
        """
        应用本地合并规则
        返回: 是否有更新
        """
        updated = False
        if not self.merge_rules:
            return False
            
        # 遍历现有实体，看是否匹配规则
        # 注意：需要在遍历时处理，避免字典大小变化问题，通常收集后再处理
        to_merge = []
        for entity in list(self.graph['entities'].keys()):
            if entity in self.merge_rules:
                target = self.merge_rules[entity]
                # 只有当目标实体也存在，或者目标就是我们想要统一到的名称时（这里简化为如果目标在库里或我们决定改名）
                # 为简单起见，我们假设规则是 A -> B，如果 A 存在，就尝试合并到 B。
                # 如果 B 不在库里，就把 A 重命名为 B。
                if target != entity:
                    to_merge.append((target, entity))
        
        for primary, duplicate in to_merge:
            # 如果目标实体不存在，先重命名
            if primary not in self.graph['entities'] and duplicate in self.graph['entities']:
                self.graph['entities'][primary] = self.graph['entities'][duplicate]
                del self.graph['entities'][duplicate]
                # 更新事件引用
                for abstract, event in self.graph['events'].items():
                    entities = event.get('entities', [])
                    if duplicate in entities:
                        event['entities'] = [primary if e == duplicate else e for e in entities]
                self.tools.log(f"[知识图谱][规则] 重命名实体: {duplicate} -> {primary}")
                updated = True
            elif primary in self.graph['entities'] and duplicate in self.graph['entities']:
                # 如果都存在，则合并
                self._merge_entities(primary, duplicate)
                updated = True
                
        return updated

    def compress_with_llm(self) -> Dict[str, List[str]]:
        """
        使用LLM分析压缩知识图谱，输出重复的实体和事件抽象。
        分批处理以避免上下文超长。
        集成规则优先策略。
        """
        # 0. 首先应用本地规则
        rule_applied = self._apply_merge_rules()
        if rule_applied:
            self._save_data() # 规则应用后先保存一次状态
            
        self._init_llm_pool()
        if self.llm_pool is None:
            self.tools.log("[知识图谱] ❌ LLM不可用，跳过压缩")
            return {"duplicate_entities": [], "duplicate_events": []}
        
        # 并发与速率控制
        entity_workers = int(self.settings.get("entity_max_workers", 3))
        event_workers = int(self.settings.get("event_max_workers", 3))
        rate_limit = float(self.settings.get("rate_limit_per_sec", 1.0))
        limiter = RateLimiter(rate_limit) if rate_limit > 0 else None

        all_duplicate_entities = []
        all_duplicate_events = []
        
        # 1. 处理实体 (分批)
        # 排序以增加相似实体相邻的概率
        entities_list = sorted(list(self.graph['entities'].keys()))
        BATCH_SIZE_ENT = int(self.settings.get("entity_batch_size", 80))
        precluster_entities = self._precluster_entities_by_string(
            entities_list,
            threshold=float(self.settings.get("entity_precluster_similarity", 0.93)),
            limit=int(self.settings.get("entity_precluster_limit", 500))
        )
        if precluster_entities:
            all_duplicate_entities.extend(precluster_entities)
        
        entity_batches = []
        for i in range(0, len(entities_list), BATCH_SIZE_ENT):
            entity_batches.append((i // BATCH_SIZE_ENT, entities_list[i:i+BATCH_SIZE_ENT]))

        def _run_entity_batch(idx: int, batch: List[str]) -> List[List[str]]:
            self.tools.log(f"[知识图谱] 处理实体批次 {idx+1}/{len(entity_batches)} (大小: {len(batch)})")
            prompt = self._prepare_entity_compression_prompt_strict(batch)
            response = self._call_llm_limited(prompt, timeout=90, limiter=limiter)
            return self._parse_entity_response(response) if response else []

        new_rules_count = 0
        if entity_batches:
            with ThreadPoolExecutor(max_workers=entity_workers) as executor:
                futures = [executor.submit(_run_entity_batch, idx, batch) for idx, batch in entity_batches]
                for fut in as_completed(futures):
                    batch_dupes = fut.result() or []
                    if batch_dupes:
                        for group in batch_dupes:
                            if len(group) >= 2 and self._valid_entity_group(group):
                                primary, dupes = self._choose_primary_entity(group)
                                for duplicate in dupes:
                                    if duplicate not in self.merge_rules:
                                        self.merge_rules[duplicate] = primary
                                        new_rules_count += 1
                                all_duplicate_entities.append([primary] + dupes)
                    else:
                        all_duplicate_entities.extend(batch_dupes)
        if new_rules_count > 0:
            self._save_merge_rules()
                
        # 2. 处理事件 (分批)
        events_list = sorted(list(self.graph['events'].keys()))
        if not events_list:
            return {
                "duplicate_entities": all_duplicate_entities, 
                "duplicate_events": all_duplicate_events
            }

        BATCH_SIZE_EVT = int(self.settings.get("event_batch_size", 15))
        bucket_days = int(self.settings.get("event_bucket_days", 7))
        bucket_overlap = int(self.settings.get("event_bucket_entity_overlap", 1))
        bucket_max_size = int(self.settings.get("event_bucket_max_size", 40))
        evt_similarity = float(self.settings.get("event_precluster_similarity", 0.82))
        evt_limit = int(self.settings.get("event_precluster_limit", 300))
        max_summary_chars = int(self.settings.get("max_summary_chars", 360))

        buckets = self._bucket_events_by_time_and_entity(bucket_days, bucket_overlap, bucket_max_size)

        def _run_event_bucket(idx: int, bucket: Dict[str, Any]) -> List[List[str]]:
            bucket_keys = bucket.get("keys", [])
            bucket_events = {k: self.graph['events'][k] for k in bucket_keys if k in self.graph['events']}
            local_dupes: List[List[str]] = []

            pre_clusters = self._precluster_events_by_string(
                bucket_events,
                bucket_keys,
                threshold=evt_similarity,
                limit=evt_limit,
                max_summary_chars=max_summary_chars
            )
            if pre_clusters:
                local_dupes.extend(pre_clusters)

            if len(bucket_keys) <= 1:
                self.tools.log(f"[知识图谱] 跳过事件桶 {idx+1}/{len(buckets)}（仅1条，无需去重）")
                return local_dupes

            total_batches = (len(bucket_keys) - 1) // BATCH_SIZE_EVT + 1
            for i in range(0, len(bucket_keys), BATCH_SIZE_EVT):
                batch_keys = bucket_keys[i:i+BATCH_SIZE_EVT]
                batch_events = {
                    k: {
                        **bucket_events.get(k, {}),
                        "event_summary": self._trim_text(
                            bucket_events.get(k, {}).get("event_summary", "") or "",
                            max_summary_chars
                        )
                    }
                    for k in batch_keys
                }
                self.tools.log(
                    f"[知识图谱] 处理事件桶 {idx+1}/{len(buckets)} 的批次 {i//BATCH_SIZE_EVT + 1}/{total_batches} (大小: {len(batch_keys)})"
                )
                prompt = self._prepare_event_compression_prompt(batch_events)
                response = self._call_llm_limited(prompt, timeout=120, limiter=limiter)
                if response:
                    batch_dupes = self._parse_event_response(response)
                    local_dupes.extend(batch_dupes)
            return local_dupes

        if buckets:
            with ThreadPoolExecutor(max_workers=event_workers) as executor:
                futures = [executor.submit(_run_event_bucket, idx, bucket) for idx, bucket in enumerate(buckets)]
                for fut in as_completed(futures):
                    batch_dupes = fut.result() or []
                    all_duplicate_events.extend(batch_dupes)

        return {
            "duplicate_entities": all_duplicate_entities, 
            "duplicate_events": all_duplicate_events
        }

    def _call_llm(self, prompt: str, timeout: int) -> Optional[str]:
        """统一LLM调用"""
        return self.llm_pool.call(
            prompt=prompt,
            max_tokens=4000,
            timeout=timeout,
            retries=2
        )

    def _call_llm_limited(self, prompt: str, timeout: int, limiter: Optional[RateLimiter]) -> Optional[str]:
        """带全局QPS限制的LLM调用"""
        if limiter:
            limiter.acquire()
        return self._call_llm(prompt, timeout)

    def _choose_primary_entity(self, group: List[str]) -> (str, List[str]):
        """
        选择主实体：优先中文，其次 first_seen 最早，再其次名称长度。
        返回 (primary, duplicates)
        """
        if not group:
            return "", []
        best = None
        best_key = None
        for name in group:
            info = self.graph['entities'].get(name, {})
            is_cn = self._is_chinese(name)
            ts = self._parse_time(info.get('first_seen', ''))
            ts_key = ts if ts > 0 else float('inf')
            key = (0 if is_cn else 1, ts_key, len(name))
            if best_key is None or key < best_key:
                best = name
                best_key = key
        duplicates = [n for n in group if n != best]
        return best, duplicates

    def _prepare_entity_compression_prompt_strict(self, entities_batch: List[str]) -> str:
        evidence_map = self._collect_entity_evidence(entities_batch)
        evidence_lines = []
        for ent, evs in evidence_map.items():
            if evs:
                for ev in evs:
                    evidence_lines.append(f"{ent} <= {ev}")

        prompt = f"""你是一名知识图谱专家。任务：仅在有充分证据时认定实体为同一主体（别名/缩写/中英文/法定全称差异）。不要因为行业相似、上下级关系或地域相似而合并。

【高风险误判示例】
- 实体具有特化职能或功效或用途，不可合并
- 行使职能的组织、机构与其下辖的更具体职能的组织、机构不可合并
- “大学” 与 “联盟/协会/部门/央行” 不是同一主体
- 不同国家/地区的同名机构，不可合并
- 上市公司 vs 子公司/控股股东，不可合并
- 政府部门 vs 上级政府，不可合并
- 国家/省州/城市/区县 之间不得互并，也不得跨国合并
- 公司/机构 ≠ 产品/品牌/型号，不得互并
- 人名 ≠ 公司/机构/地理/产品，不得互并
- 媒体名称 ≠ 地点/政府/企业/人名
- 体育俱乐部/赛事 ≠ 城市/国家/政府/个人

【实体列表】
{json.dumps(entities_batch, ensure_ascii=False, indent=2)}

【证据（部分相关事件摘要，供参考，避免幻觉）】
格式: 实体 <= 摘要 | 参与实体 | 描述
{chr(10).join(evidence_lines) if evidence_lines else "（无可用事件，谨慎合并）"}

【要求】
- 主实体优先更通用（或更倾向中国人表达）、更详细、更精确、更学术（“更XX”按优先级顺序）
- 只输出确定为同一主体的组合；不确定就返回空。
- 优先严格匹配：同名、明显译名、缩写展开。
- 不要改写名称格式（不要添加书名号/引号/括号等标点）。
- 地理/机构类合并需同一国家/行政层级；人名不得与非人名合并；公司不得与产品/品牌合并。
- 如果没有重复，返回空列表。

【输出格式】
严格返回 JSON：
{{
  "duplicate_entities": [
    ["主实体", "别名或重复"],
    ["主实体2", "别名2", "别名3"]
  ]
}}
如果没有重复，返回 {{ "duplicate_entities": [] }}。只输出JSON。
"""
        return prompt

    def _prepare_event_compression_prompt(self, events_batch: Dict) -> str:
        prompt = f"""你是一名知识图谱专家。任务：仅在描述“同一具体事实”时才视为重复事件。不要合并不同主体、上下游关联或时间不同的相似事件。

【拒绝合并的情况示例】
- 行业相似但主体不同的事件
- 上游/下游/监管/联盟关系 ≠ 同一事件
- 时间间隔明显不同的多次事件
- 事件具有连续发生或一前一后的关系
- 对于相同实体，不同时间点发生的事件

【事件列表】
格式: 摘要 | 参与实体 | 事件描述
"""
        for abstract, event in events_batch.items():
            entities = event.get('entities', [])
            summary = event.get('event_summary', '')
            prompt += f"{abstract} | {', '.join(entities)} | {summary}\n"

        prompt += """
【任务】
找出语义上高度重叠、描述同一事实的事件。

【输出格式】
严格返回 JSON：
{
  "duplicate_events": [
    ["事件摘要1", "事件摘要2"],
    ["事件摘要3", "事件摘要4", "事件摘要5"]
  ]
}
如果没有重复，返回 { "duplicate_events": [] }。只输出JSON。
"""
        return prompt

    def _parse_entity_response(self, raw_content: str) -> List[List[str]]:
        try:
            data = self._extract_json(raw_content)
            res = data.get("duplicate_entities", [])
            return res if isinstance(res, list) else []
        except Exception:
            return []

    def _parse_event_response(self, raw_content: str) -> List[List[str]]:
        try:
            data = self._extract_json(raw_content)
            res = data.get("duplicate_events", [])
            return res if isinstance(res, list) else []
        except Exception:
            return []

    def _extract_json(self, text: str) -> Dict:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```")[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```")[0]
        return json.loads(text)
    
    # 旧方法保留或删除（这里替换旧的 _prepare_compression_prompt 和 _parse_llm_response）
    
    def update_entities_and_events(self, duplicates: Dict[str, List[List[str]]]):
        """根据重复检测结果更新实体库和事件库"""
        updated = False
        
        # 合并重复实体
        for group in duplicates.get("duplicate_entities", []):
            if len(group) < 2:
                continue
            if not self._valid_entity_group(group):
                continue
            primary, dupes = self._choose_primary_entity(group)
            # 确保主实体存在；若不存在但有重复实体存在，可交换
            if primary not in self.graph['entities'] and dupes:
                for d in dupes:
                    if d in self.graph['entities']:
                        primary, dupes = d, [x for x in group if x != d]
                        break
            for duplicate in dupes:
                if duplicate in self.graph['entities'] and primary in self.graph['entities']:
                    self._merge_entities(primary, duplicate)
                    updated = True
        
        # 合并重复事件
        for group in duplicates.get("duplicate_events", []):
            if len(group) < 2:
                continue
            primary = group[0]
            for duplicate in group[1:]:
                if duplicate in self.graph['events']:
                    self._merge_events(primary, duplicate)
                    updated = True
        
        if updated:
            self._save_data()
            self.tools.log("[知识图谱] 实体和事件更新完成")
        else:
            self.tools.log("[知识图谱] 无重复项需要更新")
    
    def _merge_entities(self, primary: str, duplicate: str):
        """合并重复实体"""
        if primary not in self.graph['entities'] or duplicate not in self.graph['entities']:
            return
        
        primary_data = self.graph['entities'][primary]
        duplicate_data = self.graph['entities'][duplicate]
        
        # 合并sources (确保转换为可哈希的tuple或直接列表处理)
        primary_sources = set()
        for s in primary_data.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue # 暂时忽略复杂结构
            else: primary_sources.add(s)
            
        for s in duplicate_data.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue 
            else: primary_sources.add(s)
            
        # 转回list
        primary_data['sources'] = list(primary_sources)
        
        # 合并original_forms
        primary_forms = set()
        for f in primary_data.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)
            
        for f in duplicate_data.get('original_forms', []):
            if isinstance(f, list): primary_forms.add(tuple(f))
            elif isinstance(f, dict): continue
            else: primary_forms.add(f)

        # 将重复实体名也作为主实体的其他表述记录，防止丢失别名
        primary_forms.add(duplicate)
        primary_forms.add(primary)
            
        primary_data['original_forms'] = list(primary_forms)
        
        # 更新first_seen为更早的时间
        primary_first = primary_data.get('first_seen', '')
        duplicate_first = duplicate_data.get('first_seen', '')
        if duplicate_first and (not primary_first or duplicate_first < primary_first):
            primary_data['first_seen'] = duplicate_first
        
        # 删除重复实体
        del self.graph['entities'][duplicate]
        
        # 更新事件中的实体引用
        for abstract, event in self.graph['events'].items():
            entities = event.get('entities', [])
            if duplicate in entities:
                # 替换为primary，并去重
                new_entities = [primary if ent == duplicate else ent for ent in entities]
                # 去重
                unique_entities = []
                seen = set()
                for ent in new_entities:
                    if ent not in seen:
                        seen.add(ent)
                        unique_entities.append(ent)
                event['entities'] = unique_entities
        
        self.tools.log(f"[知识图谱] 合并实体: {duplicate} -> {primary}")
    
    def _merge_events(self, primary: str, duplicate: str):
        """合并重复事件"""
        if primary not in self.graph['events'] or duplicate not in self.graph['events']:
            return
        
        primary_event = self.graph['events'][primary]
        duplicate_event = self.graph['events'][duplicate]
        
        # 合并sources
        primary_sources = set()
        for s in primary_event.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
            
        for s in duplicate_event.get('sources', []):
            if isinstance(s, list): primary_sources.add(tuple(s))
            elif isinstance(s, dict): continue
            else: primary_sources.add(s)
            
        primary_event['sources'] = list(primary_sources)
        
        # 合并entities
        primary_entities = set(primary_event.get('entities', []))
        duplicate_entities = set(duplicate_event.get('entities', []))
        primary_event['entities'] = list(primary_entities.union(duplicate_entities))
        
        # 更新first_seen
        primary_first = primary_event.get('first_seen', '')
        duplicate_first = duplicate_event.get('first_seen', '')
        if duplicate_first and (not primary_first or duplicate_first < primary_first):
            primary_event['first_seen'] = duplicate_first
        
        # 事件描述合并：保留更详细的
        if not primary_event.get('event_summary') and duplicate_event.get('event_summary'):
            primary_event['event_summary'] = duplicate_event['event_summary']
        
        # 删除重复事件
        del self.graph['events'][duplicate]
        
        self.tools.log(f"[知识图谱] 合并事件: {duplicate} -> {primary}")
    
    def _save_data(self):
        """保存更新后的数据到文件"""
        try:
            # 保存实体
            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(self.graph['entities'], f, ensure_ascii=False, indent=2)
            
            # 保存事件（abstract_map格式）
            abstract_map = {}
            for abstract, event in self.graph['events'].items():
                abstract_map[abstract] = {
                    "entities": event.get('entities', []),
                    "event_summary": event.get('event_summary', ''),
                    "sources": event.get('sources', []),
                    "first_seen": event.get('first_seen', '')
                }
            
            with open(self.abstract_map_file, 'w', encoding='utf-8') as f:
                json.dump(abstract_map, f, ensure_ascii=False, indent=2)
            
            # 保存知识图谱状态（可选）
            with open(self.kg_file, 'w', encoding='utf-8') as f:
                json.dump(self.graph, f, ensure_ascii=False, indent=2)
            
            self.tools.log("[知识图谱] 数据保存完成")
        except Exception as e:
            self.tools.log(f"[知识图谱] ❌ 保存数据失败: {e}")
    
    def append_only_update(self, events_list: List[Dict[str, Any]], default_source: str = "auto_pipeline", allow_append_original_forms: bool = True) -> Dict[str, int]:
        """
        只追加新数据，不改动已有实体/事件的旧字段。
        - 实体已存在：不改 first_seen/sources，默认仅可选地追加原始表述
        - 事件已存在（同 abstract）：跳过，不改旧事件
        """
        if not events_list:
            return {"added_entities": 0, "added_events": 0}

        if not self.build_graph():
            return {"added_entities": 0, "added_events": 0}

        # 准备 LLM 去重映射（仅映射“新实体”到“已有实体”，不改旧实体字段）
        self._init_llm_pool()
        merge_rules = self.merge_rules or {}
        existing_entities = set(self.graph["entities"].keys())

        # 收集新实体集合
        new_entities_set = set()
        for ev in events_list:
            ents = ev.get("entities", []) or []
            ents = [merge_rules.get(e, e) for e in ents if e]
            for e in ents:
                if e not in existing_entities:
                    new_entities_set.add(e)

        # 构建映射：新实体 -> 已有实体（LLM 判断可能同名）
        llm_merge_map: Dict[str, str] = {}
        if self.llm_pool and new_entities_set:
            # 分桶降低上下文长度：按首字母/字符分桶
            from collections import defaultdict
            from concurrent.futures import ThreadPoolExecutor, as_completed
            buckets = defaultdict(list)
            for ent in new_entities_set:
                prefix = ent[0] if ent else "#"
                buckets[prefix].append(ent)

            llm_lock = threading.Lock()
            max_workers = int(self.settings.get("entity_max_workers", 3)) or 1

            def handle_bucket(prefix: str, bucket_new: List[str]):
                local_map = {}
                # 取同前缀的部分已有实体做对比，避免过长
                existing_subset = [e for e in existing_entities if e.startswith(prefix)]
                existing_subset = existing_subset[: max(10, min(80, len(existing_subset)))]
                candidates = list(existing_subset) + list(bucket_new)
                if len(candidates) < 2:
                    return local_map
                try:
                    prompt = self._prepare_entity_compression_prompt_strict(candidates)
                    resp = self._call_llm_limited(prompt, timeout=60, limiter=None)
                    groups = self._parse_entity_response(resp) if resp else []
                    for g in groups:
                        if len(g) < 2:
                            continue
                        primary, dupes = self._choose_primary_entity(g)
                        if primary in existing_entities:
                            for d in dupes:
                                if d in new_entities_set:
                                    local_map[d] = primary
                except Exception as e:
                    self.tools.log(f"[知识图谱] 追加模式 LLM 去重失败: {e}")
                return local_map

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(handle_bucket, p, b) for p, b in buckets.items()]
                for fut in as_completed(futures):
                    res_map = fut.result() or {}
                    if res_map:
                        with llm_lock:
                            llm_merge_map.update(res_map)

        added_entities = 0
        added_events = 0

        def normalize_entity(name: str) -> str:
            if not name:
                return name
            name = merge_rules.get(name, name)
            return llm_merge_map.get(name, name)

        for ev in events_list:
            ents = ev.get("entities", []) or []
            ents = [normalize_entity(e) for e in ents]
            ents_original = ev.get("entities_original") or ents
            src = ev.get("source", default_source)
            published_at = ev.get("published_at") or ev.get("datetime") or ""

            # 追加实体（仅当不存在）
            for ent, ent_orig in zip(ents, ents_original):
                if not ent:
                    continue
                if ent not in self.graph["entities"]:
                    self.graph["entities"][ent] = {
                        "first_seen": published_at or datetime.utcnow().isoformat(),
                        "sources": [src] if src else [],
                        "original_forms": [ent_orig] if ent_orig else []
                    }
                    added_entities += 1
                else:
                    # 不改旧字段，仅可选追加原始表述
                    if allow_append_original_forms and ent_orig:
                        forms = self.graph["entities"][ent].get("original_forms", [])
                        if ent_orig not in forms:
                            forms.append(ent_orig)
                            self.graph["entities"][ent]["original_forms"] = forms

            # 追加事件（仅当摘要不存在）
            abstract = ev.get("abstract")
            if abstract and abstract not in self.graph["events"]:
                self.graph["events"][abstract] = {
                    "abstract": abstract,
                    "entities": ents,
                    "event_summary": ev.get("event_summary", ""),
                    "sources": [src] if src else [],
                    "first_seen": published_at
                }
                added_events += 1

        # 重建边并保存（只在有新增时）
        if added_entities or added_events:
            self._build_edges()
            self._save_data()
            self.tools.log(f"[知识图谱] 追加模式完成：新增实体 {added_entities}，新增事件 {added_events}")
        else:
            self.tools.log("[知识图谱] 追加模式：没有新增实体/事件")

        return {"added_entities": added_entities, "added_events": added_events}
    
    def refresh_graph(self):
        """刷新知识图谱：构建、压缩、更新"""
        self.tools.log("[知识图谱] 开始刷新知识图谱")
        
        # 构建图
        if not self.build_graph():
            self.tools.log("[知识图谱] ❌ 构建图失败")
            return
        
        # 压缩：使用LLM检测重复
        duplicates = self.compress_with_llm()
        
        # 更新实体和事件
        self.update_entities_and_events(duplicates)
        
        self.tools.log("[知识图谱] 知识图谱刷新完成")
        # 清理已加载的临时文件
        self._cleanup_tmp_files()

# 全局函数，供agent1和agent2调用
def refresh_graph():
    """刷新知识图谱（供外部调用）"""
    kg = KnowledgeGraph()
    kg.refresh_graph()

def append_only_update_graph(events_list: List[Dict[str, Any]], default_source: str = "auto_pipeline", allow_append_original_forms: bool = True) -> Dict[str, int]:
    """
    只追加新事件/实体到现有图谱，不修改旧记录。
    """
    kg = KnowledgeGraph()
    return kg.append_only_update(events_list, default_source=default_source, allow_append_original_forms=allow_append_original_forms)

def build_graph() -> bool:
    """构建知识图谱"""
    kg = KnowledgeGraph()
    return kg.build_graph()

if __name__ == "__main__":
    # 测试代码
    kg = KnowledgeGraph()
    kg.refresh_graph()
