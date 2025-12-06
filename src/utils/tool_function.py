import os
import sys
import json
import re
import time
from typing import Set
from datetime import datetime, timezone
import hashlib
from pathlib import Path

class tools:
    # ======================
    # 路径与配置（类变量，可通过类直接访问）
    # ======================
    
    # 基础路径（类变量）
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    CONFIG_DIR = ROOT_DIR / "config"
    RAW_NEWS_DIR = DATA_DIR / "raw_news"
    DEDUPED_NEWS_DIR = DATA_DIR / "deduped_news"
    LOG_FILE = DATA_DIR / "logs" / "agent1.log"
    
    # 数据文件（类变量）
    ENTITIES_FILE = DATA_DIR / "entities.json"
    EVENTS_FILE = DATA_DIR / "events.json"
    ABSTRACT_MAP_FILE = DATA_DIR / "abstract_to_event_map.json"
    PROCESSED_IDS_FILE = DATA_DIR / "processed_ids.txt"
    STOP_WORDS_FILE = DATA_DIR / "stop_words.txt"
    KNOWLEDGE_GRAPH_FILE = DATA_DIR / "knowledge_graph.json"
    
    # 配置常量（类变量）
    DEDUPE_THRESHOLD = int(os.getenv("AGENT1_DEDUPE_THRESHOLD", "3"))
    
    def __init__(self):
        # 确保目录存在
        for d in [self.DATA_DIR, self.RAW_NEWS_DIR, self.DEDUPED_NEWS_DIR, self.DATA_DIR / "logs"]:
            d.mkdir(parents=True, exist_ok=True)

        # 实例变量
        stop_words = set()
        if self.STOP_WORDS_FILE.exists():
            with open(self.STOP_WORDS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        stop_words.add(word)
        self.STOP_WORDS = stop_words

    # ======================
    # 工具函数
    # ======================

    def log(self, msg: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] {msg}"
        print(line)
        with open(self.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_stop_words(self) -> Set[str]:
        stop_words = set()
        if self.STOP_WORDS_FILE.exists():
            with open(self.STOP_WORDS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        stop_words.add(word)
        return stop_words

    

    def is_valid_entity(self, entity: str) -> bool:
        '''
            检查实体是否有效
            暂时弃用
        '''
        word = entity.strip()
        if word in self.STOP_WORDS:
            return False
        return True

    def simhash(self, text: str, bits=64) -> int:
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

    def hamming_distance(self, h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count("1")
