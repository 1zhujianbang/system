# src/agents/knowledge_graph.py
"""
知识图谱构建模块

核心功能：
1. 从现有的实体库和事件映射中构建知识图谱
2. 维护实体-事件-实体的关系网络
3. 支持动态更新和查询
4. 提供可解释性的知识图谱数据结构
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import logging

from ..utils.tool_function import tools

# 初始化工具
tools = tools()

class KnowledgeGraph:
    """动态可解释知识图谱类"""
    
    def __init__(self):
        """
        初始化知识图谱
        """
        print(f"[DEBUG] 初始化知识图谱实例，当前时间: {datetime.now().isoformat()}")
        # 节点：实体和事件
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # 关系：实体-事件-实体
        self.relationships: List[Dict[str, Any]] = []
        # 实体索引
        self.entity_index: Dict[str, str] = {}
        # 事件索引
        self.event_index: Dict[str, str] = {}
        # 最后更新时间
        self.last_updated: str = ""
        print(f"[DEBUG] 知识图谱初始化完成: nodes={len(self.nodes)}, relationships={len(self.relationships)}")
    
    def load_from_existing_data(self) -> None:
        """
        从现有的实体库和事件映射文件加载数据并构建知识图谱
        """
        print("[DEBUG] load_from_existing_data: 开始从数据源加载知识图谱数据")
        tools.log("🔄 开始从现有数据构建知识图谱...")
        
        try:
            # 加载实体数据
            print("[DEBUG] load_from_existing_data: 开始加载实体数据")
            entities = self._load_entities()
            print(f"[DEBUG] load_from_existing_data: 实体数据加载完成，实体数: {len(entities)}")
            
            # 加载事件数据
            print("[DEBUG] load_from_existing_data: 开始加载事件数据")
            events = self._load_events()
            print(f"[DEBUG] load_from_existing_data: 事件数据加载完成，事件数: {len(events)}")
            
            # 构建知识图谱
            print("[DEBUG] load_from_existing_data: 开始构建知识图谱")
            self._build_graph(entities, events)
            
            # 更新最后更新时间
            self.last_updated = datetime.now().isoformat()
            
            print(f"[DEBUG] load_from_existing_data: 知识图谱构建完成: 节点数={len(self.nodes)}, 关系数={len(self.relationships)}")
            tools.log(f"✅ 知识图谱构建完成: {len(self.nodes)}个节点, {len(self.relationships)}个关系")
        except Exception as e:
            print(f"[DEBUG-ERROR] load_from_existing_data: 加载构建过程失败: {e}")
            import traceback
            print(f"[DEBUG-ERROR] load_from_existing_data: 错误堆栈: {traceback.format_exc()}")
            raise
    
    def _load_entities(self) -> Dict[str, Dict[str, Any]]:
        """
        从实体库文件加载实体数据
        """
        print("[DEBUG] _load_entities: 开始加载实体数据")
        entities = {}
        
        print(f"[DEBUG] _load_entities: 检查实体库文件: {tools.ENTITIES_FILE}")
        if tools.ENTITIES_FILE.exists():
            try:
                print(f"[DEBUG] _load_entities: 实体库文件存在，尝试加载")
                with open(tools.ENTITIES_FILE, "r", encoding="utf-8") as f:
                    entities = json.load(f)
                print(f"[DEBUG] _load_entities: 成功从文件加载 {len(entities)} 个实体")
                # 打印前3个实体作为示例
                entity_keys = list(entities.keys())
                for i in range(min(3, len(entity_keys))):
                    print(f"[DEBUG] _load_entities: 实体示例 {i+1}: {entity_keys[i]}")
                tools.log(f"📊 加载了 {len(entities)} 个实体")
            except Exception as e:
                print(f"[DEBUG-ERROR] _load_entities: 加载实体库失败: {e}")
                tools.log(f"⚠️ 加载实体库失败: {e}")
        else:
            print(f"[DEBUG-WARNING] _load_entities: 实体库文件不存在: {tools.ENTITIES_FILE}")
            # 如果文件不存在，提供一些模拟数据用于调试
            print("[DEBUG] _load_entities: 使用模拟数据创建实体")
            entities = {
                "实体1": {"first_seen": "2023-01-01", "sources": ["source1"], "original_forms": ["实体1"]},
                "实体2": {"first_seen": "2023-01-02", "sources": ["source2"], "original_forms": ["实体2"]},
                "中国": {"first_seen": "2023-01-03", "sources": ["source3"], "original_forms": ["中国"]},
                "美国": {"first_seen": "2023-01-04", "sources": ["source4"], "original_forms": ["美国"]},
                "公司A": {"first_seen": "2023-01-05", "sources": ["source5"], "original_forms": ["公司A"]}
            }
            print(f"[DEBUG] _load_entities: 已创建 {len(entities)} 个模拟实体")
            tools.log(f"⚠️ 实体库文件不存在: {tools.ENTITIES_FILE}")
        
        return entities
    
    def _load_events(self) -> Dict[str, Dict[str, Any]]:
        """
        从事件映射文件加载事件数据
        """
        print("[DEBUG] _load_events: 开始加载事件数据")
        events = {}
        
        print(f"[DEBUG] _load_events: 检查事件映射文件: {tools.ABSTRACT_MAP_FILE}")
        if tools.ABSTRACT_MAP_FILE.exists():
            try:
                print(f"[DEBUG] _load_events: 事件映射文件存在，尝试加载")
                with open(tools.ABSTRACT_MAP_FILE, "r", encoding="utf-8") as f:
                    events = json.load(f)
                print(f"[DEBUG] _load_events: 成功从文件加载 {len(events)} 个事件")
                # 打印前3个事件作为示例
                event_keys = list(events.keys())
                for i in range(min(3, len(event_keys))):
                    print(f"[DEBUG] _load_events: 事件示例 {i+1}: {event_keys[i][:30]}...")
                    # 检查事件是否包含entities字段
                    if "entities" in events[event_keys[i]]:
                        print(f"[DEBUG] _load_events: 事件 {i+1} 包含实体数: {len(events[event_keys[i]]['entities'])}")
                tools.log(f"📊 加载了 {len(events)} 个事件")
            except Exception as e:
                print(f"[DEBUG-ERROR] _load_events: 加载事件映射失败: {e}")
                tools.log(f"⚠️ 加载事件映射失败: {e}")
        else:
            print(f"[DEBUG-WARNING] _load_events: 事件映射文件不存在: {tools.ABSTRACT_MAP_FILE}")
            # 如果文件不存在，提供一些模拟数据用于调试
            print("[DEBUG] _load_events: 使用模拟数据创建事件")
            events = {
                "事件1摘要": {
                    "event_summary": "这是事件1的详细描述", 
                    "first_seen": "2023-01-01", 
                    "sources": ["source1"],
                    "entities": ["实体1", "实体2"]
                },
                "事件2摘要": {
                    "event_summary": "这是事件2的详细描述", 
                    "first_seen": "2023-01-02", 
                    "sources": ["source2"],
                    "entities": ["实体1"]
                },
                "中国与美国举行贸易谈判": {
                    "event_summary": "中美两国进行贸易谈判", 
                    "first_seen": "2023-02-01", 
                    "sources": ["source3"],
                    "entities": ["中国", "美国"]
                },
                "公司A发布新产品": {
                    "event_summary": "公司A推出全新产品线", 
                    "first_seen": "2023-02-15", 
                    "sources": ["source4"],
                    "entities": ["公司A"]
                }
            }
            print(f"[DEBUG] _load_events: 已创建 {len(events)} 个模拟事件")
            tools.log(f"⚠️ 事件映射文件不存在: {tools.ABSTRACT_MAP_FILE}")
        
        return events
    
    def _build_graph(self, entities: Dict[str, Dict[str, Any]], events: Dict[str, Dict[str, Any]]) -> None:
        """
        构建知识图谱
        
        Args:
            entities: 实体数据字典
            events: 事件数据字典
        """
        print("[DEBUG] _build_graph: 开始构建知识图谱")
        # 清空现有数据
    
    def build_graph(self, entities: Dict[str, Dict[str, Any]] = None, events: Dict[str, Dict[str, Any]] = None) -> None:
        """
        构建知识图谱（公共接口）
        
        Args:
            entities: 实体数据字典（可选）
            events: 事件数据字典（可选）
        """
        print("[DEBUG] build_graph: 开始构建知识图谱（公共接口）")
        print(f"[DEBUG] build_graph: 构建前状态 - 节点数: {len(self.nodes)}, 关系数: {len(self.relationships)}")
        
        if entities is None or events is None:
            # 如果没有提供数据，从文件加载
            print("[DEBUG] build_graph: 没有提供数据，从文件加载")
            entities = self._load_entities()
            events = self._load_events()
        
        print(f"[DEBUG] build_graph: 数据状态 - 实体数: {len(entities)}, 事件数: {len(events)}")
        
        print("[DEBUG] build_graph: 清空现有数据")
        self.nodes.clear()
        self.relationships.clear()
        self.entity_index.clear()
        self.event_index.clear()
        print(f"[DEBUG] build_graph: 数据清空后 - 节点数: {len(self.nodes)}, 关系数: {len(self.relationships)}")
        
        # 添加实体节点
        print("[DEBUG] build_graph: 开始添加实体节点")
        for i, (entity_name, entity_info) in enumerate(entities.items()):
            entity_id = self._generate_entity_id(entity_name)
            self.entity_index[entity_name] = entity_id
            
            self.nodes[entity_id] = {
                "id": entity_id,
                "type": "entity",
                "name": entity_name,
                "first_seen": entity_info.get("first_seen", ""),
                "sources": entity_info.get("sources", []),
                "original_forms": entity_info.get("original_forms", []),
                "properties": {
                    "entity_type": self._infer_entity_type(entity_name)
                }
            }
            
            if i < 3 or i == len(entities) - 1:
                entity_type = self._infer_entity_type(entity_name)
                print(f"[DEBUG] build_graph: 添加实体 {i+1}/{len(entities)}: {entity_name}, ID: {entity_id}, 类型: {entity_type}")
            
            print(f"[DEBUG] build_graph: 实体节点添加完成，节点数: {len(self.nodes)}, 实体索引大小: {len(self.entity_index)}")
        
        # 添加事件节点和关系
            print("[DEBUG] build_graph: 开始添加事件节点和关系")
            event_count = 0
            relation_count = 0
            
            for i, (event_abstract, event_info) in enumerate(events.items()):
                event_id = self._generate_event_id(event_abstract)
                self.event_index[event_abstract] = event_id
                event_count += 1
                
                # 添加事件节点
                self.nodes[event_id] = {
                    "id": event_id,
                    "type": "event",
                    "abstract": event_abstract,
                    "event_summary": event_info.get("event_summary", ""),
                    "first_seen": event_info.get("first_seen", ""),
                    "sources": event_info.get("sources", []),
                    "properties": {
                        "event_time": event_info.get("first_seen", ""),
                        "source_count": len(event_info.get("sources", [])),
                        "entity_count": len(event_info.get("entities", []))
                    }
                }
                
                if i < 3 or i == len(events) - 1:
                    print(f"[DEBUG] build_graph: 添加事件 {i+1}/{len(events)}: {event_abstract[:30]}..., ID: {event_id}")
                
                # 添加实体-事件关系
                entities_in_event = event_info.get("entities", [])
                print(f"[DEBUG] build_graph: 事件 {event_abstract[:20]}... 包含 {len(entities_in_event)} 个实体")
                
                for entity_name in entities_in_event:
                    if entity_name in self.entity_index:
                        entity_id = self.entity_index[entity_name]
                        
                        # 实体-参与->事件
                        self.relationships.append({
                            "id": f"{entity_id}_participates_in_{event_id}",
                            "source": entity_id,
                            "target": event_id,
                            "type": "participates_in",
                            "properties": {
                                "relation_type": "participation",
                                "confidence": 0.9,
                                "extraction_time": datetime.now().isoformat()
                            }
                        })
                        relation_count += 1
                        
                        # 事件-涉及->实体
                        self.relationships.append({
                            "id": f"{event_id}_involves_{entity_id}",
                            "source": event_id,
                            "target": entity_id,
                            "type": "involves",
                            "properties": {
                                "relation_type": "involvement",
                                "confidence": 0.9,
                                "extraction_time": datetime.now().isoformat()
                            }
                        })
                        relation_count += 1
                        
                        if relation_count <= 5 or relation_count % 10 == 0:
                            print(f"[DEBUG] build_graph: 添加关系 {relation_count}: {entity_name} <-> {event_abstract[:20]}...")
                    else:
                        print(f"[DEBUG-WARNING] build_graph: 实体 '{entity_name}' 在实体索引中不存在")
            
            print(f"[DEBUG] build_graph: 事件节点和关系添加完成")
            print(f"[DEBUG] build_graph: 添加了 {event_count} 个事件节点和 {relation_count} 个关系")
            print(f"[DEBUG] build_graph: 当前状态 - 节点数: {len(self.nodes)}, 关系数: {len(self.relationships)}, 事件索引大小: {len(self.event_index)}")
        
        # 建立实体之间的间接关系（通过共同事件）
        self._build_entity_relationships()
    
    def _build_entity_relationships(self) -> None:
        """
        建立实体之间的间接关系（基于共同参与的事件）
        """
        print("[DEBUG] _build_entity_relationships: 开始构建实体间关系")
        print(f"[DEBUG] _build_entity_relationships: 当前关系数: {len(self.relationships)}")
        
        # 统计实体之间的共同事件数量
        entity_pairs: Dict[tuple, Set[str]] = {}
        
        # 对于每个事件，找出所有参与的实体对
        involves_relationships = [rel for rel in self.relationships if rel["type"] == "involves"]
        print(f"[DEBUG] _build_entity_relationships: 'involves' 关系数量: {len(involves_relationships)}")
        
        processed_events = set()
        for i, relationship in enumerate(involves_relationships):
            event_id = relationship["source"]
            entity_id = relationship["target"]
            
            if event_id in processed_events:
                continue
            
            processed_events.add(event_id)
            print(f"[DEBUG] _build_entity_relationships: 处理事件 {i+1}/{len(involves_relationships)}: {event_id}")
            
            # 找出同一事件中的所有实体
            event_entities = [rel["target"] for rel in involves_relationships if rel["source"] == event_id]
            print(f"[DEBUG] _build_entity_relationships: 事件 {event_id} 包含 {len(event_entities)} 个实体")
            
            # 生成实体对
            for j, entity1 in enumerate(event_entities):
                for entity2 in event_entities[j+1:]:
                    # 确保实体ID排序，避免重复计算
                    pair = tuple(sorted([entity1, entity2]))
                    if pair not in entity_pairs:
                        entity_pairs[pair] = set()
                    entity_pairs[pair].add(event_id)
        
        print(f"[DEBUG] _build_entity_relationships: 找到 {len(entity_pairs)} 对实体关联")
        
        # 为共同事件数大于0的实体对创建关系
        entity_relation_count = 0
        for (entity1_id, entity2_id), common_events in entity_pairs.items():
            common_count = len(common_events)
            
            # print(f"[DEBUG] _build_entity_relationships: 实体对 ({entity1_id}, {entity2_id}) 共享 {common_count} 个事件")
            
            # 计算关系强度
            strength = min(common_count * 0.1, 1.0)
            
            # 创建实体之间的关联关系
            self.relationships.append({
                "id": f"{entity1_id}_related_to_{entity2_id}",
                "source": entity1_id,
                "target": entity2_id,
                "type": "related_to",
                "properties": {
                    "relation_type": "co_occurrence",
                    "common_event_count": common_count,
                    "common_events": list(common_events),
                    "strength": strength,
                    "confidence": min(common_count * 0.2, 1.0),
                    "inference_time": datetime.now().isoformat()
                }
            })
            entity_relation_count += 1
            
            if entity_relation_count <= 5 or entity_relation_count % 10 == 0:
                print(f"[DEBUG] _build_entity_relationships: 创建实体关系 {entity_relation_count}: {entity1_id} -> {entity2_id}")
        
        print(f"[DEBUG] _build_entity_relationships: 实体间关系构建完成，新增 {entity_relation_count} 个关系")
        print(f"[DEBUG] _build_entity_relationships: 当前总关系数: {len(self.relationships)}")
    
    def _generate_entity_id(self, entity_name: str) -> str:
        """生成实体ID"""
        return f"entity_{hash(entity_name) % 1000000:06d}"
    
    def _generate_event_id(self, event_abstract: str) -> str:
        """生成事件ID"""
        return f"event_{hash(event_abstract) % 1000000:06d}"
    
    def _infer_entity_type(self, entity_name: str) -> str:
        """
        推断实体类型
        简单规则：
        - 包含人名特征的为person
        - 包含组织特征的为organization
        - 包含地点特征的为location
        - 其他为entity
        """
        # 简单的实体类型推断规则
        person_keywords = ["主席", "总统", "总理", "省长", "市长", "部长", "议员", "先生", "女士", "博士", "教授"]
        organization_keywords = ["公司", "协会", "委员会", "政府", "部门", "机构", "局", "部", "厅", "处", "法院", "检察院", "银行", "大学", "学院", "医院"]
        location_keywords = ["省", "市", "区", "县", "州", "市辖区", "镇", "乡", "村", "街道", "街道办事处", "路", "街", "道", "广场", "公园", "山", "河", "湖", "海", "洋"]
        
        for keyword in person_keywords:
            if keyword in entity_name:
                return "person"
        
        for keyword in organization_keywords:
            if keyword in entity_name:
                return "organization"
        
        for keyword in location_keywords:
            if keyword in entity_name and len(entity_name) > len(keyword):
                return "location"
        
        # 检查国家名称
        countries = ["中国", "美国", "英国", "法国", "德国", "日本", "韩国", "印度", "俄罗斯", "意大利", "加拿大", "澳大利亚", "巴西", "阿根廷", "墨西哥", "西班牙", "葡萄牙", "荷兰", "比利时", "瑞士"]
        for country in countries:
            if entity_name == country:
                return "country"
        
        return "entity"
    
    def update_graph(self) -> None:
        """更新知识图谱（重新从数据源加载）"""
        tools.log("🔄 更新知识图谱...")
        self.load_from_existing_data()
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        获取实体信息
        
        Args:
            entity_name: 实体名称
            
        Returns:
            实体信息字典，如果不存在返回None
        """
        if entity_name not in self.entity_index:
            return None
        
        entity_id = self.entity_index[entity_name]
        return self.nodes.get(entity_id)
    
    def get_event_info(self, event_abstract: str) -> Optional[Dict[str, Any]]:
        """
        获取事件信息
        
        Args:
            event_abstract: 事件摘要
            
        Returns:
            事件信息字典，如果不存在返回None
        """
        if event_abstract not in self.event_index:
            return None
        
        event_id = self.event_index[event_abstract]
        return self.nodes.get(event_id)
    
    def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        获取实体的所有关系
        
        Args:
            entity_name: 实体名称
            
        Returns:
            关系列表
        """
        if entity_name not in self.entity_index:
            return []
        
        entity_id = self.entity_index[entity_name]
        return [rel for rel in self.relationships if rel["source"] == entity_id or rel["target"] == entity_id]
    
    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取与实体相关的其他实体（基于关系网络）
        
        Args:
            entity_name: 实体名称
            max_depth: 搜索深度
            
        Returns:
            相关实体字典，键为关系类型，值为相关实体列表
        """
        if entity_name not in self.entity_index:
            return {}
        
        entity_id = self.entity_index[entity_name]
        related_entities: Dict[str, List[Dict[str, Any]]] = {}
        visited = set([entity_id])
        
        def dfs(current_id: str, depth: int):
            if depth > max_depth:
                return
            
            for rel in self.relationships:
                if rel["source"] == current_id and rel["target"] not in visited:
                    target_id = rel["target"]
                    visited.add(target_id)
                    
                    if self.nodes[target_id]["type"] == "entity":
                        rel_type = rel["type"]
                        if rel_type not in related_entities:
                            related_entities[rel_type] = []
                        
                        related_entities[rel_type].append({
                            "entity": self.nodes[target_id],
                            "relationship": rel,
                            "depth": depth
                        })
                    
                    dfs(target_id, depth + 1)
        
        dfs(entity_id, 1)
        return related_entities
    
    def get_entity_events(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        获取实体参与的所有事件
        
        Args:
            entity_name: 实体名称
            
        Returns:
            事件列表
        """
        if entity_name not in self.entity_index:
            return []
        
        entity_id = self.entity_index[entity_name]
        events = []
        
        for rel in self.relationships:
            if rel["source"] == entity_id and rel["type"] == "participates_in":
                event_id = rel["target"]
                if event_id in self.nodes:
                    events.append({
                        "event": self.nodes[event_id],
                        "relationship": rel
                    })
        
        # 按时间排序
        events.sort(key=lambda x: x["event"].get("first_seen", ""), reverse=True)
        return events
    
    def get_all_entities(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有实体
        
        Returns:
            实体字典，键为实体名称，值为实体信息
        """
        entities = {}
        for entity_name, entity_id in self.entity_index.items():
            if entity_id in self.nodes:
                entities[entity_name] = self.nodes[entity_id]
        return entities
    
    def get_all_events(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有事件
        
        Returns:
            事件字典，键为事件摘要，值为事件信息
        """
        events = {}
        for event_abstract, event_id in self.event_index.items():
            if event_id in self.nodes:
                events[event_abstract] = self.nodes[event_id]
        return events
    
    def search_entities(self, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索实体
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            匹配的实体列表
        """
        results = []
        for entity_name, entity_id in self.entity_index.items():
            if keyword in entity_name and entity_id in self.nodes:
                results.append({
                    "name": entity_name,
                    "info": self.nodes[entity_id]
                })
        return results
    
    def search_events(self, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索事件
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            匹配的事件列表
        """
        results = []
        for event_abstract, event_id in self.event_index.items():
            if keyword in event_abstract and event_id in self.nodes:
                results.append({
                    "abstract": event_abstract,
                    "info": self.nodes[event_id]
                })
        return results
    
    def get_events_by_entity_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        获取包含特定类型实体的事件
        
        Args:
            entity_type: 实体类型
            
        Returns:
            事件列表
        """
        events = []
        # 先找出所有该类型的实体
        target_entities = []
        for entity_id, node in self.nodes.items():
            if node["type"] == "entity" and node["properties"].get("entity_type") == entity_type:
                target_entities.append(entity_id)
        
        # 找出这些实体参与的事件
        event_ids = set()
        for rel in self.relationships:
            if rel["source"] in target_entities and rel["type"] == "participates_in":
                event_ids.add(rel["target"])
        
        # 获取事件信息
        for event_id in event_ids:
            if event_id in self.nodes:
                events.append(self.nodes[event_id])
        
        return events
    
    def get_relationship_details(self, source_id: str, target_id: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        """
        获取两个节点之间的关系详情
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_type: 关系类型（可选）
            
        Returns:
            关系列表
        """
        relationships = []
        for rel in self.relationships:
            if rel["source"] == source_id and rel["target"] == target_id:
                if relationship_type is None or rel["type"] == relationship_type:
                    relationships.append(rel)
        return relationships
    
    def get_entity_event_relations(self) -> List[Dict[str, Any]]:
        """
        获取实体与事件之间的关系
        
        Returns:
            实体与事件关系列表
        """
        entity_event_rels = []
        for rel in self.relationships:
            if rel.get('type') == 'participates_in':
                entity_event_rels.append(rel)
        return entity_event_rels
    
    def get_entity_relations(self) -> List[Dict[str, Any]]:
        """
        获取实体之间的关系
        
        Returns:
            实体间关系列表
        """
        entity_rels = []
        for rel in self.relationships:
            if rel.get('type') == 'related_to':
                entity_rels.append(rel)
        return entity_rels
    
    def save_graph(self, output_path: Optional[Path] = None) -> None:
        """
        保存知识图谱到文件
        
        Args:
            output_path: 输出文件路径，默认保存到data目录
        """
        print("[DEBUG] save_graph: 开始保存知识图谱")
        if output_path is None:
            output_path = tools.DATA_DIR / "knowledge_graph.json"
        
        print(f"[DEBUG] save_graph: 保存路径: {output_path}")
        print(f"[DEBUG] save_graph: 保存前数据状态 - 节点数: {len(self.nodes)}, 关系数: {len(self.relationships)}")
        print(f"[DEBUG] save_graph: 实体索引大小: {len(self.entity_index)}, 事件索引大小: {len(self.event_index)}")
        
        graph_data = {
            "metadata": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "node_count": len(self.nodes),
                "relationship_count": len(self.relationships),
                "entity_count": len(self.entity_index),
                "event_count": len(self.event_index)
            },
            "nodes": list(self.nodes.values()),
            "relationships": self.relationships,
            "entity_index": self.entity_index,
            "event_index": self.event_index
        }
        
        # 调试输出将要保存的数据大小
        print(f"[DEBUG] save_graph: 节点数据大小: {len(graph_data['nodes'])}")
        print(f"[DEBUG] save_graph: 关系数据大小: {len(graph_data['relationships'])}")
        print(f"[DEBUG] save_graph: 元数据 - 节点数: {graph_data['metadata']['node_count']}, 关系数: {graph_data['metadata']['relationship_count']}")
        
        try:
            # 确保目录存在
            print(f"[DEBUG] save_graph: 确保输出目录存在")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[DEBUG] save_graph: 开始写入文件")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            # 验证文件是否保存成功
            if output_path.exists():
                file_size = output_path.stat().st_size
                print(f"[DEBUG] save_graph: 知识图谱保存成功，文件大小: {file_size} 字节")
                tools.log(f"✅ 知识图谱已保存到: {output_path}")
            else:
                print(f"[DEBUG-ERROR] save_graph: 文件保存失败，文件不存在")
        except Exception as e:
            print(f"[DEBUG-ERROR] save_graph: 保存知识图谱失败: {e}")
            import traceback
            print(f"[DEBUG-ERROR] save_graph: 错误堆栈: {traceback.format_exc()}")
            tools.log(f"⚠️ 保存知识图谱失败: {e}")
    
    def load_graph(self, input_path: Optional[Path] = None) -> bool:
        """
        从文件加载知识图谱
        
        Args:
            input_path: 输入文件路径，默认从data目录加载
            
        Returns:
            是否加载成功
        """
        print("[DEBUG] load_graph: 开始加载知识图谱")
        if input_path is None:
            input_path = tools.DATA_DIR / "knowledge_graph.json"
        
        print(f"[DEBUG] load_graph: 加载路径: {input_path}")
        
        if not input_path.exists():
            print(f"[DEBUG-WARNING] load_graph: 知识图谱文件不存在: {input_path}")
            tools.log(f"⚠️ 知识图谱文件不存在: {input_path}")
            return False
        
        try:
            print(f"[DEBUG] load_graph: 文件存在，开始读取")
            file_size = input_path.stat().st_size
            print(f"[DEBUG] load_graph: 文件大小: {file_size} 字节")
            
            with open(input_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            
            print(f"[DEBUG] load_graph: 文件读取成功，开始恢复数据")
            print(f"[DEBUG] load_graph: 元数据 - 节点数: {graph_data['metadata']['node_count']}, 关系数: {graph_data['metadata']['relationship_count']}")
            print(f"[DEBUG] load_graph: 实体索引大小: {len(graph_data['entity_index']) if 'entity_index' in graph_data else 0}")
            print(f"[DEBUG] load_graph: 事件索引大小: {len(graph_data['event_index']) if 'event_index' in graph_data else 0}")
            
            # 恢复节点
            if 'nodes' in graph_data:
                self.nodes = {node["id"]: node for node in graph_data["nodes"]}
                print(f"[DEBUG] load_graph: 恢复节点完成，节点数: {len(self.nodes)}")
            else:
                print(f"[DEBUG-WARNING] load_graph: 数据中没有nodes字段")
                self.nodes = {}
            
            # 恢复关系
            if 'relationships' in graph_data:
                self.relationships = graph_data["relationships"]
                print(f"[DEBUG] load_graph: 恢复关系完成，关系数: {len(self.relationships)}")
            else:
                print(f"[DEBUG-WARNING] load_graph: 数据中没有relationships字段")
                self.relationships = []
            
            # 恢复索引
            if 'entity_index' in graph_data:
                self.entity_index = graph_data["entity_index"]
                print(f"[DEBUG] load_graph: 恢复实体索引完成，大小: {len(self.entity_index)}")
            else:
                print(f"[DEBUG-WARNING] load_graph: 数据中没有entity_index字段")
                self.entity_index = {}
            
            if 'event_index' in graph_data:
                self.event_index = graph_data["event_index"]
                print(f"[DEBUG] load_graph: 恢复事件索引完成，大小: {len(self.event_index)}")
            else:
                print(f"[DEBUG-WARNING] load_graph: 数据中没有event_index字段")
                self.event_index = {}
            
            # 更新最后更新时间
            self.last_updated = datetime.now().isoformat()
            
            print(f"[DEBUG] load_graph: 数据恢复完成")
            print(f"[DEBUG] load_graph: 加载后状态 - 节点数: {len(self.nodes)}, 关系数: {len(self.relationships)}")
            
            # 验证加载的数据是否为空
            if len(self.nodes) == 0 and len(self.relationships) == 0:
                print(f"[DEBUG-WARNING] load_graph: 加载的数据为空")
            
            tools.log(f"✅ 知识图谱已从 {input_path} 加载: {len(self.nodes)}个节点, {len(self.relationships)}个关系")
            return True
        except json.JSONDecodeError as e:
            print(f"[DEBUG-ERROR] load_graph: JSON解析错误: {e}")
            print(f"[DEBUG-ERROR] load_graph: 文件内容可能损坏或格式错误")
            tools.log(f"⚠️ 加载知识图谱失败: JSON格式错误")
            return False
        except Exception as e:
            print(f"[DEBUG-ERROR] load_graph: 加载知识图谱失败: {e}")
            import traceback
            print(f"[DEBUG-ERROR] load_graph: 错误堆栈: {traceback.format_exc()}")
            tools.log(f"⚠️ 加载知识图谱失败: {e}")
            return False
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取知识图谱统计信息
        
        Returns:
            统计信息字典
        """
        entity_count = 0
        event_count = 0
        entity_types = {}
        
        for node in self.nodes.values():
            if node["type"] == "entity":
                entity_count += 1
                entity_type = node["properties"].get("entity_type", "entity")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            elif node["type"] == "event":
                event_count += 1
        
        relationship_types = {}
        for rel in self.relationships:
            rel_type = rel["type"]
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.relationships),
            "entity_count": entity_count,
            "event_count": event_count,
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "last_updated": self.last_updated
        }

# 全局知识图谱实例
KG_INSTANCE = None

def get_knowledge_graph() -> KnowledgeGraph:
    """
    获取知识图谱单例
    
    Returns:
        KnowledgeGraph实例
    """
    global KG_INSTANCE
    if KG_INSTANCE is None:
        KG_INSTANCE = KnowledgeGraph()
    return KG_INSTANCE

def build_knowledge_graph():  
    """
    构建知识图谱的主函数
    """
    print("[DEBUG] build_knowledge_graph: 启动知识图谱构建主函数")
    tools.log("🚀 启动知识图谱构建...")
    
    # 获取或创建知识图谱实例
    print("[DEBUG] build_knowledge_graph: 获取或创建知识图谱实例")
    kg = get_knowledge_graph()
    
    # 尝试从文件加载，如果失败则重新构建
    print("[DEBUG] build_knowledge_graph: 尝试从文件加载知识图谱")
    load_success = kg.load_graph()
    print(f"[DEBUG] build_knowledge_graph: 加载结果: {'成功' if load_success else '失败'}")
    
    if not load_success or len(kg.nodes) == 0:
        print("[DEBUG] build_knowledge_graph: 文件加载失败或数据为空，开始重新构建")
        kg.load_from_existing_data()
    else:
        print(f"[DEBUG] build_knowledge_graph: 从文件加载成功，数据状态 - 节点数: {len(kg.nodes)}, 关系数: {len(kg.relationships)}")
    
    # 保存知识图谱
    print("[DEBUG] build_knowledge_graph: 保存知识图谱")
    kg.save_graph()
    
    # 输出统计信息
    print("[DEBUG] build_knowledge_graph: 生成统计信息")
    stats = kg.get_graph_statistics()
    print(f"[DEBUG] build_knowledge_graph: 统计信息 - 总节点数: {stats['total_nodes']}, 总关系数: {stats['total_relationships']}")
    print(f"[DEBUG] build_knowledge_graph: 统计信息 - 实体数: {stats['entity_count']}, 事件数: {stats['event_count']}")
    
    tools.log(f"📊 知识图谱统计信息:")
    tools.log(f"   总节点数: {stats['total_nodes']}")
    tools.log(f"   总关系数: {stats['total_relationships']}")
    tools.log(f"   实体数: {stats['entity_count']}")
    tools.log(f"   事件数: {stats['event_count']}")
    tools.log(f"   实体类型分布: {stats['entity_types']}")
    tools.log(f"   关系类型分布: {stats['relationship_types']}")
    
    print("[DEBUG] build_knowledge_graph: 知识图谱构建流程完成")
    tools.log("🎉 知识图谱构建完成！")
    return kg

if __name__ == "__main__":
    build_knowledge_graph()
