# src/agents/kg_interface.py
"""
知识图谱接口函数模块

提供统一的知识图谱查询和操作API，作为其他模块与知识图谱交互的桥梁
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

from ..utils.tool_function import tools
from .knowledge_graph import get_knowledge_graph, build_knowledge_graph

# 初始化工具
tools = tools()

class KnowledgeGraphInterface:
    """
    知识图谱接口类
    
    提供高层API封装，简化知识图谱的使用
    """
    
    def __init__(self):
        """初始化接口类"""
        tools.log(f"🔍 [DEBUG] 知识图谱接口初始化开始...")
        try:
            self.kg = get_knowledge_graph()
            tools.log(f"🔍 [DEBUG] 成功获取知识图谱实例，当前节点数: {len(self.kg.nodes)}")
            self._ensure_graph_loaded()
            tools.log(f"🔍 [DEBUG] 知识图谱接口初始化完成")
        except Exception as e:
            tools.log(f"❌ [DEBUG] 知识图谱接口初始化失败: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
    
    def _ensure_graph_loaded(self) -> None:
        """确保知识图谱已加载"""
        tools.log(f"🔍 [DEBUG] 检查知识图谱加载状态，当前节点数: {len(self.kg.nodes)}")
        if len(self.kg.nodes) == 0:
            tools.log("🔄 知识图谱未加载，尝试构建...")
            try:
                tools.log(f"🔍 [DEBUG] 开始构建知识图谱...")
                build_knowledge_graph()
                tools.log(f"✅ [DEBUG] 知识图谱构建完成，更新后节点数: {len(self.kg.nodes)}")
            except Exception as e:
                tools.log(f"❌ 构建知识图谱失败: {e}")
                import traceback
                tools.log(f"❌ [DEBUG] 构建失败错误堆栈: {traceback.format_exc()}")
    
    def search_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        搜索实体信息
        
        Args:
            entity_name: 实体名称
            
        Returns:
            实体详情字典，如果不存在返回None
        """
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 搜索实体: '{entity_name}'")
        
        try:
            entity_info = self.kg.get_entity_info(entity_name)
            processing_time = (time.time() - start_time) * 1000
            
            if entity_info:
                result = {
                    "id": entity_info["id"],
                    "name": entity_info["name"],
                    "type": entity_info["properties"].get("entity_type", "entity"),
                    "first_seen": entity_info["first_seen"],
                    "sources": entity_info["sources"],
                    "original_forms": entity_info["original_forms"]
                }
                tools.log(f"✅ [DEBUG] 找到实体: '{entity_name}'，耗时: {processing_time:.2f}ms")
                return result
            else:
                tools.log(f"⚠️ [DEBUG] 未找到实体: '{entity_name}'，耗时: {processing_time:.2f}ms")
                return None
        except Exception as e:
            tools.log(f"❌ [DEBUG] 搜索实体出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return None
    
    def search_event(self, event_abstract: str) -> Optional[Dict[str, Any]]:
        """
        搜索事件信息
        
        Args:
            event_abstract: 事件摘要
            
        Returns:
            事件详情字典，如果不存在返回None
        """
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 搜索事件，摘要开头: '{event_abstract[:50]}...'")
        
        try:
            event_info = self.kg.get_event_info(event_abstract)
            processing_time = (time.time() - start_time) * 1000
            
            if event_info:
                result = {
                    "id": event_info["id"],
                    "abstract": event_info["abstract"],
                    "summary": event_info["event_summary"],
                    "time": event_info["first_seen"],
                    "sources": event_info["sources"],
                    "properties": event_info["properties"]
                }
                tools.log(f"✅ [DEBUG] 找到事件，耗时: {processing_time:.2f}ms，事件ID: {event_info['id']}")
                return result
            else:
                tools.log(f"⚠️ [DEBUG] 未找到事件，耗时: {processing_time:.2f}ms")
                return None
        except Exception as e:
            tools.log(f"❌ [DEBUG] 搜索事件出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return None
    
    def get_entity_events(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取实体参与的事件列表
        
        Args:
            entity_name: 实体名称
            limit: 返回事件数量限制
            
        Returns:
            事件列表
        """
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 获取实体事件列表，实体: '{entity_name}'，限制: {limit}")
        
        try:
            # 检查实体是否存在
            if entity_name not in self.kg.entity_index:
                tools.log(f"⚠️ [DEBUG] 实体 '{entity_name}' 不存在于索引中")
                return []
                
            events_data = self.kg.get_entity_events(entity_name)
            tools.log(f"🔍 [DEBUG] 从知识图谱获取到 {len(events_data)} 个事件记录")
            
            result = []
            
            for i, event_item in enumerate(events_data):
                if i >= limit:
                    break
                
                event = event_item["event"]
                result.append({
                    "id": event["id"],
                    "abstract": event["abstract"],
                    "summary": event["event_summary"],
                    "time": event["first_seen"],
                    "source_count": len(event["sources"]),
                    "entity_count": event["properties"].get("entity_count", 0)
                })
            
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"✅ [DEBUG] 获取实体事件完成，返回 {len(result)} 个事件，耗时: {processing_time:.2f}ms")
            return result
        except Exception as e:
            tools.log(f"❌ [DEBUG] 获取实体事件出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return []
    
    def get_related_entities(self, entity_name: str, max_depth: int = 1, min_strength: float = 0.1) -> List[Dict[str, Any]]:
        """
        获取与指定实体相关的其他实体
        
        Args:
            entity_name: 实体名称
            max_depth: 关系搜索深度
            min_strength: 最小关系强度
            
        Returns:
            相关实体列表
        """
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 获取相关实体，实体: '{entity_name}'，深度: {max_depth}，最小强度: {min_strength}")
        
        try:
            # 检查实体是否存在
            if entity_name not in self.kg.entity_index:
                tools.log(f"⚠️ [DEBUG] 实体 '{entity_name}' 不存在于索引中")
                return []
                
            related_data = self.kg.get_related_entities(entity_name, max_depth)
            tools.log(f"🔍 [DEBUG] 从知识图谱获取到 {len(related_data)} 种关系类型")
            
            related_entities = []
            
            for rel_type, entities_list in related_data.items():
                tools.log(f"🔍 [DEBUG] 处理关系类型: '{rel_type}'，实体列表长度: {len(entities_list)}")
                for item in entities_list:
                    relationship = item["relationship"]
                    strength = relationship["properties"].get("strength", 1.0)
                    
                    if strength >= min_strength:
                        related_entities.append({
                            "entity": {
                                "id": item["entity"]["id"],
                                "name": item["entity"]["name"],
                                "type": item["entity"]["properties"].get("entity_type", "entity")
                            },
                            "relationship": {
                                "type": rel_type,
                                "strength": strength,
                                "common_events": relationship["properties"].get("common_events", []),
                                "common_event_count": relationship["properties"].get("common_event_count", 0)
                            },
                            "depth": item["depth"]
                        })
            
            # 按关系强度排序
            related_entities.sort(key=lambda x: x["relationship"]["strength"], reverse=True)
            
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"✅ [DEBUG] 获取相关实体完成，返回 {len(related_entities)} 个相关实体，耗时: {processing_time:.2f}ms")
            return related_entities
        except Exception as e:
            tools.log(f"❌ [DEBUG] 获取相关实体出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return []
    
    def get_entity_path(self, start_entity: str, end_entity: str, max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """
        查找两个实体之间的关系路径
        
        Args:
            start_entity: 起始实体名称
            end_entity: 目标实体名称
            max_depth: 最大搜索深度
            
        Returns:
            关系路径列表，如果不存在路径返回None
        """
        # 检查实体是否存在
        if start_entity not in self.kg.entity_index or end_entity not in self.kg.entity_index:
            return None
        
        start_id = self.kg.entity_index[start_entity]
        end_id = self.kg.entity_index[end_entity]
        
        # 简单的BFS搜索
        from collections import deque
        
        visited = {start_id: None}
        queue = deque([(start_id, 0)])
        found = False
        
        while queue and not found:
            current_id, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for rel in self.kg.relationships:
                if rel["source"] == current_id and rel["target"] not in visited:
                    target_id = rel["target"]
                    visited[target_id] = (current_id, rel)
                    
                    if target_id == end_id:
                        found = True
                        break
                    
                    # 只搜索实体节点
                    if self.kg.nodes[target_id]["type"] == "entity":
                        queue.append((target_id, depth + 1))
            
            if found:
                break
        
        # 重构路径
        if not found:
            return None
        
        path = []
        current = end_id
        
        while visited[current]:
            prev_id, relationship = visited[current]
            path.append({
                "from": self.kg.nodes[prev_id],
                "to": self.kg.nodes[current],
                "relationship": relationship
            })
            current = prev_id
        
        # 反转路径，使其从起始实体到目标实体
        path.reverse()
        return path
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """
        获取知识图谱摘要信息
        
        Returns:
            知识图谱摘要统计
        """
        stats = self.kg.get_graph_statistics()
        
        return {
            "total_nodes": stats["total_nodes"],
            "total_relationships": stats["total_relationships"],
            "entity_count": stats["entity_count"],
            "event_count": stats["event_count"],
            "entity_types": stats["entity_types"],
            "relationship_types": stats["relationship_types"],
            "last_updated": stats["last_updated"]
        }
    
    def refresh_graph(self, force: bool = False) -> bool:
        """
        刷新知识图谱（重新构建）
        
        Args:
            force: 是否强制刷新
            
        Returns:
            是否刷新成功
        """
        try:
            build_knowledge_graph()
            self.kg = get_knowledge_graph()  # 重新获取实例
            return True
        except Exception as e:
            tools.log(f"❌ 刷新知识图谱失败: {e}")
            return False
    
    def export_graph_data(self, output_path: Optional[str] = None) -> bool:
        """
        导出知识图谱数据
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            path = Path(output_path) if output_path else tools.DATA_DIR / "kg_export.json"
            self.kg.save_graph(path)
            return True
        except Exception as e:
            tools.log(f"❌ 导出知识图谱失败: {e}")
            return False
    
    def search_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        按类型搜索实体
        
        Args:
            entity_type: 实体类型
            
        Returns:
            实体列表
        """
        entities = []
        
        for node in self.kg.nodes.values():
            if node["type"] == "entity" and node["properties"].get("entity_type") == entity_type:
                entities.append({
                    "id": node["id"],
                    "name": node["name"],
                    "first_seen": node["first_seen"],
                    "sources": node["sources"]
                })
        
        return entities
    
    def get_events_by_time_range(self, start_time: str = "", end_time: str = "") -> List[Dict[str, Any]]:
        """
        按时间范围获取事件
        
        Args:
            start_time: 开始时间（ISO格式字符串）
            end_time: 结束时间（ISO格式字符串）
            
        Returns:
            事件列表
        """
        events = []
        
        for node in self.kg.nodes.values():
            if node["type"] == "event":
                event_time = node["first_seen"]
                
                if start_time and event_time < start_time:
                    continue
                if end_time and event_time > end_time:
                    continue
                
                events.append({
                    "id": node["id"],
                    "abstract": node["abstract"],
                    "summary": node["event_summary"],
                    "time": event_time,
                    "entity_count": node["properties"].get("entity_count", 0)
                })
        
        # 按时间排序
        events.sort(key=lambda x: x["time"], reverse=True)
        return events
    
    def get_top_related_entities(self, entity_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        获取与指定实体最相关的实体（按共同事件数排序）
        
        Args:
            entity_name: 实体名称
            top_n: 返回数量
            
        Returns:
            最相关实体列表
        """
        related_entities = self.get_related_entities(entity_name, max_depth=2)
        
        # 按共同事件数排序
        related_entities.sort(key=lambda x: x["relationship"].get("common_event_count", 0), reverse=True)
        
        # 截取前top_n个
        return related_entities[:top_n]
    
    def get_knowledge_graph_insights(self, entity_name: str) -> Dict[str, Any]:
        """
        获取关于实体的知识图谱洞察
        
        Args:
            entity_name: 实体名称
            
        Returns:
            洞察结果字典
        """
        if entity_name not in self.kg.entity_index:
            return {"error": "实体不存在"}
        
        # 获取实体基本信息
        entity_info = self.search_entity(entity_name)
        
        # 获取实体参与的事件
        events = self.get_entity_events(entity_name, limit=5)
        
        # 获取相关实体
        related_entities = self.get_top_related_entities(entity_name, top_n=5)
        
        # 计算参与事件统计
        event_stats = {
            "total_events": len(self.get_entity_events(entity_name, limit=1000)),
            "recent_events": len(events),
            "event_categories": self._categorize_events(events)
        }
        
        # 计算相关实体统计
        entity_stats = {
            "related_entity_count": len(self.get_related_entities(entity_name)),
            "top_related_entities": related_entities,
            "entity_type_distribution": self._calculate_related_type_distribution(related_entities)
        }
        
        return {
            "entity": entity_info,
            "event_statistics": event_stats,
            "relationship_statistics": entity_stats,
            "insights": self._generate_entity_insights(entity_info, event_stats, entity_stats)
        }
    
    def _categorize_events(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        对事件进行简单分类
        
        Args:
            events: 事件列表
            
        Returns:
            分类统计
        """
        categories = {
            "政治": 0,
            "经济": 0,
            "社会": 0,
            "其他": 0
        }
        
        # 简单的关键词分类
        for event in events:
            summary = event.get("summary", "") + " " + event.get("abstract", "")
            
            if any(keyword in summary for keyword in ["政府", "政策", "选举", "官员", "会议", "法案"]):
                categories["政治"] += 1
            elif any(keyword in summary for keyword in ["经济", "投资", "市场", "价格", "贸易", "企业"]):
                categories["经济"] += 1
            elif any(keyword in summary for keyword in ["社会", "民生", "教育", "医疗", "环境", "科技"]):
                categories["社会"] += 1
            else:
                categories["其他"] += 1
        
        return categories
    
    def _calculate_related_type_distribution(self, related_entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        计算相关实体类型分布
        
        Args:
            related_entities: 相关实体列表
            
        Returns:
            类型分布统计
        """
        distribution = {}
        
        for item in related_entities:
            entity_type = item["entity"].get("type", "entity")
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        
        return distribution
    
    def _generate_entity_insights(self, entity_info: Dict[str, Any], 
                               event_stats: Dict[str, Any], 
                               relationship_stats: Dict[str, Any]) -> List[str]:
        """
        生成实体洞察
        
        Args:
            entity_info: 实体信息
            event_stats: 事件统计
            relationship_stats: 关系统计
            
        Returns:
            洞察列表
        """
        insights = []
        
        # 根据统计信息生成洞察
        if event_stats["total_events"] > 10:
            insights.append(f"{entity_info['name']}在知识图谱中参与了{event_stats['total_events']}个事件，活跃度较高。")
        
        # 分析事件类别
        categories = event_stats["event_categories"]
        max_category = max(categories, key=categories.get)
        if categories[max_category] / sum(categories.values()) > 0.5:
            insights.append(f"{entity_info['name']}的活动主要集中在{max_category}领域。")
        
        # 分析相关实体
        if relationship_stats["related_entity_count"] > 20:
            insights.append(f"{entity_info['name']}与{relationship_stats['related_entity_count']}个其他实体存在关联，网络影响力较大。")
        
        # 分析实体类型
        if entity_info["type"] == "person":
            insights.append(f"{entity_info['name']}是一个人物实体，可能在多个事件中扮演重要角色。")
        elif entity_info["type"] == "organization":
            insights.append(f"{entity_info['name']}是一个组织实体，可能涉及多方面的活动和关系。")
        
        return insights

# 全局接口实例
KG_INTERFACE = None

def get_kg_interface() -> KnowledgeGraphInterface:
    """
    获取知识图谱接口单例
    
    Returns:
        KnowledgeGraphInterface实例
    """
    global KG_INTERFACE
    tools.log(f"🔍 [DEBUG] 获取知识图谱接口单例，当前状态: {'已存在' if KG_INTERFACE else '不存在'}")
    if KG_INTERFACE is None:
        tools.log(f"🔍 [DEBUG] 创建新的知识图谱接口实例")
        KG_INTERFACE = KnowledgeGraphInterface()
        tools.log(f"✅ [DEBUG] 知识图谱接口实例创建完成")
    else:
        tools.log(f"✅ [DEBUG] 复用已存在的知识图谱接口实例")
    return KG_INTERFACE

# 便捷函数
def search_entity(entity_name: str) -> Optional[Dict[str, Any]]:
    """搜索实体信息的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 search_entity: '{entity_name}'")
    return get_kg_interface().search_entity(entity_name)

def get_entity_events(entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """获取实体事件的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 get_entity_events: '{entity_name}'，限制: {limit}")
    return get_kg_interface().get_entity_events(entity_name, limit)

def get_related_entities(entity_name: str, max_depth: int = 1) -> List[Dict[str, Any]]:
    """获取相关实体的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 get_related_entities: '{entity_name}'，深度: {max_depth}")
    return get_kg_interface().get_related_entities(entity_name, max_depth)

def get_graph_summary() -> Dict[str, Any]:
    """获取知识图谱摘要的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 get_graph_summary")
    return get_kg_interface().get_graph_summary()

def refresh_graph(force: bool = False) -> bool:
    """刷新知识图谱的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 refresh_graph，强制刷新: {force}")
    return get_kg_interface().refresh_graph(force)

def get_entity_insights(entity_name: str) -> Dict[str, Any]:
    """获取实体洞察的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 get_entity_insights: '{entity_name}'")
    return get_kg_interface().get_knowledge_graph_insights(entity_name)

def search_entities(keyword: str) -> List[Dict[str, Any]]:
    """搜索实体的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 search_entities: '{keyword}'")
    return get_kg_interface().kg.search_entities(keyword)

def search_events(keyword: str) -> List[Dict[str, Any]]:
    """搜索事件的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 search_events: '{keyword}'")
    return get_kg_interface().kg.search_events(keyword)

def get_entity_relations(entity_name: str) -> List[Dict[str, Any]]:
    """获取实体关系的便捷函数"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 get_entity_relations: '{entity_name}'")
    return get_kg_interface().kg.get_entity_relationships(entity_name)

if __name__ == "__main__":
    # 测试接口功能
    interface = get_kg_interface()
    print("知识图谱摘要:")
    print(json.dumps(get_graph_summary(), ensure_ascii=False, indent=2))
