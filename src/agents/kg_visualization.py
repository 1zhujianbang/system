# src/agents/kg_visualization.py
"""
知识图谱可解释性展示组件

该组件提供：
1. 实体关系可视化功能
2. 交互式分析界面
3. 图谱解释报告生成
4. 异常关系检测与解释
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import io
import base64
import warnings
warnings.filterwarnings("ignore")

from ..utils.tool_function import tools
from .kg_interface import get_knowledge_graph

# 全局知识图谱实例
KG = None

# 确保KG实例被正确初始化
def _ensure_kg():
    global KG
    tools.log(f"🔍 [DEBUG] 确保知识图谱实例，当前状态: {'已存在' if KG else '不存在'}")
    if KG is None:
        tools.log(f"🔍 [DEBUG] 创建新的知识图谱实例")
        KG = get_knowledge_graph()
        tools.log(f"✅ [DEBUG] 知识图谱实例创建完成，节点数: {len(KG.nodes) if KG else 0}")
    return KG

# 初始化KG实例
KG = _ensure_kg()

class KGVisualizer:
    """知识图谱可视化器"""
    
    def __init__(self):
        tools.log(f"🔍 [DEBUG] KGVisualizer初始化开始")
        self.reset()
        tools.log(f"✅ [DEBUG] KGVisualizer初始化完成")
        
    def reset(self):
        """重置可视化器状态"""
        tools.log(f"🔍 [DEBUG] 重置KGVisualizer状态")
        self.graph = nx.Graph()
        self.node_colors = {}
        self.node_sizes = {}
        self.edge_weights = {}
        self.edge_colors = {}
        self.entity_to_events = {}
    
    def build_visualization_graph(self, entities: List[str] = None, depth: int = 2, 
                                 include_events: bool = True) -> nx.Graph:
        """
        构建可视化图谱
        
        Args:
            entities: 起始实体列表，为空时展示全部图谱
            depth: 关系深度
            include_events: 是否包含事件节点
        
        Returns:
            构建好的NetworkX图
        """
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 开始构建可视化图谱，实体列表: {entities}, 深度: {depth}, 包含事件: {include_events}")
        
        try:
            _ensure_kg()
            
            self.reset()
            
            # 获取基础图谱数据
            tools.log(f"🔍 [DEBUG] 获取基础图谱数据...")
            all_entities = KG.get_all_entities()
            all_events = KG.get_all_events()
            entity_entity_relations = KG.get_entity_relations()
            entity_event_relations = KG.get_entity_event_relations()
            
            tools.log(f"📊 [DEBUG] 图谱基础数据统计 - 实体数: {len(all_entities)}, 事件数: {len(all_events)}")
        
            # 如果指定了实体，使用广度优先搜索构建子图
            if entities:
                tools.log(f"🔍 [DEBUG] 使用指定实体构建子图，起始实体数: {len(entities)}")
                visited_entities = set()
                queue = [(entity, 0) for entity in entities]
            
                while queue:
                    current_entity, current_depth = queue.pop(0)
                    if current_entity in visited_entities or current_depth > depth:
                        continue
                    
                    visited_entities.add(current_entity)
                    tools.log(f"🔍 [DEBUG] 处理实体: '{current_entity}'，当前深度: {current_depth}")
                
                    # 添加实体节点
                    self.graph.add_node(current_entity, type="entity")
                    self.node_colors[current_entity] = "#3498db"  # 蓝色表示实体
                    
                    # 计算实体参与的事件数量
                    entity_info = KG.get_entity_info(current_entity)
                    entity_id = entity_info["id"] if entity_info else None
                    event_count = 0
                    if entity_id:
                        for rel in KG.relationships:
                            if rel["source"] == entity_id and rel["type"] == "participates_in":
                                event_count += 1
                    self.node_sizes[current_entity] = 1000 + min(event_count * 100, 3000)
                
                    # 添加直接相关的事件
                    if include_events:
                        tools.log(f"🔍 [DEBUG] 获取实体 '{current_entity}' 参与的事件")
                        related_events = KG.get_entity_events(current_entity)
                        tools.log(f"📊 [DEBUG] 实体 '{current_entity}' 参与了 {len(related_events)} 个事件")
                        self.entity_to_events[current_entity] = related_events
                        
                        for event in related_events:
                            if event in all_events:
                                self.graph.add_node(event, type="event")
                                self.node_colors[event] = "#e74c3c"  # 红色表示事件
                                self.node_sizes[event] = 800
                                
                                # 添加实体-事件边
                                edge_key = (current_entity, event)
                                self.graph.add_edge(*edge_key)
                                self.edge_colors[edge_key] = "#95a5a6"
                                self.edge_weights[edge_key] = 1
                    
                    # 添加相关实体
                    if current_depth < depth:
                        tools.log(f"🔍 [DEBUG] 获取实体 '{current_entity}' 的相关实体")
                        related_entities = KG.get_related_entities(current_entity)
                        tools.log(f"📊 [DEBUG] 实体 '{current_entity}' 相关实体数: {len(related_entities)}")
                        for related_entity in related_entities:
                            if related_entity in all_entities:
                                # 添加实体-实体边
                                edge_key = tuple(sorted([current_entity, related_entity]))
                                self.graph.add_edge(*edge_key)
                                self.edge_colors[edge_key] = "#2ecc71"
                                self.edge_weights[edge_key] = 1.5
                                
                                if related_entity not in visited_entities:
                                    queue.append((related_entity, current_depth + 1))
            else:
                # 展示全部图谱（限制数量以防性能问题）
                tools.log(f"🔍 [DEBUG] 构建全局图谱，限制节点数: 100")
                max_nodes = 100
                # 确保正确处理实体列表
                sampled_entities = list(all_entities)[:max_nodes]
                tools.log(f"📊 [DEBUG] 采样实体数: {len(sampled_entities)}")
            
                for entity in sampled_entities:
                    self.graph.add_node(entity, type="entity")
                    self.node_colors[entity] = "#3498db"
                    self.node_sizes[entity] = 1000
        
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"✅ [DEBUG] 可视化图谱构建完成，节点数: {len(self.graph.nodes())}, 边数: {len(self.graph.edges())}, 耗时: {processing_time:.2f}ms")
            return self.graph
        except Exception as e:
            tools.log(f"❌ [DEBUG] 构建可视化图谱出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return self.graph
    
    def generate_plot_image(self, output_format: str = "base64") -> Union[str, plt.Figure]:
        """
        生成图谱可视化图像
        
        Args:
            output_format: 输出格式，"base64"或"figure"
        
        Returns:
            图像的base64编码字符串或matplotlib Figure对象
        """
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 生成可视化图像，输出格式: {output_format}")
        
        try:
            if len(self.graph.nodes()) == 0:
                tools.log(f"❌ [DEBUG] 图谱为空，无法生成可视化图像")
                raise ValueError("图谱为空，请先构建可视化图谱")
            
            tools.log(f"📊 [DEBUG] 当前图谱统计 - 节点数: {len(self.graph.nodes())}, 边数: {len(self.graph.edges())}")
        
            plt.figure(figsize=(16, 12))
        
            # 使用spring布局
            pos = nx.spring_layout(self.graph, seed=42, k=0.15)
            
            # 绘制节点
            node_list = list(self.graph.nodes())
            node_color_values = [self.node_colors.get(node, "#95a5a6") for node in node_list]
            node_size_values = [self.node_sizes.get(node, 500) for node in node_list]
            
            nx.draw_networkx_nodes(
                self.graph, pos, node_color=node_color_values, 
                node_size=node_size_values, alpha=0.8
            )
            
            # 绘制边
            edge_list = list(self.graph.edges())
            edge_color_values = [self.edge_colors.get(tuple(sorted(edge)), "#95a5a6") for edge in edge_list]
            edge_width_values = [self.edge_weights.get(tuple(sorted(edge)), 1.0) for edge in edge_list]
            
            nx.draw_networkx_edges(
                self.graph, pos, edge_color=edge_color_values, 
                width=edge_width_values, alpha=0.6
            )
            
            # 添加标签（只对重要节点）
            label_nodes = [node for node in node_list if self.node_sizes.get(node, 0) > 800]
            labels = {node: node[:20] + "..." if len(node) > 20 else node for node in label_nodes}
            
            nx.draw_networkx_labels(
                self.graph, pos, labels=labels, font_size=10, font_color="#333333"
            )
            
            plt.title("知识图谱可视化 - 实体与事件关系", fontsize=16)
            plt.axis("off")
            plt.tight_layout()
        
            if output_format == "base64":
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                processing_time = (time.time() - start_time) * 1000
                tools.log(f"✅ [DEBUG] 图像生成完成，格式: base64，大小: {len(image_base64)} bytes，耗时: {processing_time:.2f}ms")
                return image_base64
            else:
                processing_time = (time.time() - start_time) * 1000
                tools.log(f"✅ [DEBUG] 图像生成完成，格式: figure，耗时: {processing_time:.2f}ms")
                return plt.gcf()
        except Exception as e:
            tools.log(f"❌ [DEBUG] 生成可视化图像出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            raise

class KGExplainer:
    """知识图谱解释器"""
    
    def __init__(self):
        tools.log(f"🔍 [DEBUG] KGExplainer初始化开始")
        self.visualizer = KGVisualizer()
        tools.log(f"✅ [DEBUG] KGExplainer初始化完成")
        
    def generate_explanation_report(self, focus_entity: Optional[str] = None) -> Dict:
        """
        生成知识图谱解释报告
        
        Args:
            focus_entity: 关注的实体，如果为None则生成全局报告
        
        Returns:
            解释报告字典
        """
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 生成解释报告，关注实体: {focus_entity if focus_entity else '全局'}")
        
        try:
            global KG
            if KG is None:
                tools.log(f"🔍 [DEBUG] 获取知识图谱实例")
                KG = get_knowledge_graph()
        
            report = {
                "generated_at": datetime.now().isoformat(),
                "graph_statistics": self._get_graph_statistics(),
                "key_entities": [],
                "key_events": [],
                "relationship_insights": [],
                "temporal_patterns": [],
                "recommendations": []
            }
        
            if focus_entity:
                report["focus_entity"] = focus_entity
                tools.log(f"🔍 [DEBUG] 生成实体 '{focus_entity}' 的详细报告")
                entity_report = self._generate_entity_report(focus_entity)
                report.update(entity_report)
            else:
                # 生成全局报告
                tools.log(f"🔍 [DEBUG] 生成全局报告")
                report["key_entities"] = self._get_top_entities(n=10)
                tools.log(f"📊 [DEBUG] 提取了 {len(report['key_entities'])} 个关键实体")
                report["key_events"] = self._get_top_events(n=10)
                tools.log(f"📊 [DEBUG] 提取了 {len(report['key_events'])} 个关键事件")
                report["relationship_insights"] = self._generate_relationship_insights()
                tools.log(f"📊 [DEBUG] 生成了 {len(report['relationship_insights'])} 个关系洞察")
                report["temporal_patterns"] = self._analyze_temporal_patterns()
                tools.log(f"📊 [DEBUG] 分析了 {len(report['temporal_patterns'])} 个时间模式")
        
            report["recommendations"] = self._generate_recommendations(report)
        
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"✅ [DEBUG] 解释报告生成完成，耗时: {processing_time:.2f}ms")
            return report
        except Exception as e:
            tools.log(f"❌ [DEBUG] 生成解释报告出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return {"error": str(e), "generated_at": datetime.now().isoformat()}
    
    def _get_graph_statistics(self) -> Dict:
        """获取图谱统计信息"""
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 计算图谱统计信息")
        
        try:
            global KG
            all_entities = KG.get_all_entities()
            tools.log(f"📊 [DEBUG] 当前实体总数: {len(all_entities)}")
            total_event_relations = 0
            total_entity_relations = 0
            
            # 计算所有实体-事件关系数量
            tools.log(f"🔍 [DEBUG] 计算实体-事件关系数量")
            for entity in all_entities:
                total_event_relations += len(KG.get_entity_events(entity))
            
            # 计算所有实体-实体关系数量
            tools.log(f"🔍 [DEBUG] 计算实体-实体关系数量")
            for entity in all_entities:
                total_entity_relations += len(KG.get_related_entities(entity))
            
            stats = {
                "total_entities": len(all_entities),
                "total_events": len(KG.get_all_events()),
                "total_entity_event_relations": total_event_relations,
                "total_entity_entity_relations": total_entity_relations,
                "avg_events_per_entity": self._calculate_avg_events_per_entity()
            }
            
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"📊 [DEBUG] 图谱统计完成 - 实体: {stats['total_entities']}, 事件: {stats['total_events']}, 实体-事件关系: {stats['total_entity_event_relations']}, 实体-实体关系: {stats['total_entity_entity_relations']}, 耗时: {processing_time:.2f}ms")
            return stats
        except Exception as e:
            tools.log(f"❌ [DEBUG] 计算图谱统计出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _calculate_avg_events_per_entity(self) -> float:
        """计算每个实体平均关联的事件数"""
        global KG
        all_entities = KG.get_all_entities()
        if not all_entities:
            return 0.0
        
        total_events = 0
        for entity in all_entities:
            total_events += len(KG.get_entity_events(entity))
            
        return round(total_events / len(all_entities), 2)
    
    def _get_top_entities(self, n: int = 10) -> List[Dict]:
        """获取事件关联最多的前N个实体"""
        global KG
        entity_event_counts = {}
        
        # 统计每个实体的事件数量
        all_entities = KG.get_all_entities()
        for entity in all_entities:
            entity_event_counts[entity] = len(KG.get_entity_events(entity))
        
        # 排序并返回前N个
        sorted_entities = sorted(entity_event_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return [{
            "entity": entity,
            "event_count": count,
            "related_entities": list(KG.get_related_entities(entity))[:5]
        } for entity, count in sorted_entities]
    
    def _get_top_events(self, n: int = 10) -> List[Dict]:
        """获取实体关联最多的前N个事件"""
        global KG
        event_entity_counts = {}
        
        # 统计每个事件关联的实体数量
        all_entities = KG.get_all_entities()
        for entity in all_entities:
            events = KG.get_entity_events(entity)
            for event in events:
                if event not in event_entity_counts:
                    event_entity_counts[event] = 0
                event_entity_counts[event] += 1
        
        # 排序并返回前N个
        sorted_events = sorted(event_entity_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return [{
            "event": event,
            "entity_count": count
        } for event, count in sorted_events]
    
    def _generate_entity_report(self, entity: str) -> Dict:
        """生成特定实体的详细报告"""
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 生成实体 '{entity}' 的详细报告")
        
        try:
            global KG
            
            if entity not in KG.get_all_entities():
                tools.log(f"❌ [DEBUG] 实体 '{entity}' 不存在于知识图谱中")
                return {"error": f"实体 '{entity}' 不存在于知识图谱中"}
        
            entity_data = KG.get_all_entities()[entity]
            tools.log(f"🔍 [DEBUG] 获取实体 '{entity}' 的相关事件")
            related_events = KG.get_entity_events(entity)
            tools.log(f"📊 [DEBUG] 实体 '{entity}' 参与了 {len(related_events)} 个事件")
            
            tools.log(f"🔍 [DEBUG] 获取实体 '{entity}' 的相关实体")
            related_entities = KG.get_related_entities(entity)
            tools.log(f"📊 [DEBUG] 实体 '{entity}' 关联了 {len(related_entities)} 个其他实体")
            
            # 分析实体的事件时间分布
            tools.log(f"🔍 [DEBUG] 分析实体 '{entity}' 的事件时间分布")
            event_timeline = []
            for event in related_events:
                event_data = KG.get_event_data(event)
                if event_data and "first_seen" in event_data:
                    event_timeline.append({
                        "event": event,
                        "timestamp": event_data["first_seen"]
                    })
            
            # 按时间排序
            event_timeline.sort(key=lambda x: x["timestamp"])
            
            importance = self._calculate_entity_importance(entity)
            tools.log(f"📊 [DEBUG] 实体 '{entity}' 重要性评分: {importance}")
            
            result = {
                "entity_details": entity_data,
                "related_events": related_events[:20],  # 限制数量
                "related_entities": list(related_entities)[:10],  # 限制数量
                "event_timeline": event_timeline,
                "entity_importance": importance
            }
            
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"✅ [DEBUG] 实体 '{entity}' 报告生成完成，耗时: {processing_time:.2f}ms")
            return result
        except Exception as e:
            tools.log(f"❌ [DEBUG] 生成实体报告出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _calculate_entity_importance(self, entity: str) -> float:
        """计算实体的重要性分数（0-1）"""
        global KG
        
        # 基于事件数量、关系数量和时间因素计算重要性
        event_count = len(KG.get_entity_events(entity))
        relation_count = len(KG.get_related_entities(entity))
        
        # 归一化计算
        max_event_count = 100  # 假设最大事件数为100
        max_relation_count = 50  # 假设最大关系数为50
        
        event_score = min(event_count / max_event_count, 1.0)
        relation_score = min(relation_count / max_relation_count, 1.0)
        
        # 综合分数
        importance = (event_score * 0.6) + (relation_score * 0.4)
        return round(importance, 3)
    
    def _generate_relationship_insights(self) -> List[Dict]:
        """生成关系洞察"""
        insights = []
        
        # 检测高度连接的实体组（社区）
        self.visualizer.build_visualization_graph(depth=1)
        if len(self.visualizer.graph.nodes()) > 5:
            communities = nx.community.greedy_modularity_communities(self.visualizer.graph)
            if len(communities) > 1:
                insights.append({
                    "type": "community_detection",
                    "message": f"检测到 {len(communities)} 个实体社区",
                    "details": f"最大社区包含 {max(len(c) for c in communities)} 个实体"
                })
        
        # 检测关键桥接实体
        bridges = list(nx.bridges(self.visualizer.graph))
        if bridges:
            bridge_entities = set()
            for u, v in bridges:
                bridge_entities.add(u)
                bridge_entities.add(v)
            
            insights.append({
                "type": "bridge_entities",
                "message": f"发现 {len(bridge_entities)} 个桥接实体，它们连接不同的实体社区",
                "details": list(bridge_entities)[:5]  # 最多显示5个
            })
        
        return insights
    
    def _analyze_temporal_patterns(self) -> List[Dict]:
        """分析时间模式"""
        global KG
        patterns = []
        
        # 收集所有事件的时间戳
        all_timestamps = []
        for event, event_data in KG.get_all_events().items():
            if "first_seen" in event_data:
                try:
                    dt = datetime.fromisoformat(event_data["first_seen"])
                    all_timestamps.append(dt)
                except:
                    pass
        
        if all_timestamps:
            # 按日期分组统计事件数量
            from collections import defaultdict
            date_counts = defaultdict(int)
            for dt in all_timestamps:
                date_key = dt.date()
                date_counts[date_key] += 1
            
            # 找出事件最活跃的日期
            if date_counts:
                top_date = max(date_counts.items(), key=lambda x: x[1])
                patterns.append({
                    "type": "active_date",
                    "message": f"事件最活跃的日期：{top_date[0]}，共有 {top_date[1]} 个事件",
                    "details": {"date": str(top_date[0]), "count": top_date[1]}
                })
        
        return patterns
    
    def _generate_recommendations(self, report: Dict) -> List[Dict]:
        """生成知识图谱优化建议"""
        recommendations = []
        
        # 基于图谱统计生成建议
        stats = report.get("graph_statistics", {})
        
        # 如果实体数量较少
        if stats.get("total_entities", 0) < 100:
            recommendations.append({
                "type": "entity_expansion",
                "message": "知识图谱中的实体数量较少，建议扩大数据源或增加实体提取的广度",
                "priority": "high"
            })
        
        # 如果平均每个实体关联的事件太少
        if stats.get("avg_events_per_entity", 0) < 5:
            recommendations.append({
                "type": "event_richness",
                "message": "实体关联的事件数量偏少，可能影响图谱的丰富度和分析价值",
                "priority": "medium"
            })
        
        # 如果实体间关系较少
        if stats.get("total_entity_entity_relations", 0) < stats.get("total_entities", 1) * 2:
            recommendations.append({
                "type": "relation_enhancement",
                "message": "实体间关系密度较低，建议增强实体共现分析",
                "priority": "medium"
            })
        
        return recommendations
    
    def detect_anomalies(self) -> List[Dict]:
        """检测知识图谱中的异常关系"""
        import time
        start_time = time.time()
        tools.log(f"🔍 [DEBUG] 开始检测知识图谱异常")
        
        try:
            global KG
            anomalies = []
            
            # 检测孤点实体（没有关联事件和关系的实体）
            tools.log(f"🔍 [DEBUG] 检测孤点实体")
            lonely_entities = []
            all_entities = KG.get_all_entities()
            tools.log(f"🔍 [DEBUG] 总实体数: {len(all_entities)}")
            
            for entity in all_entities:
                entity_events = KG.get_entity_events(entity)
                related_entities = KG.get_related_entities(entity)
                if len(entity_events) == 0 and len(related_entities) == 0:
                    lonely_entities.append(entity)
            
            if lonely_entities:
                tools.log(f"⚠️ [DEBUG] 发现 {len(lonely_entities)} 个孤点实体")
                anomalies.append({
                    "type": "lonely_entities",
                    "message": f"发现 {len(lonely_entities)} 个孤点实体（没有关联事件和关系）",
                    "details": lonely_entities[:10]  # 最多显示10个
                })
        
            # 检测过度关联的实体（可能存在噪声）
            tools.log(f"🔍 [DEBUG] 检测过度关联的实体")
            overly_connected = []
            
            for entity in all_entities:
                events = KG.get_entity_events(entity)
                if len(events) > 100:  # 关联事件超过100个
                    overly_connected.append({
                        "entity": entity,
                        "event_count": len(events)
                    })
            
            if overly_connected:
                tools.log(f"⚠️ [DEBUG] 发现 {len(overly_connected)} 个过度关联的实体")
                anomalies.append({
                    "type": "overly_connected",
                    "message": f"发现 {len(overly_connected)} 个过度关联的实体（事件关联数异常高）",
                    "details": overly_connected[:5]
                })
            
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"✅ [DEBUG] 异常检测完成，发现 {len(anomalies)} 种异常，耗时: {processing_time:.2f}ms")
            return anomalies
        except Exception as e:
            tools.log(f"❌ [DEBUG] 检测异常出错: {str(e)}")
            import traceback
            tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
            return [{"error": str(e)}]

# 全局实例
kg_visualizer = KGVisualizer()
kg_explainer = KGExplainer()

# 便捷函数
def visualize_entities(entities: List[str], depth: int = 2, output_format: str = "base64") -> Union[str, plt.Figure]:
    """可视化指定实体及其关系"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 visualize_entities，实体数: {len(entities)}, 深度: {depth}")
    kg_visualizer.build_visualization_graph(entities=entities, depth=depth)
    return kg_visualizer.generate_plot_image(output_format)

def visualize_full_graph(max_nodes: int = 100, output_format: str = "base64") -> Union[str, plt.Figure]:
    """可视化完整图谱（限制节点数量）"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 visualize_full_graph，最大节点数: {max_nodes}")
    # 构建一个包含重要实体的图谱
    global KG
    if KG is None:
        tools.log(f"🔍 [DEBUG] 获取知识图谱实例")
        from .kg_interface import get_knowledge_graph
        KG = get_knowledge_graph()
    
    # 获取事件最多的实体作为起始点
    entity_event_counts = {}
    # 统计每个实体参与的事件数量
    all_entities = KG.get_all_entities()
    for entity in all_entities:
        entity_events = KG.get_entity_events(entity)
        entity_event_counts[entity] = len(entity_events)
    
    # 排序并获取前N个实体
    sorted_entities = sorted(entity_event_counts.items(), key=lambda x: x[1], reverse=True)[:max_nodes//2]
    start_entities = [entity for entity, _ in sorted_entities]
    
    kg_visualizer.build_visualization_graph(entities=start_entities, depth=1)
    return kg_visualizer.generate_plot_image(output_format)

def generate_report(focus_entity: Optional[str] = None) -> Dict:
    """生成知识图谱解释报告"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 generate_report，关注实体: {focus_entity if focus_entity else '全局'}")
    return kg_explainer.generate_explanation_report(focus_entity)

def detect_graph_anomalies() -> List[Dict]:
    """检测知识图谱异常"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 detect_graph_anomalies")
    return kg_explainer.detect_anomalies()

def get_graph_statistics() -> Dict:
    """获取图谱统计信息"""
    tools.log(f"🔍 [DEBUG] 调用便捷函数 get_graph_statistics")
    return kg_explainer._get_graph_statistics()

def export_graph_data(format: str = "json") -> Union[Dict, str]:
    """
    导出知识图谱数据
    
    Args:
        format: 导出格式，支持"json"
    
    Returns:
        导出的数据
    """
    import time
    start_time = time.time()
    tools.log(f"🔍 [DEBUG] 调用便捷函数 export_graph_data，格式: {format}")
    
    try:
        global KG
        if KG is None:
            tools.log(f"🔍 [DEBUG] 获取知识图谱实例")
            from .kg_interface import get_knowledge_graph
            KG = get_knowledge_graph()
            
        tools.log(f"🔍 [DEBUG] 准备导出知识图谱数据")
        export_data = {
            "entities": KG.get_all_entities(),
            "events": KG.get_all_events(),
            "entity_relations": KG.get_entity_relations(),
            "entity_event_relations": KG.get_entity_event_relations(),
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "entity_count": len(KG.get_all_entities()),
                "event_count": len(KG.get_all_events())
            }
        }
        
        tools.log(f"📊 [DEBUG] 导出数据统计 - 实体数: {export_data['metadata']['entity_count']}, 事件数: {export_data['metadata']['event_count']}")
        
        if format == "json":
            processing_time = (time.time() - start_time) * 1000
            tools.log(f"✅ [DEBUG] 数据导出完成，格式: json，耗时: {processing_time:.2f}ms")
            return export_data
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    except Exception as e:
        tools.log(f"❌ [DEBUG] 导出数据出错: {str(e)}")
        import traceback
        tools.log(f"❌ [DEBUG] 错误堆栈: {traceback.format_exc()}")
        raise
    
    export_data = {
        "entities": KG.get_all_entities(),
        "events": KG.get_all_events(),
        "entity_relations": KG.get_entity_relations(),
        "entity_event_relations": KG.get_entity_event_relations(),
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "entity_count": len(KG.get_all_entities()),
            "event_count": len(KG.get_all_events())
        }
    }
    
    if format == "json":
        return export_data
    else:
        raise ValueError(f"不支持的导出格式: {format}")
