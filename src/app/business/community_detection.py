"""
业务层 - 社区图谱分析

实现社区检测和超实体建模功能。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import networkx as nx
import pandas as pd
from networkx.algorithms import community

from ...infra import get_logger
from ...adapters.neo4j.store import get_neo4j_store


class CommunityDetector:
    """社区检测器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.store = get_neo4j_store()
        
    def detect_communities_louvain(self, date_range: Tuple[str, str]) -> Dict[str, Any]:
        """
        使用Louvain算法检测社区
        
        Args:
            date_range: 日期范围 (start_date, end_date)
            
        Returns:
            Dict: 社区检测结果
        """
        try:
            # 构建实体关系图
            graph = self._build_entity_relation_graph(date_range)
            
            # 应用Louvain算法检测社区
            communities = community.louvain_communities(graph)
            
            # 分析社区特征
            community_analysis = self._analyze_communities(graph, communities)
            
            # 保存社区信息到图数据库
            self._save_communities(community_analysis)
            
            return community_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to detect communities: {e}")
            raise
    
    def _build_entity_relation_graph(self, date_range: Tuple[str, str]) -> nx.Graph:
        """
        构建实体关系图
        
        Args:
            date_range: 日期范围
            
        Returns:
            nx.Graph: 实体关系图
        """
        start_date, end_date = date_range
        
        with self.store._driver.session() as session:
            # 获取实体关系数据
            result = session.run(
                """
                MATCH (e1:Entity)-[r:RELATION]-(e2:Entity)
                MATCH (event:Event)-[:PARTICIPATES]->(e1:Entity)
                WHERE event.date >= $start_date AND event.date <= $end_date
                RETURN e1.name AS entity1, e2.name AS entity2, 
                       count(r) AS relation_count, avg(r.strength) AS avg_strength
                """,
                start_date=start_date,
                end_date=end_date
            )
            
            # 构建图
            graph = nx.Graph()
            
            for record in result:
                entity1 = record["entity1"]
                entity2 = record["entity2"]
                weight = record["avg_strength"] or 0.0
                
                # 添加节点
                if not graph.has_node(entity1):
                    graph.add_node(entity1)
                if not graph.has_node(entity2):
                    graph.add_node(entity2)
                
                # 添加边
                if graph.has_edge(entity1, entity2):
                    # 如果边已存在，累加权重
                    current_weight = graph[entity1][entity2].get("weight", 0.0)
                    graph[entity1][entity2]["weight"] = current_weight + weight
                else:
                    graph.add_edge(entity1, entity2, weight=weight)
            
            return graph
    
    def _analyze_communities(self, graph: nx.Graph, communities: List[set]) -> Dict[str, Any]:
        """
        分析社区特征
        
        Args:
            graph: 实体关系图
            communities: 社区列表
            
        Returns:
            Dict: 社区分析结果
        """
        analysis = {
            "communities": [],
            "statistics": {}
        }
        
        # 分析每个社区
        for i, community_nodes in enumerate(communities):
            if len(community_nodes) < 2:
                continue
                
            # 构建社区子图
            subgraph = graph.subgraph(community_nodes)
            
            # 计算社区特征
            community_info = {
                "id": f"community_{i}",
                "members": list(community_nodes),
                "size": len(community_nodes),
                "density": nx.density(subgraph),
                "avg_clustering": nx.average_clustering(subgraph),
                "central_entities": self._find_central_entities(subgraph),
                "internal_relations": subgraph.number_of_edges(),
                "total_weight": sum([d["weight"] for u, v, d in subgraph.edges(data=True)])
            }
            
            analysis["communities"].append(community_info)
        
        # 计算总体统计信息
        analysis["statistics"] = {
            "total_communities": len(analysis["communities"]),
            "avg_community_size": sum([c["size"] for c in analysis["communities"]]) / len(analysis["communities"]) if analysis["communities"] else 0,
            "largest_community": max([c["size"] for c in analysis["communities"]]) if analysis["communities"] else 0,
            "smallest_community": min([c["size"] for c in analysis["communities"]]) if analysis["communities"] else 0
        }
        
        return analysis
    
    def _find_central_entities(self, subgraph: nx.Graph) -> List[Dict[str, Any]]:
        """
        查找社区中的中心实体
        
        Args:
            subgraph: 社区子图
            
        Returns:
            List[Dict]: 中心实体列表
        """
        # 计算度中心性
        degree_centrality = nx.degree_centrality(subgraph)
        
        # 计算介数中心性
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        
        # 计算接近中心性
        closeness_centrality = nx.closeness_centrality(subgraph)
        
        # 综合排名
        central_entities = []
        for node in subgraph.nodes():
            centrality_score = (
                degree_centrality[node] + 
                betweenness_centrality[node] + 
                closeness_centrality[node]
            ) / 3
            
            central_entities.append({
                "name": node,
                "centrality_score": centrality_score,
                "degree": degree_centrality[node],
                "betweenness": betweenness_centrality[node],
                "closeness": closeness_centrality[node]
            })
        
        # 按中心性得分排序，返回前5个
        central_entities.sort(key=lambda x: x["centrality_score"], reverse=True)
        return central_entities[:5]
    
    def _save_communities(self, analysis: Dict[str, Any]) -> None:
        """
        保存社区信息到图数据库
        
        Args:
            analysis: 社区分析结果
        """
        with self.store._driver.session() as session:
            # 保存每个社区
            for community_info in analysis["communities"]:
                # 创建社区节点
                session.run(
                    """
                    MERGE (c:Community {id: $id})
                    SET c.size = $size, c.density = $density, 
                        c.avg_clustering = $avg_clustering,
                        c.internal_relations = $internal_relations,
                        c.total_weight = $total_weight,
                        c.created_at = $created_at
                    """,
                    id=community_info["id"],
                    size=community_info["size"],
                    density=community_info["density"],
                    avg_clustering=community_info["avg_clustering"],
                    internal_relations=community_info["internal_relations"],
                    total_weight=community_info["total_weight"],
                    created_at=datetime.now().isoformat()
                )
                
                # 关联社区成员
                for member in community_info["members"]:
                    session.run(
                        """
                        MATCH (c:Community {id: $community_id})
                        MATCH (e:Entity {name: $entity_name})
                        MERGE (c)-[:MEMBER]->(e)
                        """,
                        community_id=community_info["id"],
                        entity_name=member
                    )
                
                # 保存中心实体
                for central_entity in community_info["central_entities"]:
                    session.run(
                        """
                        MATCH (c:Community {id: $community_id})
                        MATCH (e:Entity {name: $entity_name})
                        MERGE (c)-[r:CENTRAL_ENTITY]->(e)
                        SET r.centrality_score = $centrality_score,
                            r.degree = $degree,
                            r.betweenness = $betweenness,
                            r.closeness = $closeness
                        """,
                        community_id=community_info["id"],
                        entity_name=central_entity["name"],
                        centrality_score=central_entity["centrality_score"],
                        degree=central_entity["degree"],
                        betweenness=central_entity["betweenness"],
                        closeness=central_entity["closeness"]
                    )
    
    def get_community_by_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        根据实体获取所属社区信息
        
        Args:
            entity_name: 实体名称
            
        Returns:
            Dict: 社区信息
        """
        with self.store._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $entity_name})<-[:MEMBER]-(c:Community)
                OPTIONAL MATCH (c)-[r:CENTRAL_ENTITY]->(ce:Entity)
                RETURN c.id AS community_id, c.size AS size, c.density AS density,
                       collect({name: ce.name, centrality_score: r.centrality_score}) AS central_entities
                """,
                entity_name=entity_name
            )
            
            record = result.single()
            if record:
                return {
                    "community_id": record["community_id"],
                    "size": record["size"],
                    "density": record["density"],
                    "central_entities": record["central_entities"]
                }
            
            return None
    
    def get_community_evolution(self, community_id: str, date_ranges: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        获取社区演化信息
        
        Args:
            community_id: 社区ID
            date_ranges: 日期范围列表
            
        Returns:
            List[Dict]: 社区演化信息
        """
        evolution_data = []
        
        with self.store._driver.session() as session:
            for i, (start_date, end_date) in enumerate(date_ranges):
                result = session.run(
                    """
                    MATCH (c:Community {id: $community_id})
                    MATCH (c)-[:MEMBER]->(e:Entity)
                    MATCH (e)-[:PARTICIPATES]->(event:Event)
                    WHERE event.date >= $start_date AND event.date <= $end_date
                    RETURN count(DISTINCT e) AS member_count,
                           count(DISTINCT event) AS event_count,
                           avg(event.goldstein_scale) AS avg_goldstein
                    """,
                    community_id=community_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                record = result.single()
                if record:
                    evolution_data.append({
                        "period": f"Period {i+1}",
                        "start_date": start_date,
                        "end_date": end_date,
                        "member_count": record["member_count"],
                        "event_count": record["event_count"],
                        "avg_goldstein": record["avg_goldstein"] or 0.0
                    })
        
        return evolution_data
    
    def detect_viewpoint_camps(self, date_range: Tuple[str, str]) -> Dict[str, Any]:
        """
        检测观点阵营
        
        Args:
            date_range: 日期范围
            
        Returns:
            Dict: 观点阵营分析结果
        """
        start_date, end_date = date_range
        
        with self.store._driver.session() as session:
            # 获取具有不同情感倾向的实体关系
            result = session.run(
                """
                MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                MATCH (event:Event)-[:PARTICIPATES]->(e1)
                WHERE event.date >= $start_date AND event.date <= $end_date
                RETURN e1.name AS source, e2.name AS target, 
                       avg(event.avg_tone) AS avg_tone,
                       count(r) AS relation_count
                """,
                start_date=start_date,
                end_date=end_date
            )
            
            # 分析正面和负面阵营
            positive_relations = []
            negative_relations = []
            
            for record in result:
                avg_tone = record["avg_tone"] or 0.0
                if avg_tone > 0:
                    positive_relations.append(record)
                elif avg_tone < 0:
                    negative_relations.append(record)
            
            return {
                "positive_camp": {
                    "relations": positive_relations,
                    "size": len(positive_relations)
                },
                "negative_camp": {
                    "relations": negative_relations,
                    "size": len(negative_relations)
                },
                "neutral_count": len(positive_relations) + len(negative_relations),
                "analyzed_at": datetime.now().isoformat()
            }


# 工厂函数
def get_community_detector() -> CommunityDetector:
    """
    获取社区检测器实例
    
    Returns:
        CommunityDetector: 社区检测器实例
    """
    return CommunityDetector()