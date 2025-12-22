"""
业务层 - 因果网络构建

实现事件-事件效应图谱，构建因果网络并提供查询接口。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import networkx as nx
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

from ...infra import get_logger
from ...adapters.neo4j.store import get_neo4j_store


class CausalNetworkBuilder:
    """因果网络构建器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.graph = nx.DiGraph()
        self.store = get_neo4j_store()
        
    def build_from_gdelt_data(self, start_date: str, end_date: str) -> nx.DiGraph:
        """
        从GDELT数据构建因果网络
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            nx.DiGraph: 因果网络图
        """
        try:
            # 从Neo4j获取事件数据
            events = self._fetch_events_from_neo4j(start_date, end_date)
            
            # 构建事件类型序列
            event_sequences = self._build_event_sequences(events)
            
            # 使用PC算法构建因果网络
            causal_graph = self._apply_pc_algorithm(event_sequences)
            
            # 保存到图数据库
            self._save_causal_network(causal_graph)
            
            return causal_graph
            
        except Exception as e:
            self.logger.error(f"Failed to build causal network: {e}")
            raise
    
    def _fetch_events_from_neo4j(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        从Neo4j获取事件数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[Dict]: 事件数据列表
        """
        with self.store._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)
                WHERE e.date >= $start_date AND e.date <= $end_date
                RETURN e.event_code AS event_code, e.date AS date, 
                       e.goldstein_scale AS goldstein_scale, e.num_mentions AS num_mentions
                ORDER BY e.date
                """,
                start_date=start_date,
                end_date=end_date
            )
            
            events = []
            for record in result:
                events.append({
                    "event_code": record["event_code"],
                    "date": record["date"],
                    "goldstein_scale": record["goldstein_scale"],
                    "num_mentions": record["num_mentions"]
                })
            
            return events
    
    def _build_event_sequences(self, events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        构建事件类型时间序列
        
        Args:
            events: 事件数据列表
            
        Returns:
            pd.DataFrame: 事件类型时间序列数据
        """
        # 按日期分组，统计各事件类型的频次
        df = pd.DataFrame(events)
        df["date"] = pd.to_datetime(df["date"])
        
        # 创建每日事件类型计数矩阵
        daily_counts = df.groupby([df["date"].dt.date, "event_code"]).size().unstack(fill_value=0)
        
        return daily_counts
    
    def _apply_pc_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        应用PC算法构建因果网络
        
        Args:
            data: 时间序列数据
            
        Returns:
            nx.DiGraph: 因果网络图
        """
        # 转换为numpy数组
        data_array = data.values
        
        # 应用PC算法
        cg = pc(data_array, alpha=0.05, indep_test=fisherz, stable=True)
        
        # 转换为NetworkX图
        graph = nx.DiGraph()
        
        # 添加节点
        for i, col in enumerate(data.columns):
            graph.add_node(col, label=f"EventCode_{col}")
        
        # 添加边
        adjacency_matrix = cg.G.graph
        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                if adjacency_matrix[i, j] == 1:  # 存在因果关系
                    source = data.columns[i]
                    target = data.columns[j]
                    # 计算边的权重（基于相关性和Goldstein Scale）
                    weight = self._calculate_edge_weight(data, source, target)
                    graph.add_edge(source, target, weight=weight)
        
        return graph
    
    def _calculate_edge_weight(self, data: pd.DataFrame, source: str, target: str) -> float:
        """
        计算边的权重
        
        Args:
            data: 时间序列数据
            source: 源节点
            target: 目标节点
            
        Returns:
            float: 边的权重
        """
        # 计算相关性作为权重的基础
        correlation = data[source].corr(data[target])
        
        # 可以进一步结合Goldstein Scale等指标
        # 这里简化处理，直接返回相关性
        return abs(correlation)
    
    def _save_causal_network(self, graph: nx.DiGraph) -> None:
        """
        保存因果网络到图数据库
        
        Args:
            graph: 因果网络图
        """
        with self.store._driver.session() as session:
            # 创建因果关系节点
            for node in graph.nodes():
                session.run(
                    """
                    MERGE (c:CausalNode {event_code: $event_code})
                    SET c.label = $label
                    """,
                    event_code=node,
                    label=graph.nodes[node].get("label", f"EventCode_{node}")
                )
            
            # 创建因果关系边
            for source, target, data in graph.edges(data=True):
                session.run(
                    """
                    MATCH (s:CausalNode {event_code: $source})
                    MATCH (t:CausalNode {event_code: $target})
                    MERGE (s)-[r:CAUSES]->(t)
                    SET r.weight = $weight, r.created_at = $created_at
                    """,
                    source=source,
                    target=target,
                    weight=data.get("weight", 0.0),
                    created_at=datetime.now().isoformat()
                )
    
    def query_causal_probability(self, source_event: str, target_event: str) -> Dict[str, Any]:
        """
        查询事件间的因果概率
        
        Args:
            source_event: 源事件类型
            target_event: 目标事件类型
            
        Returns:
            Dict: 因果概率信息
        """
        with self.store._driver.session() as session:
            result = session.run(
                """
                MATCH (s:CausalNode {event_code: $source})-[r:CAUSES]->(t:CausalNode {event_code: $target})
                RETURN r.weight AS probability, r.created_at AS created_at
                """,
                source=source_event,
                target=target_event
            )
            
            record = result.single()
            if record:
                return {
                    "source": source_event,
                    "target": target_event,
                    "probability": record["probability"],
                    "created_at": record["created_at"]
                }
            else:
                return {
                    "source": source_event,
                    "target": target_event,
                    "probability": 0.0,
                    "created_at": None
                }
    
    def trace_causal_chain(self, start_event: str, max_depth: int = 5) -> List[List[str]]:
        """
        追踪因果链
        
        Args:
            start_event: 起始事件类型
            max_depth: 最大深度
            
        Returns:
            List[List[str]]: 因果链路径列表
        """
        with self.store._driver.session() as session:
            paths = []
            
            # 使用Cypher查询追踪因果链
            result = session.run(
                """
                MATCH p = (start:CausalNode {event_code: $start})-[:CAUSES*1..$max_depth]->(end:CausalNode)
                RETURN [n IN nodes(p) | n.event_code] AS path
                """,
                start=start_event,
                max_depth=max_depth
            )
            
            for record in result:
                paths.append(record["path"])
            
            return paths
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """
        获取网络统计信息
        
        Returns:
            Dict: 网络统计信息
        """
        with self.store._driver.session() as session:
            # 获取节点数
            node_count_result = session.run("MATCH (n:CausalNode) RETURN count(n) AS count")
            node_count = node_count_result.single()["count"]
            
            # 获取边数
            edge_count_result = session.run("MATCH ()-[r:CAUSES]->() RETURN count(r) AS count")
            edge_count = edge_count_result.single()["count"]
            
            # 获取平均权重
            avg_weight_result = session.run("MATCH ()-[r:CAUSES]->() RETURN avg(r.weight) AS avg_weight")
            avg_weight = avg_weight_result.single()["avg_weight"] or 0.0
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "average_weight": avg_weight,
                "density": edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0
            }


# 工厂函数
def get_causal_network_builder() -> CausalNetworkBuilder:
    """
    获取因果网络构建器实例
    
    Returns:
        CausalNetworkBuilder: 因果网络构建器实例
    """
    return CausalNetworkBuilder()