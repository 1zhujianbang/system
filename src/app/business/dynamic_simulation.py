"""
业务层 - 动态仿真引擎

实现事件流驱动的动态仿真引擎。
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import networkx as nx
import pandas as pd

from ...infra import get_logger
from ...adapters.neo4j.store import get_neo4j_store


class DynamicSimulationEngine:
    """动态仿真引擎"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.store = get_neo4j_store()
        self.entity_network = nx.Graph()
        self.event_rules = {}
        
    def initialize_simulation(self, date_range: Tuple[str, str]) -> None:
        """
        初始化仿真环境
        
        Args:
            date_range: 日期范围 (start_date, end_date)
        """
        try:
            # 构建初始实体网络
            self.entity_network = self._build_initial_entity_network(date_range)
            
            # 加载仿真规则
            self.event_rules = self._load_simulation_rules()
            
            self.logger.info("Simulation environment initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simulation: {e}")
            raise
    
    def _build_initial_entity_network(self, date_range: Tuple[str, str]) -> nx.Graph:
        """
        构建初始实体网络
        
        Args:
            date_range: 日期范围
            
        Returns:
            nx.Graph: 实体网络
        """
        start_date, end_date = date_range
        
        with self.store._driver.session() as session:
            # 获取实体关系数据
            result = session.run(
                """
                MATCH (e1:Entity)-[r:RELATION]-(e2:Entity)
                MATCH (event:Event)-[:PARTICIPATES]->(e1)
                WHERE event.date >= $start_date AND event.date <= $end_date
                RETURN e1.name AS entity1, e2.name AS entity2, 
                       avg(r.strength) AS avg_strength, count(r) AS relation_count
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
                    graph.add_node(entity1, state={"influence": random.random()})
                if not graph.has_node(entity2):
                    graph.add_node(entity2, state={"influence": random.random()})
                
                # 添加边
                graph.add_edge(entity1, entity2, weight=weight)
            
            return graph
    
    def _load_simulation_rules(self) -> Dict[str, Any]:
        """
        加载仿真规则
        
        Returns:
            Dict: 仿真规则
        """
        # 默认规则
        default_rules = {
            "conflict_update": {
                "event_code": "190",  # 冲突事件
                "probability": 0.7,
                "effect_radius": 2,
                "state_change_factor": 0.1
            },
            "cooperation_update": {
                "event_code": "050",  # 合作事件
                "probability": 0.6,
                "effect_radius": 3,
                "state_change_factor": 0.05
            },
            "influence_propagation": {
                "decay_factor": 0.9,
                "threshold": 0.3
            }
        }
        
        # 从数据库加载自定义规则（如果有）
        with self.store._driver.session() as session:
            result = session.run(
                """
                MATCH (r:SimulationRule)
                RETURN r.rule_type AS rule_type, r.parameters AS parameters
                """
            )
            
            custom_rules = {}
            for record in result:
                custom_rules[record["rule_type"]] = record["parameters"]
            
            # 合并默认规则和自定义规则
            rules = default_rules.copy()
            rules.update(custom_rules)
            
            return rules
    
    def run_simulation(
        self, 
        iterations: int = 100, 
        event_stream: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        运行仿真
        
        Args:
            iterations: 迭代次数
            event_stream: 事件流（可选）
            
        Returns:
            Dict: 仿真结果
        """
        try:
            simulation_results = {
                "iterations": iterations,
                "initial_state": self._capture_network_state(),
                "final_state": {},
                "evolution_history": [],
                "key_events": [],
                "statistics": {}
            }
            
            # 如果没有提供事件流，则生成模拟事件流
            if event_stream is None:
                event_stream = self._generate_event_stream(iterations)
            
            # 运行仿真迭代
            for i in range(iterations):
                if i < len(event_stream):
                    event = event_stream[i]
                    self._process_event(event)
                
                # 记录当前状态
                current_state = self._capture_network_state()
                simulation_results["evolution_history"].append({
                    "iteration": i,
                    "timestamp": datetime.now().isoformat(),
                    "state": current_state
                })
                
                # 检查关键事件
                key_events = self._detect_key_events(current_state)
                if key_events:
                    simulation_results["key_events"].extend(key_events)
            
            # 记录最终状态
            simulation_results["final_state"] = self._capture_network_state()
            
            # 计算统计信息
            simulation_results["statistics"] = self._calculate_statistics(simulation_results)
            
            self.logger.info(f"Simulation completed with {iterations} iterations")
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
    
    def _generate_event_stream(self, count: int) -> List[Dict[str, Any]]:
        """
        生成模拟事件流
        
        Args:
            count: 事件数量
            
        Returns:
            List[Dict]: 事件流
        """
        event_stream = []
        event_codes = ["190", "050", "140", "080", "100"]  # 常见事件代码
        
        for i in range(count):
            # 随机选择实体
            entities = list(self.entity_network.nodes())
            if len(entities) >= 2:
                actor1 = random.choice(entities)
                actor2 = random.choice([e for e in entities if e != actor1])
                
                event = {
                    "event_id": f"sim_event_{i}",
                    "event_code": random.choice(event_codes),
                    "actor1": actor1,
                    "actor2": actor2,
                    "date": (datetime.now() + timedelta(days=i)).isoformat(),
                    "goldstein_scale": random.uniform(-10, 10),
                    "num_mentions": random.randint(1, 100)
                }
                
                event_stream.append(event)
        
        return event_stream
    
    def _process_event(self, event: Dict[str, Any]) -> None:
        """
        处理单个事件
        
        Args:
            event: 事件数据
        """
        event_code = event.get("event_code", "")
        actor1 = event.get("actor1", "")
        actor2 = event.get("actor2", "")
        
        # 根据事件类型应用不同的规则
        if event_code == self.event_rules.get("conflict_update", {}).get("event_code"):
            self._apply_conflict_rule(actor1, actor2, event)
        elif event_code == self.event_rules.get("cooperation_update", {}).get("event_code"):
            self._apply_cooperation_rule(actor1, actor2, event)
        
        # 传播影响
        self._propagate_influence(actor1, actor2)
    
    def _apply_conflict_rule(self, actor1: str, actor2: str, event: Dict[str, Any]) -> None:
        """
        应用冲突规则
        
        Args:
            actor1: 行动者1
            actor2: 行动者2
            event: 事件数据
        """
        conflict_rule = self.event_rules.get("conflict_update", {})
        effect_radius = conflict_rule.get("effect_radius", 2)
        state_change_factor = conflict_rule.get("state_change_factor", 0.1)
        
        # 获取受影响的节点
        affected_nodes = nx.single_source_shortest_path_length(
            self.entity_network, actor1, cutoff=effect_radius
        ).keys()
        
        # 更新节点状态
        for node in affected_nodes:
            if self.entity_network.has_node(node):
                current_state = self.entity_network.nodes[node].get("state", {})
                current_influence = current_state.get("influence", 0.5)
                
                # 减少影响值
                new_influence = max(0.0, current_influence - state_change_factor)
                self.entity_network.nodes[node]["state"] = {
                    **current_state,
                    "influence": new_influence
                }
    
    def _apply_cooperation_rule(self, actor1: str, actor2: str, event: Dict[str, Any]) -> None:
        """
        应用合作规则
        
        Args:
            actor1: 行动者1
            actor2: 行动者2
            event: 事件数据
        """
        cooperation_rule = self.event_rules.get("cooperation_update", {})
        effect_radius = cooperation_rule.get("effect_radius", 3)
        state_change_factor = cooperation_rule.get("state_change_factor", 0.05)
        
        # 获取受影响的节点
        affected_nodes = nx.single_source_shortest_path_length(
            self.entity_network, actor1, cutoff=effect_radius
        ).keys()
        
        # 更新节点状态
        for node in affected_nodes:
            if self.entity_network.has_node(node):
                current_state = self.entity_network.nodes[node].get("state", {})
                current_influence = current_state.get("influence", 0.5)
                
                # 增加影响值
                new_influence = min(1.0, current_influence + state_change_factor)
                self.entity_network.nodes[node]["state"] = {
                    **current_state,
                    "influence": new_influence
                }
    
    def _propagate_influence(self, source: str, target: str) -> None:
        """
        传播影响
        
        Args:
            source: 源实体
            target: 目标实体
        """
        propagation_rule = self.event_rules.get("influence_propagation", {})
        decay_factor = propagation_rule.get("decay_factor", 0.9)
        threshold = propagation_rule.get("threshold", 0.3)
        
        # 获取源节点的影响值
        if self.entity_network.has_node(source):
            source_influence = self.entity_network.nodes[source].get("state", {}).get("influence", 0.0)
            
            # 向邻居传播影响
            neighbors = list(self.entity_network.neighbors(source))
            for neighbor in neighbors:
                if self.entity_network.has_node(neighbor):
                    current_state = self.entity_network.nodes[neighbor].get("state", {})
                    current_influence = current_state.get("influence", 0.5)
                    
                    # 计算传播的影响值
                    propagated_influence = source_influence * decay_factor
                    
                    # 只有当传播的影响值超过阈值时才更新
                    if propagated_influence > threshold:
                        new_influence = min(1.0, current_influence + propagated_influence * 0.1)
                        self.entity_network.nodes[neighbor]["state"] = {
                            **current_state,
                            "influence": new_influence
                        }
    
    def _capture_network_state(self) -> Dict[str, Any]:
        """
        捕获网络状态
        
        Returns:
            Dict: 网络状态
        """
        state = {
            "nodes": {},
            "edges": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # 捕获节点状态
        for node in self.entity_network.nodes():
            state["nodes"][node] = self.entity_network.nodes[node].get("state", {})
        
        # 捕获边的状态
        for edge in self.entity_network.edges():
            state["edges"][f"{edge[0]}-{edge[1]}"] = self.entity_network.edges[edge]
        
        return state
    
    def _detect_key_events(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检测关键事件
        
        Args:
            current_state: 当前状态
            
        Returns:
            List[Dict]: 关键事件列表
        """
        key_events = []
        
        # 检测影响力显著变化的实体
        for node, state in current_state["nodes"].items():
            influence = state.get("influence", 0.5)
            if influence > 0.8 or influence < 0.2:
                key_events.append({
                    "type": "significant_influence_change",
                    "entity": node,
                    "influence": influence,
                    "timestamp": current_state["timestamp"]
                })
        
        return key_events
    
    def _calculate_statistics(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算统计信息
        
        Args:
            simulation_results: 仿真结果
            
        Returns:
            Dict: 统计信息
        """
        statistics = {
            "total_iterations": simulation_results["iterations"],
            "total_key_events": len(simulation_results["key_events"]),
            "avg_nodes": len(self.entity_network.nodes()),
            "avg_edges": len(self.entity_network.edges()),
            "simulation_duration": datetime.now().isoformat()
        }
        
        # 计算影响力分布
        influences = []
        for node in self.entity_network.nodes():
            state = self.entity_network.nodes[node].get("state", {})
            influences.append(state.get("influence", 0.5))
        
        if influences:
            statistics["avg_influence"] = sum(influences) / len(influences)
            statistics["max_influence"] = max(influences)
            statistics["min_influence"] = min(influences)
        
        return statistics
    
    def visualize_evolution(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        可视化演化过程
        
        Args:
            simulation_results: 仿真结果
            
        Returns:
            Dict: 可视化数据
        """
        visualization_data = {
            "influence_timeline": [],
            "network_metrics": [],
            "key_events_timeline": []
        }
        
        # 构建影响力时间线
        for record in simulation_results["evolution_history"]:
            timestamp = record["timestamp"]
            state = record["state"]
            
            # 计算平均影响力
            influences = [s.get("influence", 0.5) for s in state["nodes"].values()]
            avg_influence = sum(influences) / len(influences) if influences else 0.0
            
            visualization_data["influence_timeline"].append({
                "timestamp": timestamp,
                "avg_influence": avg_influence,
                "high_influence_count": len([i for i in influences if i > 0.7]),
                "low_influence_count": len([i for i in influences if i < 0.3])
            })
        
        # 构建关键事件时间线
        for event in simulation_results["key_events"]:
            visualization_data["key_events_timeline"].append({
                "timestamp": event["timestamp"],
                "event_type": event["type"],
                "entity": event["entity"],
                "details": event
            })
        
        return visualization_data


# 工厂函数
def get_dynamic_simulation_engine() -> DynamicSimulationEngine:
    """
    获取动态仿真引擎实例
    
    Returns:
        DynamicSimulationEngine: 动态仿真引擎实例
    """
    return DynamicSimulationEngine()