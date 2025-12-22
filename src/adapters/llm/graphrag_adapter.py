"""
适配器层 - GraphRAG 集成适配器

实现 GraphRAG 功能，提供基于图谱的检索增强生成能力。
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ...ports.llm_client import (
    LLMClient, LLMCallConfig, LLMResponse, LLMProviderType
)
from ...infra import get_logger


class GraphRAGAdapter:
    """GraphRAG 适配器"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        graph_database_uri: str,
        graph_database_auth: tuple[str, str],
        index_path: Optional[Path] = None
    ):
        """
        初始化 GraphRAG 适配器
        
        Args:
            llm_client: LLM 客户端
            graph_database_uri: 图数据库 URI
            graph_database_auth: 图数据库认证信息
            index_path: 索引路径
        """
        self._llm_client = llm_client
        self._graph_database_uri = graph_database_uri
        self._graph_database_auth = graph_database_auth
        self._index_path = index_path or Path("./graphrag_index")
        self._logger = get_logger(__name__)
        
        # 初始化 GraphRAG 组件
        self._initialize_graphrag()

    def _initialize_graphrag(self):
        """初始化 GraphRAG 组件"""
        try:
            # 创建索引目录
            self._index_path.mkdir(parents=True, exist_ok=True)
            
            # 初始化图数据库连接
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._graph_database_uri, 
                auth=self._graph_database_auth
            )
            
            self._logger.info("GraphRAG adapter initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize GraphRAG adapter: {e}")
            raise

    def close(self):
        """关闭 GraphRAG 适配器"""
        if hasattr(self, '_driver'):
            self._driver.close()

    def query_graph(self, query: str, config: Optional[LLMCallConfig] = None) -> LLMResponse:
        """
        基于图谱的查询
        
        Args:
            query: 查询语句
            config: LLM 调用配置
            
        Returns:
            LLMResponse: 响应结果
        """
        try:
            # 1. 从图数据库中检索相关信息
            context = self._retrieve_graph_context(query)
            
            # 2. 构造增强提示词
            enhanced_prompt = self._construct_enhanced_prompt(query, context)
            
            # 3. 调用 LLM 生成答案
            response = self._llm_client.call(enhanced_prompt, config)
            
            return response
        except Exception as e:
            self._logger.error(f"GraphRAG query failed: {e}")
            return LLMResponse(error=str(e))

    async def query_graph_async(self, query: str, config: Optional[LLMCallConfig] = None) -> LLMResponse:
        """
        异步基于图谱的查询
        
        Args:
            query: 查询语句
            config: LLM 调用配置
            
        Returns:
            LLMResponse: 响应结果
        """
        try:
            # 1. 从图数据库中检索相关信息
            context = await self._retrieve_graph_context_async(query)
            
            # 2. 构造增强提示词
            enhanced_prompt = self._construct_enhanced_prompt(query, context)
            
            # 3. 异步调用 LLM 生成答案
            response = await self._llm_client.call_async(enhanced_prompt, config)
            
            return response
        except Exception as e:
            self._logger.error(f"GraphRAG async query failed: {e}")
            return LLMResponse(error=str(e))

    def _retrieve_graph_context(self, query: str) -> Dict[str, Any]:
        """
        从图数据库中检索上下文信息
        
        Args:
            query: 查询语句
            
        Returns:
            Dict: 上下文信息
        """
        with self._driver.session() as session:
            # 查找与查询相关的实体
            entities_result = session.run(
                """
                CALL db.index.fulltext.queryNodes('entityIndex', $query)
                YIELD node, score
                WHERE score > 0.5
                RETURN node.name AS name, node.description AS description, score
                ORDER BY score DESC
                LIMIT 10
                """,
                query=query
            )
            
            entities = []
            for record in entities_result:
                entities.append({
                    "name": record["name"],
                    "description": record["description"],
                    "score": record["score"]
                })
            
            # 查找与查询相关的事件
            events_result = session.run(
                """
                CALL db.index.fulltext.queryNodes('eventIndex', $query)
                YIELD node, score
                WHERE score > 0.5
                RETURN node.abstract AS abstract, node.summary AS summary, score
                ORDER BY score DESC
                LIMIT 10
                """,
                query=query
            )
            
            events = []
            for record in events_result:
                events.append({
                    "abstract": record["abstract"],
                    "summary": record["summary"],
                    "score": record["score"]
                })
            
            # 查找实体间的关系
            relations_result = session.run(
                """
                MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                WHERE e1.name CONTAINS $query OR e2.name CONTAINS $query
                RETURN e1.name AS source, r.type AS relation, e2.name AS target, r.description AS description
                LIMIT 20
                """,
                query=query
            )
            
            relations = []
            for record in relations_result:
                relations.append({
                    "source": record["source"],
                    "relation": record["relation"],
                    "target": record["target"],
                    "description": record["description"]
                })
            
            return {
                "entities": entities,
                "events": events,
                "relations": relations
            }

    async def _retrieve_graph_context_async(self, query: str) -> Dict[str, Any]:
        """
        异步从图数据库中检索上下文信息
        
        Args:
            query: 查询语句
            
        Returns:
            Dict: 上下文信息
        """
        # 在线程池中运行同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._retrieve_graph_context, query)

    def _construct_enhanced_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        构造增强提示词
        
        Args:
            query: 原始查询
            context: 图谱上下文信息
            
        Returns:
            str: 增强后的提示词
        """
        # 构造上下文字符串
        context_parts = []
        
        if context["entities"]:
            entities_str = "\n".join([
                f"- {e['name']}: {e['description']}" 
                for e in context["entities"]
            ])
            context_parts.append(f"相关实体:\n{entities_str}")
        
        if context["events"]:
            events_str = "\n".join([
                f"- {e['abstract']}: {e['summary']}" 
                for e in context["events"]
            ])
            context_parts.append(f"相关事件:\n{events_str}")
        
        if context["relations"]:
            relations_str = "\n".join([
                f"- {r['source']} {r['relation']} {r['target']}: {r['description']}" 
                for r in context["relations"]
            ])
            context_parts.append(f"相关关系:\n{relations_str}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "无相关上下文信息"
        
        # 构造增强提示词
        enhanced_prompt = f"""
你是一个智能助手，能够基于提供的上下文信息回答问题。

请根据以下上下文信息回答问题。如果上下文信息不足以回答问题，请说明原因。

上下文信息:
{context_str}

问题:
{query}

请提供详细且准确的回答:
"""
        
        return enhanced_prompt

    def build_knowledge_graph(self, documents: List[Dict[str, Any]]) -> bool:
        """
        构建知识图谱
        
        Args:
            documents: 文档列表，每个文档包含文本和其他元数据
            
        Returns:
            bool: 是否成功构建
        """
        try:
            with self._driver.session() as session:
                # 清空现有数据（仅用于演示，实际应用中应更谨慎）
                session.run("MATCH (n) DETACH DELETE n")
                
                # 创建全文索引
                session.run(
                    "CREATE FULLTEXT INDEX entityIndex FOR (e:Entity) ON EACH [e.name, e.description]"
                )
                session.run(
                    "CREATE FULLTEXT INDEX eventIndex FOR (e:Event) ON EACH [e.abstract, e.summary]"
                )
                
                # 处理每个文档
                for doc in documents:
                    # 提取实体和关系（简化实现）
                    entities = self._extract_entities(doc)
                    relations = self._extract_relations(doc)
                    events = self._extract_events(doc)
                    
                    # 插入实体
                    for entity in entities:
                        session.run(
                            """
                            MERGE (e:Entity {name: $name})
                            SET e.description = $description
                            """,
                            name=entity["name"],
                            description=entity.get("description", "")
                        )
                    
                    # 插入事件
                    for event in events:
                        session.run(
                            """
                            MERGE (e:Event {abstract: $abstract})
                            SET e.summary = $summary
                            """,
                            abstract=event["abstract"],
                            summary=event.get("summary", "")
                        )
                    
                    # 插入关系
                    for relation in relations:
                        session.run(
                            """
                            MATCH (e1:Entity {name: $source})
                            MATCH (e2:Entity {name: $target})
                            MERGE (e1)-[r:RELATION {type: $type}]->(e2)
                            SET r.description = $description
                            """,
                            source=relation["source"],
                            target=relation["target"],
                            type=relation["type"],
                            description=relation.get("description", "")
                        )
                
                self._logger.info(f"Successfully built knowledge graph with {len(documents)} documents")
                return True
        except Exception as e:
            self._logger.error(f"Failed to build knowledge graph: {e}")
            return False

    def _extract_entities(self, document: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        从文档中提取实体（简化实现）
        
        Args:
            document: 文档
            
        Returns:
            List[Dict]: 实体列表
        """
        # 这里应该使用实际的实体抽取模型，简化实现直接返回空列表
        return []

    def _extract_relations(self, document: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        从文档中提取关系（简化实现）
        
        Args:
            document: 文档
            
        Returns:
            List[Dict]: 关系列表
        """
        # 这里应该使用实际的关系抽取模型，简化实现直接返回空列表
        return []

    def _extract_events(self, document: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        从文档中提取事件（简化实现）
        
        Args:
            document: 文档
            
        Returns:
            List[Dict]: 事件列表
        """
        # 这里应该使用实际的事件抽取模型，简化实现直接返回空列表
        return []


# 工厂函数
def create_graphrag_adapter(
    llm_client: LLMClient,
    graph_database_uri: str = "bolt://localhost:7687",
    graph_database_auth: tuple[str, str] = ("neo4j", "password"),
    index_path: Optional[Path] = None
) -> GraphRAGAdapter:
    """
    创建 GraphRAG 适配器
    
    Args:
        llm_client: LLM 客户端
        graph_database_uri: 图数据库 URI
        graph_database_auth: 图数据库认证信息
        index_path: 索引路径
        
    Returns:
        GraphRAGAdapter: GraphRAG 适配器实例
    """
    return GraphRAGAdapter(
        llm_client=llm_client,
        graph_database_uri=graph_database_uri,
        graph_database_auth=graph_database_auth,
        index_path=index_path
    )