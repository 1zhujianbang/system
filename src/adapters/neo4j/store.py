"""
适配器层 - Neo4j 存储适配器

实现 UnifiedStore 端口，提供基于 Neo4j 图数据库的知识图谱存储功能。
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
from neo4j import GraphDatabase

from ...ports.store import (
    UnifiedStore, EntityStore, EventStore, RelationStore, 
    ParticipantStore, EventEdgeStore, MentionStore, ReviewStore,
    KGReadStore
)
from ...domain.models import (
    EntityCanonical, EntityMention, EventCanonical, EventMention,
    RelationTriple, Participant, EventEdge, ReviewTask, MergeDecision,
    SourceRef, ReviewTaskType, ReviewTaskStatus, MergeDecisionType
)


class Neo4jStore(UnifiedStore):
    """
    Neo4j 存储适配器：
    - 实现 UnifiedStore 接口
    - 提供完整的知识图谱 CRUD 操作
    - 支持事务和连接池
    """

    def __init__(self, uri: str, auth: tuple[str, str]):
        """
        初始化 Neo4j 存储适配器
        
        Args:
            uri: Neo4j 数据库 URI (例如: "bolt://localhost:7687")
            auth: 认证信息 (username, password)
        """
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._lock = threading.RLock()
        self._ensure_constraints()

    def close(self) -> None:
        """关闭数据库连接"""
        self._driver.close()

    def _ensure_constraints(self) -> None:
        """确保数据库约束存在"""
        with self._driver.session() as session:
            # 实体唯一约束
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
            )
            # 事件唯一约束
            session.run(
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE"
            )
            # 事件摘要唯一约束
            session.run(
                "CREATE CONSTRAINT event_abstract IF NOT EXISTS FOR (e:Event) REQUIRE e.abstract IS UNIQUE"
            )
            # 实体提及唯一约束
            session.run(
                "CREATE CONSTRAINT entity_mention_id IF NOT EXISTS FOR (m:EntityMention) REQUIRE m.mention_id IS UNIQUE"
            )
            # 事件提及唯一约束
            session.run(
                "CREATE CONSTRAINT event_mention_id IF NOT EXISTS FOR (m:EventMention) REQUIRE m.mention_id IS UNIQUE"
            )
            # 审查任务唯一约束
            session.run(
                "CREATE CONSTRAINT review_task_id IF NOT EXISTS FOR (t:ReviewTask) REQUIRE t.task_id IS UNIQUE"
            )

    # -------------------------
    # Entity Store 实现
    # -------------------------

    def get_entity_by_id(self, entity_id: str) -> Optional[EntityCanonical]:
        """根据 ID 获取实体"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                RETURN e
                """,
                entity_id=entity_id
            )
            record = result.single()
            if record:
                node = record["e"]
                return self._node_to_entity(node)
            return None

    def get_entity_by_name(self, name: str) -> Optional[EntityCanonical]:
        """根据名称获取实体"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})
                RETURN e
                """,
                name=name
            )
            record = result.single()
            if record:
                node = record["e"]
                return self._node_to_entity(node)
            return None

    def list_entities(
        self,
        limit: int = 1000,
        offset: int = 0,
        order_by: str = "first_seen",
    ) -> List[EntityCanonical]:
        """列出实体"""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity)
                RETURN e
                ORDER BY e.{order_by}
                SKIP $offset
                LIMIT $limit
                """,
                offset=offset,
                limit=limit
            )
            entities = []
            for record in result:
                node = record["e"]
                entities.append(self._node_to_entity(node))
            return entities

    def search_entities(
        self,
        query: str,
        limit: int = 100,
    ) -> List[EntityCanonical]:
        """搜索实体"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $query OR e.original_forms CONTAINS $query
                RETURN e
                LIMIT $limit
                """,
                query=query,
                limit=limit
            )
            entities = []
            for record in result:
                node = record["e"]
                entities.append(self._node_to_entity(node))
            return entities

    def upsert_entity(self, entity: EntityCanonical) -> str:
        """创建或更新实体，返回 entity_id"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.first_seen = $first_seen,
                    e.last_seen = $last_seen,
                    e.sources = $sources,
                    e.original_forms = $original_forms,
                    e.aliases = $aliases
                RETURN e.entity_id AS entity_id
                """,
                entity_id=entity.entity_id,
                name=entity.name,
                first_seen=entity.first_seen.isoformat(),
                last_seen=entity.last_seen.isoformat(),
                sources=json.dumps([s.to_dict() for s in entity.sources]),
                original_forms=json.dumps(entity.original_forms),
                aliases=json.dumps(entity.aliases)
            )
            record = result.single()
            return record["entity_id"]

    def upsert_entities(self, entities: List[EntityCanonical]) -> int:
        """批量创建或更新实体，返回成功数量"""
        count = 0
        for entity in entities:
            self.upsert_entity(entity)
            count += 1
        return count

    def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                DETACH DELETE e
                """,
                entity_id=entity_id
            )
            return result.consume().counters.nodes_deleted > 0

    def merge_entities(
        self,
        from_entity_id: str,
        to_entity_id: str,
        reason: str = "",
        decision_input_hash: str = "",
    ) -> Dict[str, Any]:
        """合并实体（from -> to）"""
        with self._driver.session() as session:
            # 更新指向源实体的关系
            session.run(
                """
                MATCH (from:Entity {entity_id: $from_id})<-[:REFERENCES]-(rel:Relation)
                MATCH (to:Entity {entity_id: $to_id})
                SET rel.subject_entity_id = $to_id
                """,
                from_id=from_entity_id,
                to_id=to_entity_id
            )
            
            session.run(
                """
                MATCH (from:Entity {entity_id: $from_id})<-[:OBJECT]-(rel:Relation)
                MATCH (to:Entity {entity_id: $to_id})
                SET rel.object_entity_id = $to_id
                """,
                from_id=from_entity_id,
                to_id=to_entity_id
            )
            
            # 合并实体属性
            session.run(
                """
                MATCH (from:Entity {entity_id: $from_id})
                MATCH (to:Entity {entity_id: $to_id})
                SET to.first_seen = CASE 
                        WHEN from.first_seen < to.first_seen 
                        THEN from.first_seen 
                        ELSE to.first_seen 
                    END,
                    to.last_seen = CASE 
                        WHEN from.last_seen > to.last_seen 
                        THEN from.last_seen 
                        ELSE to.last_seen 
                    END,
                    to.sources = to.sources + from.sources,
                    to.original_forms = to.original_forms + from.original_forms,
                    to.aliases = to.aliases + from.aliases
                """,
                from_id=from_entity_id,
                to_id=to_entity_id
            )
            
            # 删除源实体
            session.run(
                """
                MATCH (e:Entity {entity_id: $from_id})
                DETACH DELETE e
                """,
                from_id=from_entity_id
            )
            
            return {"status": "merged", "from": from_entity_id, "to": to_entity_id}

    def get_entity_aliases(self, entity_id: str) -> List[str]:
        """获取实体别名"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                RETURN e.aliases AS aliases
                """,
                entity_id=entity_id
            )
            record = result.single()
            if record:
                aliases_json = record["aliases"]
                return json.loads(aliases_json) if aliases_json else []
            return []

    def add_entity_alias(
        self,
        alias: str,
        entity_id: str,
        confidence: float = 1.0,
    ) -> bool:
        """添加实体别名"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                SET e.aliases = CASE 
                    WHEN NOT $alias IN e.aliases 
                    THEN coalesce(e.aliases, []) + $alias 
                    ELSE e.aliases 
                END
                """,
                entity_id=entity_id,
                alias=alias
            )
            return result.consume().counters.contains_updates

    def _node_to_entity(self, node: Any) -> EntityCanonical:
        """将 Neo4j 节点转换为 EntityCanonical 对象"""
        sources_data = json.loads(node.get("sources", "[]"))
        sources = [SourceRef.from_dict(s) for s in sources_data]
        
        return EntityCanonical(
            entity_id=node["entity_id"],
            name=node["name"],
            first_seen=datetime.fromisoformat(node["first_seen"]),
            last_seen=datetime.fromisoformat(node["last_seen"]),
            sources=sources,
            original_forms=json.loads(node.get("original_forms", "[]")),
            aliases=json.loads(node.get("aliases", "[]"))
        )

    # -------------------------
    # Event Store 实现
    # -------------------------

    def get_event_by_id(self, event_id: str) -> Optional[EventCanonical]:
        """根据 ID 获取事件"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event {event_id: $event_id})
                RETURN e
                """,
                event_id=event_id
            )
            record = result.single()
            if record:
                node = record["e"]
                return self._node_to_event(node)
            return None

    def get_event_by_abstract(self, abstract: str) -> Optional[EventCanonical]:
        """根据摘要获取事件"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event {abstract: $abstract})
                RETURN e
                """,
                abstract=abstract
            )
            record = result.single()
            if record:
                node = record["e"]
                return self._node_to_event(node)
            return None

    def list_events(
        self,
        limit: int = 1000,
        offset: int = 0,
        order_by: str = "first_seen",
        since: Optional[datetime] = None,
    ) -> List[EventCanonical]:
        """列出事件"""
        with self._driver.session() as session:
            if since:
                result = session.run(
                    f"""
                    MATCH (e:Event)
                    WHERE e.{order_by} >= $since
                    RETURN e
                    ORDER BY e.{order_by}
                    SKIP $offset
                    LIMIT $limit
                    """,
                    since=since.isoformat(),
                    offset=offset,
                    limit=limit
                )
            else:
                result = session.run(
                    f"""
                    MATCH (e:Event)
                    RETURN e
                    ORDER BY e.{order_by}
                    SKIP $offset
                    LIMIT $limit
                    """,
                    offset=offset,
                    limit=limit
                )
            
            events = []
            for record in result:
                node = record["e"]
                events.append(self._node_to_event(node))
            return events

    def search_events(
        self,
        query: str,
        limit: int = 100,
    ) -> List[EventCanonical]:
        """搜索事件"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)
                WHERE e.abstract CONTAINS $query OR e.event_summary CONTAINS $query
                RETURN e
                LIMIT $limit
                """,
                query=query,
                limit=limit
            )
            events = []
            for record in result:
                node = record["e"]
                events.append(self._node_to_event(node))
            return events

    def upsert_event(self, event: EventCanonical) -> str:
        """创建或更新事件，返回 event_id"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (e:Event {event_id: $event_id})
                SET e.abstract = $abstract,
                    e.event_summary = $event_summary,
                    e.event_types = $event_types,
                    e.event_start_time = $event_start_time,
                    e.event_start_time_text = $event_start_time_text,
                    e.event_start_time_precision = $event_start_time_precision,
                    e.reported_at = $reported_at,
                    e.first_seen = $first_seen,
                    e.last_seen = $last_seen,
                    e.sources = $sources,
                    e.entities = $entities,
                    e.entity_roles = $entity_roles
                RETURN e.event_id AS event_id
                """,
                event_id=event.event_id,
                abstract=event.abstract,
                event_summary=event.event_summary,
                event_types=json.dumps(event.event_types),
                event_start_time=event.event_start_time.isoformat() if event.event_start_time else None,
                event_start_time_text=event.event_start_time_text,
                event_start_time_precision=event.event_start_time_precision,
                reported_at=event.reported_at.isoformat() if event.reported_at else None,
                first_seen=event.first_seen.isoformat() if event.first_seen else None,
                last_seen=event.last_seen.isoformat() if event.last_seen else None,
                sources=json.dumps([s.to_dict() for s in event.sources]),
                entities=json.dumps(event.entities),
                entity_roles=json.dumps(event.entity_roles)
            )
            record = result.single()
            return record["event_id"]

    def upsert_events(self, events: List[EventCanonical]) -> int:
        """批量创建或更新事件，返回成功数量"""
        count = 0
        for event in events:
            self.upsert_event(event)
            count += 1
        return count

    def delete_event(self, event_id: str) -> bool:
        """删除事件"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event {event_id: $event_id})
                DETACH DELETE e
                """,
                event_id=event_id
            )
            return result.consume().counters.nodes_deleted > 0

    def merge_events(
        self,
        from_event_id: str,
        to_event_id: str,
        reason: str = "",
        decision_input_hash: str = "",
    ) -> Dict[str, Any]:
        """合并事件（from -> to）"""
        with self._driver.session() as session:
            # 更新指向源事件的关系
            session.run(
                """
                MATCH (from:Event {event_id: $from_id})<-[:PART_OF]-(part:Participant)
                MATCH (to:Event {event_id: $to_id})
                SET part.event_id = $to_id
                """,
                from_id=from_event_id,
                to_id=to_event_id
            )
            
            # 合并事件属性
            session.run(
                """
                MATCH (from:Event {event_id: $from_id})
                MATCH (to:Event {event_id: $to_id})
                SET to.event_summary = CASE 
                        WHEN size(to.event_summary) < size(from.event_summary) 
                        THEN from.event_summary 
                        ELSE to.event_summary 
                    END,
                    to.event_types = apoc.coll.union(to.event_types, from.event_types),
                    to.sources = to.sources + from.sources,
                    to.entities = apoc.coll.union(to.entities, from.entities),
                    to.entity_roles = apoc.map.merge(to.entity_roles, from.entity_roles),
                    to.first_seen = CASE 
                        WHEN from.first_seen < to.first_seen 
                        THEN from.first_seen 
                        ELSE to.first_seen 
                    END,
                    to.last_seen = CASE 
                        WHEN from.last_seen > to.last_seen 
                        THEN from.last_seen 
                        ELSE to.last_seen 
                    END
                """,
                from_id=from_event_id,
                to_id=to_event_id
            )
            
            # 删除源事件
            session.run(
                """
                MATCH (e:Event {event_id: $from_id})
                DETACH DELETE e
                """,
                from_id=from_event_id
            )
            
            return {"status": "merged", "from": from_event_id, "to": to_event_id}

    def _node_to_event(self, node: Any) -> EventCanonical:
        """将 Neo4j 节点转换为 EventCanonical 对象"""
        sources_data = json.loads(node.get("sources", "[]"))
        sources = [SourceRef.from_dict(s) for s in sources_data]
        
        event_start_time = None
        if node.get("event_start_time"):
            event_start_time = datetime.fromisoformat(node["event_start_time"])
            
        reported_at = None
        if node.get("reported_at"):
            reported_at = datetime.fromisoformat(node["reported_at"])
            
        first_seen = None
        if node.get("first_seen"):
            first_seen = datetime.fromisoformat(node["first_seen"])
            
        last_seen = None
        if node.get("last_seen"):
            last_seen = datetime.fromisoformat(node["last_seen"])
        
        return EventCanonical(
            event_id=node["event_id"],
            abstract=node["abstract"],
            event_summary=node["event_summary"],
            event_types=json.loads(node.get("event_types", "[]")),
            event_start_time=event_start_time,
            event_start_time_text=node.get("event_start_time_text", ""),
            event_start_time_precision=node.get("event_start_time_precision", "unknown"),
            reported_at=reported_at,
            first_seen=first_seen,
            last_seen=last_seen,
            sources=sources,
            entities=json.loads(node.get("entities", "[]")),
            entity_roles=json.loads(node.get("entity_roles", "{}"))
        )

    # -------------------------
    # Relation Store 实现
    # -------------------------

    def get_relations_by_entity(
        self,
        entity_id: str,
        direction: str = "both",
        limit: int = 100,
    ) -> List[RelationTriple]:
        """获取实体相关的关系"""
        with self._driver.session() as session:
            if direction == "subject":
                query = """
                    MATCH (e:Entity {entity_id: $entity_id})-[:SUBJECT]->(r:Relation)
                    RETURN r
                    LIMIT $limit
                """
            elif direction == "object":
                query = """
                    MATCH (e:Entity {entity_id: $entity_id})<-[:OBJECT]-(r:Relation)
                    RETURN r
                    LIMIT $limit
                """
            else:  # both
                query = """
                    MATCH (e:Entity {entity_id: $entity_id})-[:SUBJECT|OBJECT]-(r:Relation)
                    RETURN r
                    LIMIT $limit
                """
            
            result = session.run(query, entity_id=entity_id, limit=limit)
            relations = []
            for record in result:
                node = record["r"]
                relations.append(self._node_to_relation(node))
            return relations

    def get_relations_by_event(
        self,
        event_id: str,
        limit: int = 100,
    ) -> List[RelationTriple]:
        """获取事件相关的关系"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event {event_id: $event_id})<-[:PART_OF]-(r:Relation)
                RETURN r
                LIMIT $limit
                """,
                event_id=event_id,
                limit=limit
            )
            relations = []
            for record in result:
                node = record["r"]
                relations.append(self._node_to_relation(node))
            return relations

    def upsert_relation(self, relation: RelationTriple) -> int:
        """创建或更新关系，返回 ID"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (r:Relation {id: $id})
                ON CREATE SET r.id = toString(id($r))
                SET r.event_id = $event_id,
                    r.subject_entity_id = $subject_entity_id,
                    r.predicate = $predicate,
                    r.object_entity_id = $object_entity_id,
                    r.time = $time,
                    r.reported_at = $reported_at,
                    r.evidence = $evidence
                WITH r
                MATCH (subj:Entity {entity_id: $subject_entity_id})
                MATCH (obj:Entity {entity_id: $object_entity_id})
                MATCH (evt:Event {event_id: $event_id})
                MERGE (subj)-[:SUBJECT]->(r)
                MERGE (r)-[:OBJECT]->(obj)
                MERGE (r)-[:PART_OF]->(evt)
                RETURN id(r) AS relation_id
                """,
                id=relation.id,
                event_id=relation.event_id,
                subject_entity_id=relation.subject_entity_id,
                predicate=relation.predicate,
                object_entity_id=relation.object_entity_id,
                time=relation.time.isoformat() if relation.time else None,
                reported_at=relation.reported_at.isoformat() if relation.reported_at else None,
                evidence=json.dumps(relation.evidence)
            )
            record = result.single()
            return record["relation_id"]

    def upsert_relations(self, relations: List[RelationTriple]) -> int:
        """批量创建或更新关系，返回成功数量"""
        count = 0
        for relation in relations:
            self.upsert_relation(relation)
            count += 1
        return count

    def delete_relation(self, relation_id: int) -> bool:
        """删除关系"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation) WHERE id(r) = $relation_id
                DETACH DELETE r
                """,
                relation_id=relation_id
            )
            return result.consume().counters.nodes_deleted > 0

    def _node_to_relation(self, node: Any) -> RelationTriple:
        """将 Neo4j 节点转换为 RelationTriple 对象"""
        time = None
        if node.get("time"):
            time = datetime.fromisoformat(node["time"])
            
        reported_at = None
        if node.get("reported_at"):
            reported_at = datetime.fromisoformat(node["reported_at"])
        
        return RelationTriple(
            id=node.id,
            event_id=node["event_id"],
            subject_entity_id=node["subject_entity_id"],
            predicate=node["predicate"],
            object_entity_id=node["object_entity_id"],
            time=time,
            reported_at=reported_at,
            evidence=json.loads(node.get("evidence", "[]"))
        )

    # -------------------------
    # Participant Store 实现
    # -------------------------

    def get_participants_by_event(self, event_id: str) -> List[Participant]:
        """获取事件的参与者"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event {event_id: $event_id})<-[:PARTICIPATES_IN]-(p:Participant)
                RETURN p
                """,
                event_id=event_id
            )
            participants = []
            for record in result:
                node = record["p"]
                participants.append(self._node_to_participant(node))
            return participants

    def get_participants_by_entity(self, entity_id: str) -> List[Participant]:
        """获取实体参与的记录"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})-[:PARTICIPATES]->(p:Participant)
                RETURN p
                """,
                entity_id=entity_id
            )
            participants = []
            for record in result:
                node = record["p"]
                participants.append(self._node_to_participant(node))
            return participants

    def upsert_participant(self, participant: Participant) -> int:
        """创建或更新参与记录"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (p:Participant {id: $id})
                ON CREATE SET p.id = toString(id($p))
                SET p.event_id = $event_id,
                    p.entity_id = $entity_id,
                    p.roles = $roles,
                    p.time = $time,
                    p.reported_at = $reported_at
                WITH p
                MATCH (ent:Entity {entity_id: $entity_id})
                MATCH (evt:Event {event_id: $event_id})
                MERGE (ent)-[:PARTICIPATES]->(p)
                MERGE (p)-[:PARTICIPATES_IN]->(evt)
                RETURN id(p) AS participant_id
                """,
                id=participant.id,
                event_id=participant.event_id,
                entity_id=participant.entity_id,
                roles=json.dumps(participant.roles),
                time=participant.time.isoformat() if participant.time else None,
                reported_at=participant.reported_at.isoformat() if participant.reported_at else None
            )
            record = result.single()
            return record["participant_id"]

    def upsert_participants(self, participants: List[Participant]) -> int:
        """批量创建或更新参与记录"""
        count = 0
        for participant in participants:
            self.upsert_participant(participant)
            count += 1
        return count

    def _node_to_participant(self, node: Any) -> Participant:
        """将 Neo4j 节点转换为 Participant 对象"""
        time = None
        if node.get("time"):
            time = datetime.fromisoformat(node["time"])
            
        reported_at = None
        if node.get("reported_at"):
            reported_at = datetime.fromisoformat(node["reported_at"])
        
        return Participant(
            id=node.id,
            event_id=node["event_id"],
            entity_id=node["entity_id"],
            roles=json.loads(node.get("roles", "[]")),
            time=time,
            reported_at=reported_at
        )

    # -------------------------
    # Event Edge Store 实现
    # -------------------------

    def get_edges_by_event(
        self,
        event_id: str,
        direction: str = "both",
    ) -> List[EventEdge]:
        """获取事件相关的边"""
        with self._driver.session() as session:
            if direction == "from":
                query = """
                    MATCH (e:Event {event_id: $event_id})-[:FROM_EVENT]->(edge:EventEdge)
                    RETURN edge
                """
            elif direction == "to":
                query = """
                    MATCH (e:Event {event_id: $event_id})<-[:TO_EVENT]-(edge:EventEdge)
                    RETURN edge
                """
            else:  # both
                query = """
                    MATCH (e:Event {event_id: $event_id})-[:FROM_EVENT|TO_EVENT]-(edge:EventEdge)
                    RETURN edge
                """
            
            result = session.run(query, event_id=event_id)
            edges = []
            for record in result:
                node = record["edge"]
                edges.append(self._node_to_event_edge(node))
            return edges

    def list_edges(
        self,
        limit: int = 1000,
        since: Optional[datetime] = None,
    ) -> List[EventEdge]:
        """列出所有边"""
        with self._driver.session() as session:
            if since:
                result = session.run(
                    """
                    MATCH (e:EventEdge)
                    WHERE e.time >= $since
                    RETURN e
                    ORDER BY e.time
                    LIMIT $limit
                    """,
                    since=since.isoformat(),
                    limit=limit
                )
            else:
                result = session.run(
                    """
                    MATCH (e:EventEdge)
                    RETURN e
                    ORDER BY e.time
                    LIMIT $limit
                    """,
                    limit=limit
                )
            
            edges = []
            for record in result:
                node = record["e"]
                edges.append(self._node_to_event_edge(node))
            return edges

    def upsert_edge(self, edge: EventEdge) -> int:
        """创建或更新边"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (e:EventEdge {id: $id})
                ON CREATE SET e.id = toString(id($e))
                SET e.from_event_id = $from_event_id,
                    e.to_event_id = $to_event_id,
                    e.edge_type = $edge_type,
                    e.time = $time,
                    e.reported_at = $reported_at,
                    e.confidence = $confidence,
                    e.evidence = $evidence,
                    e.decision_input_hash = $decision_input_hash
                WITH e
                MATCH (from:Event {event_id: $from_event_id})
                MATCH (to:Event {event_id: $to_event_id})
                MERGE (from)-[:FROM_EVENT]->(e)
                MERGE (e)-[:TO_EVENT]->(to)
                RETURN id(e) AS edge_id
                """,
                id=edge.id,
                from_event_id=edge.from_event_id,
                to_event_id=edge.to_event_id,
                edge_type=edge.edge_type.value,
                time=edge.time.isoformat() if edge.time else None,
                reported_at=edge.reported_at.isoformat() if edge.reported_at else None,
                confidence=edge.confidence,
                evidence=json.dumps(edge.evidence),
                decision_input_hash=edge.decision_input_hash
            )
            record = result.single()
            return record["edge_id"]

    def upsert_edges(self, edges: List[EventEdge]) -> int:
        """批量创建或更新边"""
        count = 0
        for edge in edges:
            self.upsert_edge(edge)
            count += 1
        return count

    def _node_to_event_edge(self, node: Any) -> EventEdge:
        """将 Neo4j 节点转换为 EventEdge 对象"""
        time = None
        if node.get("time"):
            time = datetime.fromisoformat(node["time"])
            
        reported_at = None
        if node.get("reported_at"):
            reported_at = datetime.fromisoformat(node["reported_at"])
        
        from ...domain.models import EventEdgeType
        edge_type = EventEdgeType(node.get("edge_type", "related"))
        
        return EventEdge(
            id=node.id,
            from_event_id=node["from_event_id"],
            to_event_id=node["to_event_id"],
            edge_type=edge_type,
            time=time,
            reported_at=reported_at,
            confidence=float(node.get("confidence", 0.0)),
            evidence=json.loads(node.get("evidence", "[]")),
            decision_input_hash=node.get("decision_input_hash", "")
        )

    # -------------------------
    # Mention Store 实现
    # -------------------------

    def get_entity_mention(self, mention_id: str) -> Optional[EntityMention]:
        """获取实体提及"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (m:EntityMention {mention_id: $mention_id})
                RETURN m
                """,
                mention_id=mention_id
            )
            record = result.single()
            if record:
                node = record["m"]
                return self._node_to_entity_mention(node)
            return None

    def list_entity_mentions(
        self,
        resolved_entity_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[EntityMention]:
        """列出实体提及"""
        with self._driver.session() as session:
            if resolved_entity_id and since:
                result = session.run(
                    """
                    MATCH (m:EntityMention)
                    WHERE m.resolved_entity_id = $resolved_entity_id AND m.reported_at >= $since
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    resolved_entity_id=resolved_entity_id,
                    since=since.isoformat(),
                    limit=limit
                )
            elif resolved_entity_id:
                result = session.run(
                    """
                    MATCH (m:EntityMention {resolved_entity_id: $resolved_entity_id})
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    resolved_entity_id=resolved_entity_id,
                    limit=limit
                )
            elif since:
                result = session.run(
                    """
                    MATCH (m:EntityMention)
                    WHERE m.reported_at >= $since
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    since=since.isoformat(),
                    limit=limit
                )
            else:
                result = session.run(
                    """
                    MATCH (m:EntityMention)
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    limit=limit
                )
            
            mentions = []
            for record in result:
                node = record["m"]
                mentions.append(self._node_to_entity_mention(node))
            return mentions

    def upsert_entity_mention(self, mention: EntityMention) -> str:
        """创建或更新实体提及"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (m:EntityMention {mention_id: $mention_id})
                SET m.name_text = $name_text,
                    m.reported_at = $reported_at,
                    m.source = $source,
                    m.resolved_entity_id = $resolved_entity_id,
                    m.confidence = $confidence,
                    m.created_at = $created_at
                RETURN m.mention_id AS mention_id
                """,
                mention_id=mention.mention_id,
                name_text=mention.name_text,
                reported_at=mention.reported_at.isoformat(),
                source=json.dumps(mention.source.to_dict()),
                resolved_entity_id=mention.resolved_entity_id,
                confidence=mention.confidence,
                created_at=mention.created_at.isoformat() if mention.created_at else None
            )
            record = result.single()
            return record["mention_id"]

    def get_event_mention(self, mention_id: str) -> Optional[EventMention]:
        """获取事件提及"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (m:EventMention {mention_id: $mention_id})
                RETURN m
                """,
                mention_id=mention_id
            )
            record = result.single()
            if record:
                node = record["m"]
                return self._node_to_event_mention(node)
            return None

    def list_event_mentions(
        self,
        resolved_event_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[EventMention]:
        """列出事件提及"""
        with self._driver.session() as session:
            if resolved_event_id and since:
                result = session.run(
                    """
                    MATCH (m:EventMention)
                    WHERE m.resolved_event_id = $resolved_event_id AND m.reported_at >= $since
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    resolved_event_id=resolved_event_id,
                    since=since.isoformat(),
                    limit=limit
                )
            elif resolved_event_id:
                result = session.run(
                    """
                    MATCH (m:EventMention {resolved_event_id: $resolved_event_id})
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    resolved_event_id=resolved_event_id,
                    limit=limit
                )
            elif since:
                result = session.run(
                    """
                    MATCH (m:EventMention)
                    WHERE m.reported_at >= $since
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    since=since.isoformat(),
                    limit=limit
                )
            else:
                result = session.run(
                    """
                    MATCH (m:EventMention)
                    RETURN m
                    ORDER BY m.reported_at
                    LIMIT $limit
                    """,
                    limit=limit
                )
            
            mentions = []
            for record in result:
                node = record["m"]
                mentions.append(self._node_to_event_mention(node))
            return mentions

    def upsert_event_mention(self, mention: EventMention) -> str:
        """创建或更新事件提及"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (m:EventMention {mention_id: $mention_id})
                SET m.abstract_text = $abstract_text,
                    m.reported_at = $reported_at,
                    m.source = $source,
                    m.resolved_event_id = $resolved_event_id,
                    m.confidence = $confidence,
                    m.created_at = $created_at
                RETURN m.mention_id AS mention_id
                """,
                mention_id=mention.mention_id,
                abstract_text=mention.abstract_text,
                reported_at=mention.reported_at.isoformat(),
                source=json.dumps(mention.source.to_dict()),
                resolved_event_id=mention.resolved_event_id,
                confidence=mention.confidence,
                created_at=mention.created_at.isoformat() if mention.created_at else None
            )
            record = result.single()
            return record["mention_id"]

    def _node_to_entity_mention(self, node: Any) -> EntityMention:
        """将 Neo4j 节点转换为 EntityMention 对象"""
        source_data = json.loads(node.get("source", "{}"))
        source = SourceRef.from_dict(source_data)
        
        return EntityMention(
            mention_id=node["mention_id"],
            name_text=node["name_text"],
            reported_at=datetime.fromisoformat(node["reported_at"]),
            source=source,
            resolved_entity_id=node.get("resolved_entity_id"),
            confidence=float(node.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else None
        )

    def _node_to_event_mention(self, node: Any) -> EventMention:
        """将 Neo4j 节点转换为 EventMention 对象"""
        source_data = json.loads(node.get("source", "{}"))
        source = SourceRef.from_dict(source_data)
        
        return EventMention(
            mention_id=node["mention_id"],
            abstract_text=node["abstract_text"],
            reported_at=datetime.fromisoformat(node["reported_at"]),
            source=source,
            resolved_event_id=node.get("resolved_event_id"),
            confidence=float(node.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else None
        )

    # -------------------------
    # Review Store 实现
    # -------------------------

    def enqueue_review_task(
        self,
        task_type: ReviewTaskType,
        payload: Dict[str, Any],
        priority: int = 50,
    ) -> int:
        """入队审查任务，返回 task_id"""
        with self._driver.session() as session:
            result = session.run(
                """
                CREATE (t:ReviewTask {
                    task_id: toString(randomUUID()),
                    type: $type,
                    input_hash: $input_hash,
                    payload: $payload,
                    status: $status,
                    priority: $priority,
                    created_at: $created_at,
                    updated_at: $updated_at
                })
                RETURN id(t) AS task_id
                """,
                type=task_type.value,
                input_hash=self._hash_payload({"type": task_type.value, "payload": payload}),
                payload=json.dumps(payload),
                status=ReviewTaskStatus.PENDING.value,
                priority=priority,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            record = result.single()
            return record["task_id"]

    def claim_next_task(
        self,
        task_type: Optional[ReviewTaskType] = None,
    ) -> Optional[ReviewTask]:
        """领取下一个待处理任务"""
        with self._driver.session() as session:
            if task_type:
                result = session.run(
                    """
                    MATCH (t:ReviewTask {status: $pending_status, type: $type})
                    SET t.status = $running_status,
                        t.updated_at = $updated_at,
                        t.claimed_at = $claimed_at
                    RETURN t
                    ORDER BY t.priority DESC, t.created_at ASC
                    LIMIT 1
                    """,
                    pending_status=ReviewTaskStatus.PENDING.value,
                    running_status=ReviewTaskStatus.RUNNING.value,
                    type=task_type.value,
                    updated_at=datetime.now().isoformat(),
                    claimed_at=datetime.now().isoformat()
                )
            else:
                result = session.run(
                    """
                    MATCH (t:ReviewTask {status: $pending_status})
                    SET t.status = $running_status,
                        t.updated_at = $updated_at,
                        t.claimed_at = $claimed_at
                    RETURN t
                    ORDER BY t.priority DESC, t.created_at ASC
                    LIMIT 1
                    """,
                    pending_status=ReviewTaskStatus.PENDING.value,
                    running_status=ReviewTaskStatus.RUNNING.value,
                    updated_at=datetime.now().isoformat(),
                    claimed_at=datetime.now().isoformat()
                )
            
            record = result.single()
            if record:
                node = record["t"]
                return self._node_to_review_task(node)
            return None

    def complete_task(
        self,
        task_id: int,
        status: ReviewTaskStatus,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        model: str = "",
        prompt_version: str = "",
    ) -> bool:
        """完成任务"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (t:ReviewTask)
                WHERE id(t) = $task_id
                SET t.status = $status,
                    t.output = $output,
                    t.error = $error,
                    t.model = $model,
                    t.prompt_version = $prompt_version,
                    t.updated_at = $updated_at
                RETURN count(t) AS count
                """,
                task_id=task_id,
                status=status.value,
                output=json.dumps(output) if output else None,
                error=error,
                model=model,
                prompt_version=prompt_version,
                updated_at=datetime.now().isoformat()
            )
            record = result.single()
            return record["count"] > 0

    def requeue_stale_tasks(
        self,
        max_age_minutes: int = 10,
    ) -> int:
        """重新入队超时任务"""
        cutoff = (datetime.now() - timedelta(minutes=max_age_minutes)).isoformat()
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (t:ReviewTask {status: $running_status})
                WHERE t.updated_at < $cutoff
                SET t.status = $pending_status,
                    t.updated_at = $updated_at
                RETURN count(t) AS count
                """,
                running_status=ReviewTaskStatus.RUNNING.value,
                pending_status=ReviewTaskStatus.PENDING.value,
                cutoff=cutoff,
                updated_at=datetime.now().isoformat()
            )
            record = result.single()
            return record["count"]

    def get_task_stats(self) -> Dict[str, int]:
        """获取任务统计"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (t:ReviewTask)
                RETURN t.status AS status, count(t) AS count
                """
            )
            stats = {}
            for record in result:
                stats[record["status"]] = record["count"]
            return stats

    def upsert_merge_decision(
        self,
        decision: MergeDecision,
    ) -> int:
        """创建或更新合并决策"""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (d:MergeDecision {input_hash: $input_hash})
                SET d.type = $type,
                    d.output = $output,
                    d.model = $model,
                    d.prompt_version = $prompt_version,
                    d.created_at = $created_at
                RETURN id(d) AS decision_id
                """,
                input_hash=decision.input_hash,
                type=decision.type.value,
                output=json.dumps(decision.output),
                model=decision.model,
                prompt_version=decision.prompt_version,
                created_at=decision.created_at.isoformat() if decision.created_at else datetime.now().isoformat()
            )
            record = result.single()
            return record["decision_id"]

    def get_merge_decision_by_hash(
        self,
        input_hash: str,
    ) -> Optional[MergeDecision]:
        """根据 input_hash 获取决策"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (d:MergeDecision {input_hash: $input_hash})
                RETURN d
                """,
                input_hash=input_hash
            )
            record = result.single()
            if record:
                node = record["d"]
                return self._node_to_merge_decision(node)
            return None

    def list_merge_decisions(
        self,
        decision_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MergeDecision]:
        """列出合并决策"""
        with self._driver.session() as session:
            if decision_type:
                result = session.run(
                    """
                    MATCH (d:MergeDecision {type: $type})
                    RETURN d
                    ORDER BY d.created_at DESC
                    LIMIT $limit
                    """,
                    type=decision_type,
                    limit=limit
                )
            else:
                result = session.run(
                    """
                    MATCH (d:MergeDecision)
                    RETURN d
                    ORDER BY d.created_at DESC
                    LIMIT $limit
                    """,
                    limit=limit
                )
            
            decisions = []
            for record in result:
                node = record["d"]
                decisions.append(self._node_to_merge_decision(node))
            return decisions

    def _node_to_review_task(self, node: Any) -> ReviewTask:
        """将 Neo4j 节点转换为 ReviewTask 对象"""
        task_type = ReviewTaskType(node.get("type", "entity_merge_review"))
        status = ReviewTaskStatus(node.get("status", "pending"))
        
        created_at = None
        if node.get("created_at"):
            created_at = datetime.fromisoformat(node["created_at"])
            
        updated_at = None
        if node.get("updated_at"):
            updated_at = datetime.fromisoformat(node["updated_at"])
            
        claimed_at = None
        if node.get("claimed_at"):
            claimed_at = datetime.fromisoformat(node["claimed_at"])
        
        return ReviewTask(
            task_id=node.id,
            type=task_type,
            input_hash=node["input_hash"],
            payload=json.loads(node.get("payload", "{}")),
            status=status,
            priority=int(node.get("priority", 50)),
            created_at=created_at,
            updated_at=updated_at,
            claimed_at=claimed_at,
            output=json.loads(node.get("output", "null")) if node.get("output") else None,
            error=node.get("error"),
            model=node.get("model", ""),
            prompt_version=node.get("prompt_version", "")
        )

    def _node_to_merge_decision(self, node: Any) -> MergeDecision:
        """将 Neo4j 节点转换为 MergeDecision 对象"""
        decision_type = MergeDecisionType(node.get("type", "entity_merge"))
        
        created_at = None
        if node.get("created_at"):
            created_at = datetime.fromisoformat(node["created_at"])
        
        return MergeDecision(
            decision_id=node.id,
            type=decision_type,
            input_hash=node["input_hash"],
            output=json.loads(node.get("output", "{}")),
            model=node.get("model", ""),
            prompt_version=node.get("prompt_version", ""),
            created_at=created_at
        )

    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """计算负载的哈希值"""
        try:
            import hashlib
            s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            return hashlib.sha1(f"payload:{s}".encode("utf-8")).hexdigest()
        except Exception:
            import uuid
            return str(uuid.uuid4())

    # -------------------------
    # KGReadStore 实现
    # -------------------------

    def fetch_entities(self) -> List[Dict[str, Any]]:
        """获取所有实体"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e
                """
            )
            entities = []
            for record in result:
                node = record["e"]
                entities.append(dict(node))
            return entities

    def fetch_events(self) -> List[Dict[str, Any]]:
        """获取所有事件"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)
                RETURN e
                """
            )
            events = []
            for record in result:
                node = record["e"]
                events.append(dict(node))
            return events

    def fetch_participants_with_events(self) -> List[Dict[str, Any]]:
        """获取参与关系（带事件信息）"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)<-[:PARTICIPATES_IN]-(p:Participant)-[:PARTICIPATES]->(ent:Entity)
                RETURN e.event_id AS event_id, e.abstract AS event_abstract,
                       ent.entity_id AS entity_id, ent.name AS entity_name,
                       p.roles AS roles, p.time AS time
                """
            )
            participants = []
            for record in result:
                participants.append(dict(record))
            return participants

    def fetch_relations(self) -> List[Dict[str, Any]]:
        """获取关系三元组"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (subj:Entity)-[:SUBJECT]->(r:Relation)-[:OBJECT]->(obj:Entity)
                MATCH (r)-[:PART_OF]->(e:Event)
                RETURN subj.entity_id AS subject_id, subj.name AS subject_name,
                       r.predicate AS predicate,
                       obj.entity_id AS object_id, obj.name AS object_name,
                       e.event_id AS event_id, e.abstract AS event_abstract,
                       r.time AS time, r.evidence AS evidence
                """
            )
            relations = []
            for record in result:
                relations.append(dict(record))
            return relations

    def fetch_event_edges(self) -> List[Dict[str, Any]]:
        """获取事件演化边"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (from:Event)-[:FROM_EVENT]->(e:EventEdge)<-[:TO_EVENT]-(to:Event)
                RETURN from.event_id AS from_event_id, from.abstract AS from_abstract,
                       to.event_id AS to_event_id, to.abstract AS to_abstract,
                       e.edge_type AS edge_type, e.time AS time,
                       e.confidence AS confidence, e.evidence AS evidence
                """
            )
            edges = []
            for record in result:
                edges.append(dict(record))
            return edges

    # -------------------------
    # UnifiedStore 特有方法
    # -------------------------

    def get_schema_version(self) -> str:
        """获取当前 schema 版本"""
        return "1.0.0"

    def export_compat_json_files(self) -> Dict[str, str]:
        """导出兼容 JSON 文件"""
        # 这里应该实现导出逻辑，但为了简化，我们返回空字典
        return {}

    def export_entities_json(self) -> Dict[str, Any]:
        """导出实体 JSON"""
        entities = self.fetch_entities()
        return {entity["name"]: entity for entity in entities}

    def export_abstract_map_json(self) -> Dict[str, Any]:
        """导出事件摘要映射 JSON"""
        events = self.fetch_events()
        return {event["abstract"]: event for event in events}

    def begin_transaction(self) -> Any:
        """开始事务"""
        return self._driver.session()

    def commit_transaction(self, tx: Any) -> None:
        """提交事务"""
        if hasattr(tx, 'close'):
            tx.close()

    def rollback_transaction(self, tx: Any) -> None:
        """回滚事务"""
        if hasattr(tx, 'close'):
            tx.close()


# 单例模式支持
_store_singleton: Optional[Neo4jStore] = None
_store_lock = threading.Lock()


def get_neo4j_store(uri: str = "bolt://localhost:7687", auth: tuple[str, str] = ("neo4j", "password")) -> Neo4jStore:
    """获取 Neo4j 存储实例（单例）"""
    global _store_singleton
    if _store_singleton is None:
        with _store_lock:
            if _store_singleton is None:
                _store_singleton = Neo4jStore(uri, auth)
    return _store_singleton