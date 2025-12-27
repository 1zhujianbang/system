# 存储迁移设计文档：SQLite 到 Neo4j

## 1. 概述

本项目目前使用 SQLite 作为主要的数据存储。随着数据量的增长和对复杂图查询（如路径查找、社群检测）需求的增加，需要迁移到专业的图数据库。本阶段的目标是完成图数据库选型，并设计从 SQLite 到图数据库的平滑迁移方案。

## 2. 图数据库选型

### 2.1 候选方案对比

| 特性 | Neo4j (Community) | FalkorDB (RedisGraph) |
| :--- | :--- | :--- |
| **查询语言** | Cypher (标准) | Cypher (部分支持) |
| **存储引擎** | Native Graph Storage | Redis Module (In-memory + Disk persistence) |
| **成熟度** | 非常高，文档丰富，社区活跃 | 较新，速度快，但在复杂场景下生态较弱 |
| **Python 驱动** | 官方 `neo4j` 驱动完善 | `falkordb-py` |
| **可视化** | Neo4j Browser / Bloom | RedisInsight |
| **算法库** | GDS (Graph Data Science) | GraphBLAS |
| **项目现状** | 已在 `docker-compose.yml` 中配置 | 未配置 |

### 2.2 选型决策

**决定使用 Neo4j**。
原因：
1.  **成熟稳定**：作为市场占有率第一的图数据库，遇到问题容易找到解决方案。
2.  **现有配置**：项目基础设施（Docker Compose）已经集成了 Neo4j，减少了部署成本。
3.  **Cypher 支持**：对标准 Cypher 的支持最完整，利于未来扩展。
4.  **APOC 库**：提供了强大的数据导入导出和实用工具，对迁移非常有帮助。

## 3. 架构设计

### 3.1 适配器模式

为了保持业务逻辑与底层存储解耦，我们将引入 `GraphStore` 端口和 `Neo4jAdapter` 实现。

```python
# src/ports/graph_store.py

class GraphStore(ABC):
    @abstractmethod
    def add_entity(self, entity: Entity) -> None: ...
    
    @abstractmethod
    def add_event(self, event: Event) -> None: ...
    
    @abstractmethod
    def add_relation(self, source_id: str, target_id: str, type: str, props: Dict) -> None: ...
    
    @abstractmethod
    def query(self, cypher: str, params: Dict) -> List[Dict]: ...
```

### 3.2 数据模型映射

将 SQLite 的关系型表结构映射到图模型：

#### 节点 (Nodes)

1.  **Entity**
    *   Label: `:Entity`
    *   Properties: `id` (from entity_id), `name`, `first_seen`, `last_seen`
    
2.  **Event**
    *   Label: `:Event`
    *   Properties: `id` (from event_id), `abstract`, `start_time`, `reported_at`

#### 关系 (Relationships)

1.  **参与关系 (Participants)**
    *   SQLite: `participants` table
    *   Neo4j: `(Entity)-[:PARTICIPATED_IN {roles: [...], time: ...}]->(Event)`
    *   注意：方向可以是 Entity -> Event，表示实体参与了事件。

2.  **语义关系 (Relations)**
    *   SQLite: `relations` table (subject, predicate, object, event_id)
    *   Neo4j: `(Entity)-[:RELATED_TO {predicate: ..., event_id: ..., time: ...}]->(Entity)`
    *   *备选*：如果关系强依赖于事件上下文，也可以建模为 Hyperedge，但在 Neo4j 中通常用属性或中间节点。考虑到查询便利性，直接建立实体间边并带上 `event_id` 属性是较好的折衷。

3.  **事件演化 (Event Edges)**
    *   SQLite: `event_edges` table
    *   Neo4j: `(Event)-[:EVOLVED_TO {type: ..., confidence: ...}]->(Event)`

## 4. 迁移方案

采用 **离线批量迁移** 策略，利用 CSV 作为中间格式。

### 4.1 步骤

1.  **导出 (Export)**: 编写 Python 脚本从 SQLite 读取数据，生成符合 Neo4j 导入格式的 CSV 文件。
    *   `entities.csv`: `entity_id:ID,name,first_seen,last_seen,:LABEL`
    *   `events.csv`: `event_id:ID,abstract,event_start_time,:LABEL`
    *   `participants.csv`: `:START_ID,:END_ID,roles,time,:TYPE` (Type=PARTICIPATED_IN)
    *   `relations.csv`: `:START_ID,:END_ID,predicate,event_id,time,:TYPE` (Type=RELATED_TO)
    *   `event_edges.csv`: `:START_ID,:END_ID,edge_type,confidence,:TYPE` (Type=EVOLVED_TO)

2.  **导入 (Import)**:
    *   使用 `neo4j-admin database import` (速度最快，但需要停机或挂载卷)。
    *   或者使用 `LOAD CSV` Cypher 语句 (灵活性高，适合在线导入)。
    *   考虑到 Docker 环境，使用 `LOAD CSV` 配合挂载的 `import` 目录较为方便。

3.  **验证 (Verification)**:
    *   核对节点和关系的数量。
    *   随机抽取样本查询验证属性完整性。

## 5. 实施计划

1.  **配置环境**: 确保 `docker-compose.yml` 中的 Neo4j 正常运行，并配置 APOC 插件（可选，但推荐）。
2.  **开发适配器**: 实现 `Neo4jAdapter`。
3.  **开发迁移脚本**: `scripts/migrate_sqlite_to_neo4j.py`。
4.  **执行迁移**: 在测试环境运行并验证。

## 6. 本体设计 (初步)

为了支持 OWL 本体，我们在 Neo4j 中可以使用 `neosemantics (n10s)` 插件，或者简化为 Label + Properties。目前阶段采用简化方案：

*   **Class**: 对应 Neo4j Label (e.g., `Person`, `Organization`, `Location`)。由于目前 SQLite 中实体类型未严格分表，可能统一用 `Entity`，如果 `original_forms_json` 或其他来源能推断类型，则添加额外 Label。
*   **Object Property**: 对应 Relationship Type。
*   **Data Property**: 对应 Node Property。

后续在 2.3 阶段会详细定义 OWL 本体。
