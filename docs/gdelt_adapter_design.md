# GDELT 适配器设计文档

## 1. 概述

### 1.1 设计目标

本文档描述如何为项目设计并实现一个 GDELT（Global Database of Events, Language, and Tone）数据适配器。该适配器将 GDELT 数据源集成到现有的新闻数据采集架构中，使其能够与 GNews 等其他新闻源统一管理和使用。

### 1.2 GDELT 简介

GDELT 是全球最大的开放数据库之一，监控全球超过100种语言的新闻媒体，提供：
- **GDELT 1.0**: 每日更新的数据集，包含 'events' 和 'gkg' 表
- **GDELT 2.0**: 每15分钟更新的数据集，包含 'events'、'gkg' 和 'mentions' 表
- 历史数据可追溯到1979年1月1日
- 支持多种输出格式：pandas dataframe、CSV、JSON、GeoPandas dataframe 等

### 1.3 项目架构背景

项目采用领域驱动设计（DDD）和分层架构：
- **适配器层** (`src/adapters/`): 负责外部系统集成
- **端口层** (`src/ports/`): 定义抽象接口
- **应用层** (`src/app/`): 业务逻辑编排
- **领域层** (`src/domain/`): 核心业务模型

现有新闻适配器已实现 `NewsSource` 接口，GDELT 适配器需要遵循相同的设计模式。

## 2. 架构设计

### 2.1 适配器位置

```
src/adapters/news/
├── __init__.py
├── api_manager.py          # NewsAPIManager（新闻源池）
├── fetch_utils.py          # 公共工具函数
└── gdelt_adapter.py        # GDELT 适配器实现（新增）
```

### 2.2 接口实现

GDELT 适配器需要实现 `NewsSource` 接口（定义在 `src/ports/extraction.py`）：

```python
class NewsSource(ABC):
    @property
    @abstractmethod
    def source_type(self) -> NewsSourceType
    
    @property
    @abstractmethod
    def source_name(self) -> str
    
    @abstractmethod
    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult
    
    @abstractmethod
    async def fetch_stream(self, config: Optional[FetchConfig] = None) -> AsyncIterator[NewsItem]
    
    @abstractmethod
    def is_available(self) -> bool
```

### 2.3 数据流设计

```
用户请求 (FetchConfig)
    ↓
GDELTAdapter.fetch()
    ↓
gdeltPyR.Search() [同步调用]
    ↓
数据转换 (GDELT → NewsItem)
    ↓
FetchResult (统一格式)
    ↓
NewsAPIManager 统一管理
```

## 3. 核心设计决策

### 3.1 版本选择策略

- **默认版本**: GDELT 2.0（更实时、数据更丰富）
- **降级策略**: 如果 2.0 数据不可用，自动降级到 1.0
- **配置支持**: 允许通过配置强制指定版本

### 3.2 表选择策略

GDELT 提供三种表类型：
- **events**: 事件表（推荐用于新闻采集）
- **gkg**: Global Knowledge Graph（知识图谱数据）
- **mentions**: 提及表（2.0 特有）

适配器默认使用 **events** 表，因为其结构与新闻数据最匹配。

### 3.3 日期处理策略

- **单日期**: 转换为 GDELT 支持的日期格式（如 "2016 Nov 01"）
- **日期范围**: 支持 `from_date` 和 `to_date`，转换为列表格式
- **coverage 参数**: 
  - GDELT 2.0: 如果 `coverage=True`，拉取所有15分钟间隔的数据
  - GDELT 1.0: 忽略 coverage 参数（每日数据）

### 3.4 输出格式选择

- **默认**: pandas dataframe（便于处理）
- **转换**: 将 dataframe 转换为 `NewsItem` 列表
- **性能优化**: 对于大数据集，考虑流式处理或分批处理

## 4. 实现细节

### 4.1 类结构设计

```python
class GDELTAdapter(NewsSource):
    """GDELT 新闻源适配器"""
    
    def __init__(
        self,
        version: int = 2,
        table: str = "events",
        name: str = "GDELT",
        coverage: bool = False,
        translation: bool = True,
        output_format: str = "pandas"
    ):
        # 初始化 gdelt 客户端
        # 设置配置参数
        
    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.CUSTOM  # 或新增 GDELT 类型
        
    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult:
        # 1. 转换 FetchConfig 为 GDELT 参数
        # 2. 调用 gdelt.Search()
        # 3. 转换结果为 NewsItem 列表
        # 4. 返回 FetchResult
        
    def _convert_to_news_items(self, gdelt_data: pd.DataFrame) -> List[NewsItem]:
        # 将 GDELT 数据转换为 NewsItem
```

### 4.2 数据映射策略

GDELT events 表字段到 NewsItem 的映射：

| GDELT 字段 | NewsItem 字段 | 说明 |
|-----------|--------------|------|
| SQLDATE | published_at | 日期（需要转换格式） |
| SourceURL | source_url | 新闻源URL |
| ArticleURL | source_url (备用) | 文章URL |
| Actor1Name / Actor2Name | - | 参与者（存入 metadata） |
| EventCode | category | 事件代码（需要转换） |
| EventBaseCode | - | 基础事件代码 |
| ActionGeo_Lat / ActionGeo_Long | - | 地理位置（存入 metadata） |
| NumMentions | - | 提及次数（存入 metadata） |
| AvgTone | - | 平均语调（存入 metadata） |
| - | title | 从 ArticleURL 或 SourceURL 提取 |
| - | content | 需要从 URL 抓取或使用摘要字段 |

**注意**: GDELT 不直接提供标题和内容，需要：
1. 从 URL 抓取（可选，性能开销大）
2. 使用 GDELT 的摘要字段（如 gkg 表的 V2Tone）
3. 仅提供元数据，让后续流程处理

### 4.3 异步处理策略

gdeltPyR 的 `Search()` 方法是同步的，需要：
1. 使用 `asyncio.to_thread()` 或 `run_in_executor()` 在异步环境中调用
2. 对于大数据集，考虑分批处理
3. 实现超时机制，避免长时间阻塞

### 4.4 错误处理策略

- **数据缺失**: GDELT 可能返回空结果，需要检查并返回适当的错误信息
- **网络错误**: 捕获 gdeltPyR 的网络异常，转换为 FetchResult 错误
- **版本兼容**: 如果 2.0 不可用，自动降级到 1.0
- **内存限制**: 对于大数据集，提供警告和建议

## 5. 配置与集成

### 5.1 注册到 NewsAPIManager

在 `api_manager.py` 的 `load_from_env()` 方法中注册：

```python
# 注册 GDELT 数据源
gdelt_source = GDELTAdapter(
    version=2,
    table="events",
    name="GDELT",
    coverage=False
)
self.register(gdelt_source)
```

### 5.2 环境变量配置（可选）

```bash
# GDELT 配置
GDELT_VERSION=2
GDELT_TABLE=events
GDELT_COVERAGE=false
GDELT_TRANSLATION=true
```

### 5.3 依赖管理

在 `requirements.txt` 中添加：
```
gdelt>=0.1.10
```

## 6. 使用示例

### 6.1 基本使用

```python
from src.adapters.news.api_manager import get_news_manager

# 获取新闻管理器
manager = get_news_manager()

# 获取 GDELT 源
gdelt_source = manager.get_source("GDELT")

# 配置抓取参数
from src.ports.extraction import FetchConfig
from datetime import datetime, timedelta

config = FetchConfig(
    max_items=100,
    from_date=datetime.now() - timedelta(days=7),
    to_date=datetime.now(),
    keywords=["China", "trade"]
)

# 抓取数据
result = await gdelt_source.fetch(config)
if result.success:
    for item in result.items:
        print(f"{item.title} - {item.source_url}")
```

### 6.2 与其他源统一使用

```python
# 从所有源（包括 GDELT）抓取
results = await manager.fetch_all(config)

for source_name, result in results.items():
    print(f"{source_name}: {len(result.items)} 条新闻")
```

### 6.3 流式处理

```python
async for news_item in gdelt_source.fetch_stream(config):
    # 处理单条新闻
    process_news(news_item)
```

## 7. 性能考虑

### 7.1 内存管理

- **问题**: GDELT 数据量巨大，单日数据可能达到 500MB+
- **解决方案**:
  1. 限制 `max_items` 参数
  2. 使用流式处理（`fetch_stream`）
  3. 分批处理，写入磁盘后清空内存
  4. 提供配置选项，允许用户选择数据量

### 7.2 并发处理

- gdeltPyR 支持并行 HTTP 请求，利用多核加速
- 适配器本身是异步的，可以与其他源并发抓取
- 注意：GDELT 服务器可能有速率限制

### 7.3 缓存策略（可选）

- 对于历史数据，可以考虑缓存
- 使用项目现有的缓存机制（`src/infra/cache.py`）
- 缓存键：`gdelt:{version}:{table}:{date}`

## 8. 数据质量与验证

### 8.1 数据验证

GDELT 文档提醒用户注意数据质量问题：
- **重复报告**: 同一事件可能被多次报告
- **循环报告**: 新闻源之间相互引用
- **错误报告**: 来自不可靠源的数据
- **单一来源事件**: 仅来自未知源的重要事件

### 8.2 适配器层面的处理

1. **去重**: 使用 URL 或事件 ID 去重
2. **来源验证**: 提供 `_rooturl` 方法获取根 URL，便于验证来源
3. **元数据保留**: 保留 GDELT 原始字段，供后续分析使用
4. **警告机制**: 对于可疑数据，在 metadata 中标记

### 8.3 与项目去重机制集成

项目已有去重机制（`NewsDeduplicator`），GDELT 适配器应：
- 保留原始 URL 用于去重
- 提供足够的元数据支持语义去重
- 与现有去重流程无缝集成

## 9. 扩展性设计

### 9.1 多表支持

适配器可以扩展支持多个表：
- `GDELTEventsAdapter`: events 表
- `GDELTGKGAdapter`: gkg 表
- `GDELTMentionsAdapter`: mentions 表（2.0）

或者通过配置参数切换表类型。

### 9.2 Google BigQuery 集成（未来）

GDELT 文档提到未来支持 Google BigQuery 直接查询：
- 减少内存占用
- 服务器端处理
- 需要 SQL 知识

适配器可以预留接口，未来扩展：
```python
class GDELTBigQueryAdapter(GDELTAdapter):
    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult:
        # 使用 BigQuery 查询
        ...
```

### 9.3 其他 GDELT 数据源

- VGKG (Visual Knowledge Graph)
- TV-GKG (American Television Global Knowledge Graph)

可以创建专门的适配器或通过配置支持。

## 10. 测试策略

### 10.1 单元测试

```python
# tests/adapters/news/test_gdelt_adapter.py

async def test_gdelt_adapter_basic():
    adapter = GDELTAdapter(version=2, table="events")
    assert adapter.is_available()
    assert adapter.source_name == "GDELT"
    
async def test_gdelt_adapter_fetch():
    adapter = GDELTAdapter(version=2)
    config = FetchConfig(max_items=10)
    result = await adapter.fetch(config)
    assert result.success
    assert len(result.items) > 0
```

### 10.2 集成测试

- 测试与 NewsAPIManager 的集成
- 测试与其他源的并发抓取
- 测试数据转换的正确性

### 10.3 性能测试

- 测试大数据集的处理能力
- 测试内存使用情况
- 测试超时机制

## 11. 实现检查清单

### 11.1 核心功能

- [ ] 实现 `GDELTAdapter` 类
- [ ] 实现 `NewsSource` 接口的所有方法
- [ ] 实现 GDELT 数据到 `NewsItem` 的转换
- [ ] 支持 GDELT 1.0 和 2.0
- [ ] 支持日期范围查询
- [ ] 实现错误处理和降级策略

### 11.2 集成

- [ ] 注册到 `NewsAPIManager`
- [ ] 更新 `__init__.py` 导出
- [ ] 添加依赖到 `requirements.txt`
- [ ] 更新文档

### 11.3 优化

- [ ] 实现异步处理
- [ ] 添加超时机制
- [ ] 实现流式处理
- [ ] 添加缓存支持（可选）

### 11.4 测试

- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试
- [ ] 错误场景测试

## 12. 注意事项与限制

### 12.1 GDELT 数据特点

1. **延迟**: GDELT 1.0 数据在次日 6AM EST 发布
2. **数据缺失**: 某些时间间隔可能缺失数据
3. **数据量**: 单日数据可能非常大（500MB+）
4. **数据质量**: 需要谨慎使用，注意验证来源

### 12.2 适配器限制

1. **标题和内容**: GDELT 不直接提供，需要额外处理
2. **同步库**: gdeltPyR 是同步的，需要异步包装
3. **内存**: 大数据集需要足够内存
4. **网络**: 依赖 GDELT 服务器可用性

### 12.3 最佳实践

1. **小批量查询**: 避免一次性查询大量数据
2. **错误处理**: 始终检查 `FetchResult.success`
3. **数据验证**: 使用 `_rooturl` 方法验证来源
4. **去重**: 利用项目的去重机制
5. **监控**: 记录抓取统计和错误

## 13. 总结

GDELT 适配器的设计遵循项目的 DDD 架构原则，通过实现 `NewsSource` 接口，将 GDELT 数据源无缝集成到现有的新闻采集系统中。适配器提供了灵活的配置选项，支持 GDELT 1.0 和 2.0，并考虑了性能、错误处理和扩展性。

关键设计决策：
- 使用 events 表作为默认数据源
- 支持异步处理和流式处理
- 保留原始元数据供后续分析
- 提供降级策略和错误处理
- 与现有去重机制集成

通过这个适配器，项目可以充分利用 GDELT 的全球新闻数据，同时保持与现有架构的一致性。

---

**文档版本**: 1.0  
**创建日期**: 2025.12.27 
**最后更新**: 2025.12.27 

