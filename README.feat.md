# MarketLens feat branch update

`feat`分支的当前版本主要更新了以下内容：
v1.4

- Neo4j 迁移与验证工具：新增 `migrate_sqlite_to_neo4j` 工具与连接/迁移脚本
- LLM 池支持 Ollama（OpenAI 兼容接口）
- Neo4j 图数据库支持：通过 `KG_STORE_BACKEND=neo4j|dual` 可切换/双写（配套 `NEO4J_URI/USER/PASSWORD`）

v1.3

- 实现Neo4j适配器与快照功能
- 新增Neo4j图数据库适配器，支持Cypher查询和批量操作
- 添加快照协议验证与快照视图页面
- 实现事件ID重定向解析功能
- 更新阶段一任务状态与文档
- 优化SQLite存储接口

v1.2

- 添加GDELT适配器及集成测试
- 实现GDELT新闻源适配器，支持从GDELT API获取新闻数据
- 添加相关测试用例验证适配器功能
- 更新fetch_utils支持extra参数传递
- 新增集成测试验证完整业务流程
- 补充设计文档说明GDELT适配器实现细节

v1.1

- 增加基于GDELT数据的新闻获取和处理功能
- 添加对GDELT数据的异步抓取支持，集成gdelt库
- 实现GDELT数据转换为DataFrame的方法，支持数据预处理和去重
- 新增实体名称的标准化处理，包括名称变体映射和多余词缀过滤
- 实现角色信息的提取与事件代码对应的关系分析，丰富事件语义
- 支持对时间戳字段的解析和格式化，生成多维时间特征（如季节、周末等）
- 兼容Dask以支持大规模GDELT数据分块处理与并行计算
- 实现GKG数据与事件数据的集成，增强主题和情感分析能力
- 移除项目中的performance_test.py性能测试脚本
- 更新requirements.txt添加对gdelt和dask[dataframe]依赖声明

这些更新都是在 `feat` 分支上完成的，等待测试完全通过后将合并到主分支。

更多新增功能需要先向该分支合并，贡献者需要提交PR用于新增功能
