# README.md

## [市场透镜 - Market Lens](https://github.com/1zhujianbang/MarketLens)

基于全球市场数据和新闻资讯构建的智能感知与决策系统，通过知识图谱和多智能体协作实现从"信息→认知→行动"的闭环。

## 项目概述

本项目旨在构建一个全球市场的智能感知与决策系统，通过融合实时和历史新闻（涵盖各国各地区各领域）、多源结构化数据与实时和历史行情，自动抽取事件、实体及其关联，动态构建可解释的知识图谱，并驱动多智能体协作推理，最终输出具备归因路径的市场预期信号，结合强化学习模型实现从“信息→认知→行动”的闭环，为投资者提供兼具时效性、逻辑性与可验证性的辅助决策支持。

### 当前计划
- 见 [2025年12月10日.report.md](2025年12月10日.report.md)

### 核心功能

|功能|状态| 
|---|:---:| 
|全球新闻资讯（各国各地区各领域）接入|&#x2705;| 
|多源结构化数据与实时/历史行情整合|&#x2705;| 
|事件与实体自动抽取及关联分析|&#x2705;| 
|动态可解释知识图谱构建|&#x2705;| 
|多智能体协作推理系统|&#x274C;| 
|具备归因路径的市场预期信号生成|&#x274C;| 
|强化学习模型驱动的决策闭环|&#x274C;| 
|可验证的辅助决策支持输出|&#x274C;| 
|多渠道结果推送（邮箱、微信）|&#x274C;| 

## 知识图谱构建

### 智能体流水线设计

|阶段|功能|状态|
|---|---|:---:|
|实体/事件提取|从新闻中提取实体与事件摘要|&#x2705;|
|新闻拓展|按实体/关键词扩展相关新闻|&#x2705;|
|图谱构建/维护|构建与压缩实体-事件图谱|&#x2705;|
|市场分析/推理|基于图谱做预期分析|&#x274C;（未实现）|
|可视化|图谱与结果可视化|&#x2705;|

##### 知识图谱技术细节

1. 多源数据接入
- 从新闻、行情、公开企业信息等渠道收集原始数据
- 用简单脚本或轻量流处理（如 Python + Redis / Kafka）做初步清洗和去重

2. 图谱构建方式
- 提取新闻中的实体、事件抽象、实体与事件抽象的映射
- 通过规则或 NLP 模型（如 spaCy、LTP、BERT）扩展实体和事件抽象
- 图谱存储暂用 JSON 或 SQLite，后期可迁移到 Neo4j 等图数据库。

3. 智能体协作流程 
- 从新闻中抽取出实体 → 用这些实体去检索更多相关新闻 → 聚合事件 → 构建“实体-事件”关联 → 推导市场影响

4. 输出与验证
- 最终目标是生成可解释的推导链,这个推导链可以覆盖到所有市场和所有领域
- 所有预测会记录并回溯验证，避免模型幻觉

> 💡 注：所有组件设计为松耦合，技术选型可根据实际资源调整（例如不用 Flink、不用 OneKE、不用 Prophet），优先保证可运行和可维护

##### 图谱可视化例图
所有节点的总图，可缩放，此处展示大概
![kg1](static/kg_vis/kg1.png)
筛选词并设置深度为2
![kg2](static/kg_vis/kg2.png)
## 项目结构

    System/
    ├── main.py                   # 主程序入口
    ├── app.py                    # Web 应用入口
    ├── .gitignore
    ├── LICENSE
    ├── README.md                 # 项目说明文档
    ├── config/                   # 配置文件目录
    │   ├── .env.example          # 环境变量 (API Keys)
    │   ├── config.yaml           # 主配置文件
    │   └── pipelines/            # 预定义流水线配置
    ├── data/                     # 数据存储目录
    │   ├── tmp/                  # 临时文件（抓取/去重/提取/缓存）
    │   │   ├── raw_news/         # 原始抓取
    │   │   └── deduped_news/     # 去重结果
    │   │   └── extracted_events_*.jsonl # 提取结果缓存
    │   ├── logs/                 # 系统日志
    │   ├── entities.json         # 实体库
    │   ├── abstract_to_event_map.json  # 事件摘要映射
    │   ├── knowledge_graph.json  # 知识图谱（可为空占位）
    │   └── stop_words.txt        # 停用词表
    ├── pages/                    # Streamlit 页面
    │   ├── 1_Dashboard.py        # 仪表盘
    │   ├── 2_Pipeline_Builder.py # 任务构建器
    │   ├── 3_Data_Inspector.py   # 数据查看器
    │   └── 4_Knowledge_Graph.py  # 图谱可视化
    └── src/                      # 源代码目录
        ├── agents/               # 智能体模块
        │   ├── agent1.py         # 提取与去重智能体
        │   ├── agent2.py         # 拓展搜索智能体
        │   ├── agent3.py         # 图谱维护与合并智能体
        │   └── api_client.py     # LLM API 客户端
        ├── core/                 # 核心框架
        │   ├── engine.py         # 任务执行引擎
        │   ├── context.py        # 执行上下文管理
        │   └── registry.py       # 工具注册中心
        ├── data/                 # 数据层
        │   ├── api_client.py     # 数据源 API 客户端
        │   └── news_collector.py # 新闻采集器
        ├── functions/            # 原子功能库 (Pipeline Tools)
        │   ├── data_fetch.py     # 数据获取工具
        │   ├── extraction.py     # 提取工具
        │   ├── graph_ops.py      # 图谱操作工具
        │   └── reporting.py      # 报告生成工具
        ├── web/                  # Web 辅助模块
        │   ├── config.py         # Web 配置
        │   └── utils.py          # Web 通用工具
        └── utils/                # 通用工具
            ├── tool_function.py  # 基础工具集
            └── entity_updater.py # 实体更新逻辑


## 市场分析智能体

该项目采用可解释的市场分析模型，避免传统黑盒模型的局限性：

- 基于知识图谱的推理过程可追溯
- 分析结果提供明确的因果关系
- 支持多源数据的融合分析
- 结果可验证和回溯

## 运行说明

### 环境配置

1. 安装依赖：
requirements.txt为最小依赖清单（主要依赖与连带库）
   ```bash
   pip install -r requirements.txt
   ```

2. 配置文件：
   - 修改 `config/config.yaml` 配置数据源和分析参数
   - 设置环境变量（如API密钥），推荐可在前端渲染页编辑

3. 运行主程序：
   ```bash
   streamlit run app.py   # 前端入口，构建/运行 Pipeline
   # 可选：python main.py（需自行补充数据源与流程）
   ```

4. 智能体模块化程序：
   ```bash
   python src.agents.agent1
   python src.agents.agent2
   python src.agents.agent3
   ```

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

[Apache License 2.0](LICENSE)
