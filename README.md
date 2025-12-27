# MarketLens

## 一起把新闻知识图谱做出来

MarketLens 是一个面向建设者的开源工程，目标是以“配置驱动的流水线（Pipeline）”方式，将新闻抓取、实体/事件抽取、持久化存储、图谱更新与报告生成串联为可复现流程；同时提供 Streamlit 可视化界面用于运行、回放与结果检查。

你可以把它当作一个可持续迭代的工程底座：
- 配置驱动：Pipeline 用 YAML 定义，工具函数以 `@register_tool` 注册并复用
- 数据可复现：SQLite 作为主存储，图谱快照输出到 `data/snapshots/`
- 交互可观察：Streamlit 页面用于运行、日志观察与图谱查看

如果你也愿意一起建设：欢迎从修复一个小问题、补一条测试、完善一个工具、或优化一段抽取逻辑开始。建议 PR 尽量小而聚焦，我们会尽力快速 review。

## 🚀 快速开始

### 本地运行

```bash
git clone https://github.com/1zhujianbang/MarketLens.git
cd MarketLens
pip install -r requirements.txt
cp config/.env.example config/.env.local
# 编辑 config/.env.local，填入 API 配置（见下方“配置说明”）
streamlit run app.py
```

浏览器访问：http://localhost:8501

### Docker 运行

```bash
cp config/.env.example config/.env.local
# 编辑 config/.env.local，填入 API 配置（见下方“配置说明”）
docker-compose up -d
```

浏览器访问：http://localhost:8501


## 📁 项目结构

```
MarketLens/
├── src/
│   ├── app/pipeline/            # Pipeline 引擎
│   ├── app/business/            # 业务工具实现（抓取/抽取/图谱/报告）
│   ├── adapters/                # 外部系统适配（news/llm/sqlite）
│   ├── infra/                   # 配置、注册表、基础设施工具
│   └── web/                     # Streamlit UI（页面/服务）
├── config/
│   ├── pipelines/               # Pipeline 配置（YAML）
│   ├── .env.example             # 环境变量示例
│   └── .env.local               # 本地环境变量（自行创建）
├── data/
│   ├── store.sqlite             # SQLite 主数据库
│   ├── snapshots/               # 图谱快照输出目录
│   └── projects/<project_id>/   # 运行记录（runs/）
├── pages/                       # Streamlit 路由
├── tests/                       # 测试
└── app.py                       # 应用入口
```

## 🐳 Docker 部署

```bash
cp config/.env.example config/.env.local
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## ⚙️ 配置说明

### 配置文件结构

```
config/
├── base.yaml                   # 基础配置（用户/模型/数据）
├── pipelines/                  # Pipeline配置
│   └── default_analysis.yaml  # 默认分析流程
├── agents/                     # Agent配置（兼容旧版）
│   ├── agent1.yaml
│   ├── agent2.yaml
│   └── agent3.yaml
├── entity_merge_rules.json     # 实体合并规则
├── .env.local                  # 环境变量（需手动创建）
└── .key_store.enc              # 加密密钥存储
```

### 环境变量配置（`.env.local`）

```bash
# 新闻 API（支持多个 key 轮询）
GNEWS_APIS_POOL='["gnews_key_1","gnews_key_2"]'

# LLM（OpenAI 兼容接口）配置：JSON 数组
AGENT1_LLM_APIS='[{"name":"deepseek-chat","base_url":"https://api.deepseek.com/","api_key":"sk-xxx","model":"deepseek-chat","enabled":true}]'

# 语义匹配（可选）
HF_ENDPOINT=https://hf-mirror.com  # 国内镜像源
```

### Pipeline 配置示例（`config/pipelines/default_analysis.yaml`）

```yaml
name: "Daily Market Scan (Quick Test)"
description: "Fetch a single news item and extract events to verify the pipeline."
steps:
  - id: "fetch_news"
    tool: "fetch_news_stream"
    inputs:
      limit: 1 # 仅获取1条
      sources: ["GNews-cn"] 
    output: "raw_news_data"

  - id: "process_news"
    tool: "batch_process_news"
    inputs:
      news_list: "$raw_news_data"
      limit: 1 # 再次限制
    output: "extracted_events"

  - id: "update_kg"
    tool: "update_graph_data"
    inputs:
      events_list: "$extracted_events"
    output: "update_status"

  - id: "generate_report"
    tool: "generate_markdown_report"
    inputs:
      events_list: "$extracted_events"
      title: "最新市场动态简报 (测试版)"
    output: "final_report_md"
```

## 🤝 参与建设

我们希望把 MarketLens 逐步沉淀为一个“可持续演进的新闻图谱工程底座”。如果你愿意一起建设，以下贡献都非常欢迎：
- 修复 bug、补齐测试，让整体行为更稳定、可回归
- 新增或改进工具函数（`@register_tool`），提升 Pipeline 的可复用性
- 改进抽取/去重/合并策略，让图谱更准确、更可解释
- 优化默认 Pipeline 配置，让默认路径更顺滑、可跑通

### 建设入口（从这里开始改）

- Pipeline 引擎与执行：`src/app/pipeline/`
- 工具注册入口：`src/infra/registry.py`
- 业务工具实现：`src/app/business/`
- 新闻源适配：`src/adapters/news/`
- SQLite 存储：`src/adapters/sqlite/`
- Streamlit 页面：`pages/` 与 `src/web/`
- 默认 Pipeline：`config/pipelines/default_analysis.yaml`

### 协作方式（开源友好）

- 任务与路线：`tasks/`（欢迎认领、补充拆解）
- 如果不确定从何入手：欢迎直接提 Issue，我们会一起把问题拆解为可实现的子任务
- 工作流约定：`docs/工作流说明文档.md`
- 函数对齐：`docs/函数说明文档.md`（新增/调整函数后同步更新）
- 提交 PR 前跑测试：

```bash
python -m pytest
```

## 📄 许可证

[Apache License 2.0](LICENSE)

---

## 📞 联系方式

- **GitHub Issues**: https://github.com/1zhujianbang/MarketLens/issues
- **项目主页**: https://github.com/1zhujianbang/MarketLens
