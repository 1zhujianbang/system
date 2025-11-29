### 需求清单

|需求|状态|
|---|:---:|
|加密货币市场新闻资讯API接入|&#x274C;|
|历史几年K线数据（日线、周线、月线）获取|&#x274C;|
|利用历史数据训练模型，PPO或等价/混合算法|&#x274C;|
|模型分析实时数据，输出信号|&#x274C;|
|实时数据+信号 / 新闻 异步输入到智能体及输出分析结果|&#x274C;|
|根据分析结果推送消息到邮箱、微信公众号|&#x274C;|
|摸鱼|&#x2705;|

### 项目结构

    System/
    ├── main.py                    # 主程序入口
    ├── README.md                  # 项目说明文档
    ├── config/                    # 配置文件目录
    │   ├── .env                  # 环境变量配置
    │   └── config.yaml           # 主配置文件
    ├── models/                   # 模型文件目录
    │   ├── example.py           # 模型示例代码
    │   └── transformer_v2_7d.pt # 预训练模型文件
    └── src/                      # 源代码目录
        ├── __init__.py
        ├── agents/               # 智能体模块
        │   └── trading_agent.py  # 交易智能体
        ├── analysis/             # 分析模块
        │   └── technical_calculator.py # 技术指标计算器
        ├── config/               # 配置管理
        │   └── config_manager.py # 配置管理器
        ├── data/                 # 数据模块
        │   ├── data_collector.py # 数据收集器
        │   └── websocket_collector.py # WebSocket实时数据
        ├── models/               # 模型管理
        │   └── model_loader.py   # 模型加载器
        └── utils/                # 工具模块
            └── validators.py     # 验证工具（当前为空）
