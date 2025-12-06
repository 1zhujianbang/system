from ..config.config_manager import MarketAnalysisConfig
from ..models.model_loader import ModelLoader
from ..data.data_collector import MarketClient
from ..data.news_collector import NewsCollector
from ..agents.agent1 import process_news_stream
from ..utils.tool_function import tools
tools = tools()
from datetime import datetime, timezone
import re
import pandas as pd
import json
import os
from pathlib import Path
import uuid

class MarketAnalysisAgent:
    def __init__(self, config: MarketAnalysisConfig):
        self.config = config
        self.model = None
        self.is_ready = False
        self._cleanup_done = False

        # 初始化客户端
        self.market_client = MarketClient(config.user_config, config.data_config)
        # 使用统一的 NewsCollector（内部通过 DATA_APIS 调用 Blockbeats、GNews 等多数据源）
        self.news_collector = NewsCollector()

        # 数据存储
        self.market_data = {}
        self.realtime_data = {}
        self.technical_data = {}
        self.news_data = {}
        self.market_sentiment = {}
        self.entities_data = {}
        self.knowledge_graph = {}

    async def initialize(self):
        """初始化市场分析Agent的核心流程"""
        print("Initializing AI Market Analysis Agent...")

        try: 
             # 1. 验证配置
            print("🔍 验证模型配置...")
            if not hasattr(self.config, 'models_config'):
                raise ValueError("配置中缺少 models_config 字段")

            if self.config.models_config is None:
                raise ValueError("models_config 为 None")

            print(f"✅ 模型配置存在: {self.config.models_config.model_name}")

            # 2. 加载模型
            print("🔍 初始化模型加载器...")
            model_loader = ModelLoader()
            print(f"🔍 模型目录: {model_loader.models_dir}")
            print(f"🔍 模型名称: {self.config.models_config.model_name}")
            
            print("🔍 开始加载模型...")
            self.model = model_loader.load_model(self.config.models_config)
            print(f"✅ Model {self.config.models_config.model_name} loaded successfully.")

            # 3. 市场数据初始化
            # self._initialize_market_data()

            # 4. 新闻数据初始化
            await self._initialize_news_data()

            # 5. 实体数据初始化
            await self._initialize_entities_data()

            # 6. 标记为就绪状态
            self.is_ready = True
            print("AI Market Analysis Agent is now READY.")

        except Exception as e:
            print(f"❌ Agent初始化失败: {type(e).__name__}: {str(e)}")
            import traceback
            print("🔍 详细堆栈跟踪:")
            traceback.print_exc()
            raise

    def get_status(self):
        structured_news = self.news_data.get('structured', pd.DataFrame())
        return {
            "is_ready": self.is_ready,
            "risk_preference": self.config.user_config.risk_preference,
            "model_used": self.config.models_config.model_name,
            "market_sentiment": self.market_sentiment.get('sentiment', 'neutral'),
            "news_count": len(structured_news),
            "entities_extracted": sum(len(ents) for ents in structured_news.get('entities', [])),
            "breaking_news": self.market_sentiment.get('breaking_news_count', 0),
        }
    
    async def cleanup(self):
        """清理资源"""
        if self._cleanup_done:
            return
        print("🧹 清理交易Agent资源...")
        try:
            if hasattr(self.news_collector, 'close'):
                await self.news_collector.close()
            elif hasattr(self.news_collector, 'session') and self.news_collector.session:
                await self.news_collector.session.close()
        except Exception as e:
            print(f"⚠️ 资源清理过程中出现错误: {e}")
        finally:
            self._cleanup_done = True

    def _initialize_market_data(self):
        """初始化市场数据"""
        print("初始化市场数据...")
        
        # 获取市场符号列表
        symbols = self.market_client.get_symbols()
        print(f"配置的市场符号: {symbols}")
        
        if not symbols:
            raise ValueError("未配置市场符号")
        
        # 获取实时数据
        print("获取实时行情数据...")
        self.realtime_data = self.market_client.get_all_tickers() 
        print(f"成功获取 {len(self.realtime_data)} 个市场符号的实时数据")
        
        # 验证实时数据
        for symbol in symbols:
            if symbol not in self.realtime_data:
                print(f"⚠️  警告: 无法获取 {symbol} 的实时数据")
        
        # 获取历史K线数据
        print("获取历史K线数据...")
        self.market_data = self.market_client.get_all_historical_klines()
        print(f"成功获取 {len(self.market_data)} 个市场符号的历史数据")
        
        # 验证历史数据完整性
        self._validate_market_data()
        
        # 初始化技术指标数据
        print("计算技术指标...")
        self._initialize_technical_data()
        
        # 打印数据统计
        self._print_data_statistics()

    def _validate_market_data(self):
        """验证市场数据完整性"""
        for pair, data in self.market_data.items():
            if data.empty:
                print(f"⚠️  警告: {pair} 历史数据为空")
                continue
                
            # 检查数据量是否足够
            min_data_points = self.config.models_config.data_window
            if len(data) < min_data_points:
                print(f"⚠️  警告: {pair} 数据点不足 ({len(data)} < {min_data_points})")
            
            # 检查数据时间范围
            time_range = data.index[-1] - data.index[0]
            print(f"   {pair}: {len(data)} 根K线, 时间范围: {time_range.days}天")

    def _initialize_technical_data(self):
        """初始化技术指标数据"""
        from ..analysis.technical_calculator import TechnicalCalculator
        
        # 初始化技术指标计算器
        tech_calculator = TechnicalCalculator()
        
        for pair, data in self.market_data.items():
            if not data.empty:
                try:
                    # 计算技术指标
                    self.technical_data[pair] = tech_calculator.calculate_all_indicators(data)
                    
                    # 验证技术指标计算
                    required_features = self.config.models_config.features
                    missing_features = tech_calculator.validate_features(
                        self.technical_data[pair], required_features
                    )
                    
                    if missing_features:
                        print(f"⚠️  警告: {pair} 缺少特征 {missing_features}")
                    else:
                        print(f"✅ {pair} 技术指标计算完成，包含 {len(self.technical_data[pair].columns)} 个特征")
                        
                except Exception as e:
                    print(f"❌ {pair} 技术指标计算失败: {e}")
                    # 如果计算失败，至少保留原始数据
                    self.technical_data[pair] = data



    def _print_data_statistics(self):
        """打印数据统计信息"""
        print("\n📊 数据初始化完成:")
        print(f"   市场符号数量: {len(self.market_data)}")
        print(f"   时间框架: {self.market_client.get_timeframe()}")
        print(f"   历史天数: {self.market_client.get_historical_days()}")
        
        total_bars = sum(len(data) for data in self.market_data.values())
        print(f"   总K线数量: {total_bars}")
        
        # 显示每个市场符号的最新价格
        print("\n   最新价格:")
        tickers = self.market_client.get_all_tickers()
        for symbol, ticker in tickers.items():
            if ticker and 'price' in ticker:
                print(f"     {symbol}: {ticker['price']}")

    async def _initialize_news_data(self):
        """初始化新闻数据：通过统一 NewsCollector + Agent1 处理多数据源新闻"""
        print("📰 初始化新闻数据（NewsCollector + Agent1）...")
        try:
            # 1. 使用统一 NewsCollector 抓取多数据源新闻（Blockbeats、GNews 等），写入 raw_news 目录
            print("🔍 调用 NewsCollector.data_extract 抓取新闻（支持多数据源）...")
            await self.news_collector.data_extract()

            # 2. 调用 agent1 主流程，从 raw_news / deduped_news 中读取并结构化处理
            process_news_stream()

            # 3. 从 agent1 输出文件构建结构化 DataFrame
            df_structured = self._build_structured_news_from_agent1_output()

            # 4. 保存并分析
            self.news_data['structured'] = df_structured
            self.market_sentiment = self._analyze_market_sentiment_from_df(df_structured)

            # 5. 打印摘要
            self._print_news_summary()

        except Exception as e:
            print(f"❌ 新闻数据初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.news_data = {'structured': pd.DataFrame(), 'error': str(e)}
            self.market_sentiment = self._analyze_market_sentiment_from_df(pd.DataFrame())
    
    async def _initialize_entities_data(self):
        """初始化实体数据，用于构建知识图谱"""
        print("🔗 初始化实体数据...")
        try:
            # 从 agent1 输出的 entities.json 文件加载实体数据
            entities_file = Path(tools.DATA_DIR) / "entities.json"
            if entities_file.exists():
                with open(entities_file, "r", encoding="utf-8") as f:
                    entities_data = json.load(f)
                    self.entities_data = entities_data
                    print(f"✅ 成功加载 {len(entities_data)} 个实体数据")
            else:
                print(f"⚠️  未找到实体数据文件: {entities_file}")
        except Exception as e:
            print(f"❌ 实体数据初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.entities_data = []
    
    def _build_structured_news_from_agent1_output(self) -> pd.DataFrame:
        """从 agent1 生成的 abstract_map.json 构建结构化 DataFrame"""
        if not tools.ABSTRACT_MAP_FILE.exists():
            return pd.DataFrame()

        with open(tools.ABSTRACT_MAP_FILE, "r", encoding="utf-8") as f:
            abstract_map = json.load(f)

        records = []
        for abstract, data in abstract_map.items():
            records.append({
                "abstract": abstract,
                "entities": data["entities"],
                "event_summary": data["event_summary"],
                "sources": data["sources"],
                "first_seen": data["first_seen"]
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["title"] = df["abstract"]
        df["content"] = df["event_summary"]
        df["id"] = df.index.astype(str)
        return df

    def _analyze_market_sentiment_from_df(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'breaking_news_count': 0,
                'top_entities': [],
                'total_news': 0,
                'last_updated': datetime.now(timezone.utc)
            }

        # 实体统计（用于情绪代理）
        all_entities = []
        for ents in df['entities'].dropna():
            all_entities.extend(ents)
        from collections import Counter
        entity_freq = Counter(all_entities)
        top_entities = [ent for ent, _ in entity_freq.most_common(10)]

        # 简化情绪：仅基于新闻数量（或可后续接入LLM情感打分）
        total_news = len(df)
        sentiment_score = total_news  # 或设为 0 表示中性
        sentiment = 'active' if total_news > 5 else 'quiet'

        return {
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'breaking_news_count': total_news,
            'top_entities': top_entities,
            'total_news': total_news,
            'last_updated': datetime.now(timezone.utc)
        }

    def _print_news_summary(self):
        sentiment = self.market_sentiment
        df = self.news_data.get('structured', pd.DataFrame())
        
        print("\n📰 新闻数据摘要:")
        print(f"   总新闻数: {sentiment.get('total_news', 0)}")
        print(f"   市场活跃度: {sentiment.get('sentiment', 'unknown')}")
        print(f"   重大新闻: {sentiment.get('breaking_news_count', 0)} 条")
        print(f"   高频实体: {', '.join(sentiment.get('top_entities', [])[:5])}")

        if not df.empty:
            print("\n   最新结构化新闻:")
            for _, row in df.head(5).iterrows():
                title = row.get('abstract', '')[:60]
                entities = ', '.join(row.get('entities', []))
                print(f"     {title} | 实体: {entities}")

    async def update_news_data(self):
        """更新新闻数据（复用初始化逻辑）"""
        if not self.is_ready:
            return
        print("🔄 更新新闻数据...")
        await self._initialize_news_data()
        print("✅ 新闻数据更新完成")
    
    # ======================
    # 🧠 智能体2 & 知识图谱 占位区
    # ======================

    async def _expand_news_with_kg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【占位】智能体2：基于知识图谱扩展相关新闻
        输入：含 entities 的 DataFrame
        输出：增强后的 DataFrame（含 expanded_entities, related_news_ids 等）
        """
        # TODO: 实现基于 Neo4j / 内存图的关联扩展
        print("🚧 智能体2（KG扩展）尚未实现")
        return df

    def _build_temporal_knowledge_graph(self):
        """
        【占位】构建时序知识图谱（用于路径推理）
        """
        print("🚧 知识图谱构建模块尚未实现")
        pass

    async def update_knowledge_graph(self):
        """
        【占位】主入口：更新知识图谱
        """
        if not self.is_ready or self.news_data.get('structured') is None:
            return
        await self._expand_news_with_kg(self.news_data['structured'])
        self._build_temporal_knowledge_graph()
