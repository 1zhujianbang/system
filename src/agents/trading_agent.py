from ..config.config_manager import TradingConfig
from ..models.model_loader import ModelLoader
from ..data.data_collector import OKXMarketClient
from ..data.news_collector import BlockbeatsNewsCollector, NewsType, Language
from ..agents.agent1 import Agent1EntityExtractor
from datetime import datetime, timezone
import re
import pandas as pd

class TradingAgent:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = None
        self.portfolio = {
            # 余额
            'cash': config.user_config.cash,
            # 持仓
            'positions': {},
        }
        self.is_ready = False
        self._cleanup_done = False

        # 初始化客户端
        self.okx_client = OKXMarketClient(config.user_config, config.data_config)
        self.news_collector = BlockbeatsNewsCollector(language=Language.CN)
        self.agent1 = None

        # 数据存储
        self.market_data = {}  # 历史K线数据
        self.realtime_data = {}  # 实时行情数据
        self.technical_data = {}  # 技术指标数据
        self.news_data = {}  # 新闻数据
        self.market_sentiment = {}  # 市场情绪分析

    async def initialize(self):
        """初始化Agent的核心流程"""
        print("Initializing AI Trading Agent...")

        try:
             # 1. 验证配置
            print("🔍 验证模型配置...")
            if not hasattr(self.config, 'modeL_config'):
                raise ValueError("配置中缺少 modeL_config 字段")
            
            if self.config.modeL_config is None:
                raise ValueError("modeL_config 为 None")
            
            print(f"✅ 模型配置存在: {self.config.modeL_config.model_name}")

            # 2. 加载模型
            print("🔍 初始化模型加载器...")
            model_loader = ModelLoader()
            print(f"🔍 模型目录: {model_loader.models_dir}")
            print(f"🔍 模型名称: {self.config.modeL_config.model_name}")
            
            print("🔍 开始加载模型...")
            self.model = model_loader.load_model(self.config.modeL_config)
            print(f"✅ Model {self.config.modeL_config.model_name} loaded successfully.")

            # 3. 交易数据初始化 
            self._initialize_trading_data()

            # 4. 加载智能体1（实体提取器）
            print("🔍 初始化智能体1（实体提取器）...")
            auto_update_entities = getattr(self.config.user_config, 'auto_update_entities', False)
            self.agent1 = Agent1EntityExtractor(auto_update=auto_update_entities)
            print(f"✅ 智能体1已就绪 (auto_update={auto_update_entities})")

            # 5. 新闻数据初始化
            await self._initialize_news_data()

            # 6. 初始化数据流 (伪代码)
            # self.data_stream = DataStream(self.config.user_config.trading_pairs)

            # 7. 标记为就绪状态
            self.is_ready = True
            print("AI Trading Agent is now READY.")

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
            "cash": self.config.user_config.cash,
            "risk_appetite": self.config.user_config.risk_appetite,
            "model_used": self.config.modeL_config.model_name,
            "market_sentiment": self.market_sentiment.get('sentiment', 'neutral'),
            "news_count": len(structured_news),
            "entities_extracted": sum(len(ents) for ents in structured_news.get('entities', [])),
            "breaking_news": self.market_sentiment.get('breaking_news_count', 0),
            "event_types": self.market_sentiment.get('event_distribution', {})
        }
    
    async def cleanup(self):
        """清理资源 - 显示关闭所有客户端会话"""
        if self._cleanup_done:
            return
            
        print("🧹 清理交易Agent资源...")
        
        try:
            # 1. 关闭新闻收集器的会话
            if hasattr(self.news_collector, 'close'):
                await self.news_collector.close()
                print("✅ 新闻收集器会话已关闭")
            elif hasattr(self.news_collector, 'session') and self.news_collector.session:
                await self.news_collector.session.close()
                print("✅ 新闻收集器会话已关闭")
            
            # 2. 拓展

        except Exception as e:
            print(f"⚠️ 资源清理过程中出现错误: {e}")
        finally:
            self._cleanup_done = True

    def _initialize_trading_data(self):
        """初始化交易数据"""
        print("初始化交易数据...")
        
        # 3.1 验证交易对配置
        trading_pairs = self.okx_client.get_trading_pairs()
        print(f"配置的交易对: {trading_pairs}")
        
        if not trading_pairs:
            raise ValueError("未配置交易对")
        
        # 3.2 获取实时数据
        print("获取实时行情数据...")
        self.realtime_data = self.okx_client.get_all_tickers_with_changes() 
        print(f"成功获取 {len(self.realtime_data)} 个交易对的实时数据")
        
        # 验证实时数据
        for pair in trading_pairs:
            if pair not in self.realtime_data:
                print(f"⚠️  警告: 无法获取 {pair} 的实时数据")
        
        # 3.3 获取历史K线数据
        print("获取历史K线数据...")
        self.market_data = self.okx_client.get_all_historical_klines()
        print(f"成功获取 {len(self.market_data)} 个交易对的历史数据")
        
        # 验证历史数据完整性
        self._validate_market_data()
        
        # 3.4 初始化技术指标数据
        print("计算技术指标...")
        self._initialize_technical_data()
        
        # 3.5 打印数据统计
        self._print_data_statistics()

    def _validate_market_data(self):
        """验证市场数据完整性"""
        for pair, data in self.market_data.items():
            if data.empty:
                print(f"⚠️  警告: {pair} 历史数据为空")
                continue
                
            # 检查数据量是否足够
            min_data_points = self.config.modeL_config.data_window
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
                    required_features = self.config.modeL_config.features
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
        print(f"   交易对数量: {len(self.market_data)}")
        print(f"   时间框架: {self.okx_client.get_timeframe()}")
        print(f"   历史天数: {self.okx_client.get_historical_days()}")
        
        total_bars = sum(len(data) for data in self.market_data.values())
        print(f"   总K线数量: {total_bars}")
        
        # 显示每个交易对的最新价格
        print("\n   最新价格:")
        tickers_with_changes = self.okx_client.get_all_tickers_with_changes()
        for pair, ticker in tickers_with_changes.items():
            if ticker:
                display_str = self.okx_client.format_price_display(ticker)
                print(f"     {display_str}")

    async def _initialize_news_data(self):
        """初始化新闻数据"""
        print("📰 初始化新闻数据...")
    
        try:
            # 使用核心更新逻辑
            await self._update_news_core()
            
            # 初始化特定的设置
            self.news_data['initialized'] = True
            self.news_data['first_init_time'] = datetime.now(timezone.utc)
            
            # 打印新闻摘要
            self._print_news_summary()
            
        except Exception as e:
            print(f"❌ 新闻数据初始化失败: {str(e)}")
            self.news_data = {
                'important': [], 
                'error': str(e),
                'initialized': False
            }
    
    def _analyze_market_sentiment_from_df(self, df: pd.DataFrame) -> dict:
        """
        基于智能体1输出的结构化新闻DataFrame分析市场情绪
        输入: 包含 'event_type', 'entities' 列的DataFrame
        """
        if df.empty:
            return {
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'breaking_news_count': 0,
                'top_entities': [],
                'event_distribution': {},
                'total_news': 0,
                'last_updated': datetime.now(timezone.utc)
            }

        # 1. 事件类型分布（用于情绪倾向）
        event_counts = df['event_type'].value_counts().to_dict()
        
        # 2. 情绪映射（可配置）
        BULLISH_EVENTS = {'listing', 'partnership', 'upgrade', 'adoption'}
        BEARISH_EVENTS = {'regulation', 'hack', 'market'}  # market 可能中性，此处暂归负面
        
        bullish_score = sum(count for et, count in event_counts.items() if et in BULLISH_EVENTS)
        bearish_score = sum(count for et, count in event_counts.items() if et in BEARISH_EVENTS)
        
        sentiment_score = bullish_score - bearish_score
        
        if sentiment_score > 1:
            sentiment = 'bullish'
        elif sentiment_score < -1:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        # 3. 提取高频实体（前10）
        all_entities = [ent for ents in df['entities'].dropna() for ent in ents]
        from collections import Counter
        entity_freq = Counter(all_entities)
        top_entities = [ent for ent, _ in entity_freq.most_common(10)]

        # 4. 重大新闻计数（定义：非 None event_type 即视为重要）
        breaking_news_count = df['event_type'].notna().sum()

        return {
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'breaking_news_count': int(breaking_news_count),
            'top_entities': top_entities,
            'event_distribution': event_counts,
            'total_news': len(df),
            'last_updated': datetime.now(timezone.utc)
        }

    def _print_news_summary(self):
        """打印新闻摘要（基于结构化数据）"""
        sentiment = self.market_sentiment
        df = self.news_data.get('structured', pd.DataFrame())
        
        print("\n📰 新闻数据摘要:")
        print(f"   总新闻数: {sentiment.get('total_news', 0)}")
        print(f"   市场情绪: {sentiment.get('sentiment', 'unknown')} (分数: {sentiment.get('sentiment_score', 0)})")
        print(f"   重大新闻: {sentiment.get('breaking_news_count', 0)} 条")
        print(f"   高频实体: {', '.join(sentiment.get('top_entities', [])[:5])}")
        print(f"   事件分布: {sentiment.get('event_distribution', {})}")

        # 显示最新3条带实体的新闻
        if not df.empty:
            print("\n   最新结构化新闻:")
            for _, row in df.head(3).iterrows():
                title = row.get('title', '无标题')
                if len(title) > 60:
                    title = title[:57] + '...'
                entities = ', '.join(row['entities']) if row['entities'] else '无'
                event = row['event_type'] or 'unknown'
                print(f"     [{event}] {title} | 实体: {entities}")

    async def _update_news_core(self):
        """新闻数据核心更新逻辑"""
        # 1. 获取原始新闻列表
        important_news = await self.news_collector.get_latest_important_news(limit=20)

        # 2. 转为DataFrame
        df_raw = self.news_collector.news_to_dataframe(important_news)

        if df_raw.empty:
            self.news_data['structured'] = pd.DataFrame()
            self.market_sentiment = self._analyze_market_sentiment([])
            return

        # 3. 调用智能体1进行实体与事件类型提取
        df_enriched = self.agent1.process(df_raw)

        # 4. 保存结构化新闻数据
        self.news_data['structured'] = df_enriched

        # 5. 更新市场情绪
        self.market_sentiment = self._analyze_market_sentiment_from_df(df_enriched)

        # 6. （可选）为每个交易对保存关联新闻（后续可基于 entities 过滤）
        trading_pairs = self.okx_client.get_trading_pairs()
        for pair in trading_pairs:
            symbol = pair.split('-')[0].upper()
            # 占位：后续可由智能体2基于图谱扩展
            self.news_data[pair] = {
                'symbol': symbol,
                'related_entities': [symbol],  # 初始假设符号即实体
                'news_df': df_enriched[df_enriched['entities'].apply(lambda ents: symbol in ents)]
            }

    async def update_news_data(self):
        """更新新闻数据"""
        if not self.is_ready:
            return
        
        try:
            print("🔄 更新新闻数据...")
            
            # 使用共用的核心更新逻辑
            await self._update_news_core()
            
            # 更新特定的处理
            self.news_data['last_updated'] = datetime.now(timezone.utc)
            self.news_data['update_count'] = self.news_data.get('update_count', 0) + 1
        
            self._print_news_summary()
            
            print(f"✅ 新闻数据更新完成")
            
        except Exception as e:
            print(f"❌ 新闻数据更新失败: {str(e)}")
            self.news_data['last_update_error'] = str(e)

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