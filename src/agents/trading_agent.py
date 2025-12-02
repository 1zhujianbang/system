from ..config.config_manager import TradingConfig
from ..models.model_loader import ModelLoader
from ..data.data_collector import OKXMarketClient
from ..data.news_collector import BlockbeatsNewsCollector, NewsType, Language
from ..agents.agent1 import process_news_stream, ENTITIES_FILE, ABSTRACT_MAP_FILE, RAW_NEWS_DIR
from datetime import datetime, timezone
import re
import pandas as pd
import json
import os
from pathlib import Path
import uuid

class TradingAgent:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = None
        self.portfolio = {
            'cash': config.user_config.cash,
            'positions': {},
        }
        self.is_ready = False
        self._cleanup_done = False

        # 初始化客户端
        self.okx_client = OKXMarketClient(config.user_config, config.data_config)
        self.news_collector = BlockbeatsNewsCollector(language=Language.CN)

        # 数据存储
        self.market_data = {}
        self.realtime_data = {}
        self.technical_data = {}
        self.news_data = {}
        self.market_sentiment = {}

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
            # self._initialize_trading_data()

            # 4. 新闻数据初始化
            await self._initialize_news_data()

            # 5. 初始化数据流 (伪代码)
            # self.data_stream = DataStream(self.config.user_config.trading_pairs)

            # 6. 标记为就绪状态
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
        """初始化新闻数据：通过 agent1 处理"""
        print("📰 初始化新闻数据（调用 Agent1）...")
        try:
            # 1. 获取原始新闻
            important_news = await self.news_collector.get_latest_important_news(limit=5)
            if not important_news:
                print("📭 未获取到重要新闻")
                self.news_data['structured'] = pd.DataFrame()
                self.market_sentiment = self._analyze_market_sentiment_from_df(pd.DataFrame())
                return

            # 2. 生成唯一临时文件名
            temp_filename = f"temp_{uuid.uuid4().hex}.jsonl"
            raw_file = RAW_NEWS_DIR / temp_filename

            # 3. 写入 raw_news 目录（供 agent1 读取）
            with open(raw_file, "w", encoding="utf-8") as f:
                for idx, news in enumerate(important_news):
                    # ✅ 正确处理 dict 类型的新闻
                    title = news.get('title', '').strip()
                    content_raw = news.get('content', '').strip()
                    
                    # 清理 HTML（避免 <p>, <br> 干扰去重和 LLM）
                    clean_content = re.sub(r'<[^>]+>', '', content_raw).strip()
                    final_content = clean_content or title  # 兜底
                    
                    item = {
                        "id": str(news.get("id", f"temp_{idx}")),
                        "title": title,
                        "content": final_content,
                        "source": "blockbeats",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"✅ 写入 {len(important_news)} 条新闻到 {raw_file.name}")

            # 4. 调用 agent1 主流程
            process_news_stream()

            # 5. 从 agent1 输出文件构建结构化 DataFrame
            df_structured = self._build_structured_news_from_agent1_output()

            # 6. 保存并分析
            self.news_data['structured'] = df_structured
            self.market_sentiment = self._analyze_market_sentiment_from_df(df_structured)

            # 7. 清理临时文件
            try:
                raw_file.unlink()
                print(f"🗑️  已清理临时文件: {raw_file.name}")
            except Exception as e:
                print(f"⚠️  无法删除临时文件: {e}")

            # 8. 打印摘要
            self._print_news_summary()

        except Exception as e:
            print(f"❌ 新闻数据初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.news_data = {'structured': pd.DataFrame(), 'error': str(e)}
            self.market_sentiment = self._analyze_market_sentiment_from_df(pd.DataFrame())
    
    def _build_structured_news_from_agent1_output(self) -> pd.DataFrame:
        """从 agent1 生成的 abstract_map.json 构建结构化 DataFrame"""
        if not ABSTRACT_MAP_FILE.exists():
            return pd.DataFrame()

        with open(ABSTRACT_MAP_FILE, "r", encoding="utf-8") as f:
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