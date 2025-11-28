from ..config.config_manager import TradingConfig
from ..models.model_loader import ModelLoader
from ..data.data_collector import OKXMarketClient

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

        # 初始化 OKX 客户端
        self.okx_client = OKXMarketClient(config.user_config, config.data_config)
        
        # 数据存储
        self.market_data = {}  # 历史K线数据
        self.realtime_data = {}  # 实时行情数据
        self.technical_data = {}  # 技术指标数据

    def initialize(self):
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

            # 4. 初始化数据流 (伪代码)
            # self.data_stream = DataStream(self.config.user_config.trading_pairs)

            # 5. 标记为就绪状态
            self.is_ready = True
            print("AI Trading Agent is now READY.")

        except Exception as e:
            print(f"❌ Agent初始化失败: {type(e).__name__}: {str(e)}")
            import traceback
            print("🔍 详细堆栈跟踪:")
            traceback.print_exc()
            raise

    def get_status(self):
        return {
            "is_ready": self.is_ready,
            "cash": self.config.user_config.cash,
            "risk_appetite": self.config.user_config.risk_appetite,
            "model_used": self.config.modeL_config.model_name
        }
    
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
        self.realtime_data = self.okx_client.get_realtime_data()
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
        for pair, ticker in self.realtime_data.items():
            if ticker:
                price = float(ticker.get('last', 0))
                change_24h = float(ticker.get('24hChange', 0))
                print(f"     {pair}: {price:.2f} ({change_24h:+.2f}%)")