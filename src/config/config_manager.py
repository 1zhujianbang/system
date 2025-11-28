import yaml
from pydantic import BaseModel, ValidationError, field_validator, Field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import warnings

# 用户配置
class UserConfig(BaseModel):
    trading_pairs: List[str] = Field(..., min_length=1, description="至少需要一个交易对")
    risk_appetite: str = Field(..., pattern="^(conservative|保守|moderate|中性|aggressive|激进)$", description="风险偏好必须是 conservative/保守/moderate/中性/aggressive/激进")
    cash: float = Field(..., gt=0, description="资金必须大于0")
    base_currency: str = Field(..., min_length=3, max_length=5, description="基础货币代码长度3-5")
    trading_mode: str = Field(..., pattern="^(paper|live)$", description="交易模式必须是 paper/live")
    auto_trading: bool

    @field_validator('cash')
    @classmethod
    def validate_cash(cls, v):
        if v < 100:
            warnings.warn("资金较低，建议至少100以上", UserWarning)
        return v

# 超参数配置
class HyperParameters(BaseModel):
    sequence_length: int = Field(..., ge=1, le=1000, description="序列长度必须在1-1000之间")
    batch_size: int = Field(..., ge=1, le=1024, description="批大小必须在1-1024之间")
    learning_rate: float = Field(..., gt=0, le=1, description="学习率必须在0-1之间")
    dropout_rate: float = Field(..., ge=0, le=1, description="dropout率必须在0-1之间")

    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v):
        default_lr = 0.001
        if v != default_lr:
            warnings.warn(f"学习率使用非默认值 {v}，默认值为 {default_lr}", UserWarning)
        return v

    @field_validator('dropout_rate')
    @classmethod
    def validate_dropout_rate(cls, v):
        default_dropout = 0.2
        if v != default_dropout:
            warnings.warn(f"dropout率使用非默认值 {v}，默认值为 {default_dropout}", UserWarning)
        return v

# 模型配置
class ModelConfig(BaseModel):
    model_name: str = Field(..., min_length=1, description="模型名称不能为空")
    model_type: str = Field(..., pattern="^(LSTM|lstm|GRU|gru|transformer|Transformer|CNN|cnn|Ensemble|ensemble)$", description="模型类型必须是 LSTM/lstm/GRU/gru/transformer/Transformer/CNN/cnn/Ensemble/ensemble")
    data_window: int = Field(..., ge=10, le=10000, description="数据窗口大小必须在10-10000之间")
    prediction_target: str = Field(..., min_length=1, description="预测目标不能为空")
    prediction_horizon: int = Field(..., ge=1, le=100, description="预测范围必须在1-100之间")
    features: List[str] = Field(..., min_length=1, description="至少需要一个特征")
    hyperparameters: HyperParameters

# 仓位管理
class PositionManagement(BaseModel):
    max_position_size: float = Field(..., gt=0, le=1, description="最大单仓位比例必须在0-1之间")
    max_total_position: float = Field(..., gt=0, le=5, description="最大总仓位比例必须在0-5之间")
    min_trade_amount: float = Field(..., gt=0, description="最小交易金额必须大于0")
    enable_leverage: bool
    leverage: int = Field(..., ge=1, le=100, description="杠杆必须在1-100之间")

    @field_validator('leverage')
    @classmethod
    def validate_leverage(cls, v, values):
        if v > 10 and not values.get('enable_leverage', False):
            warnings.warn("高杠杆使用但未启用杠杆交易", UserWarning)
        return v

# 止损配置
class StopLoss(BaseModel):
    enabled: bool
    type: str = Field(..., pattern="^(fixed|trailing|atr)$", description="止损类型必须是 fixed/trailing/atr")
    fixed_stop_loss: float = Field(..., ge=0, le=1, description="固定止损比例必须在0-1之间")
    trailing_stop_loss: float = Field(..., ge=0, le=1, description="移动止损比例必须在0-1之间")
    atr_stop_multiplier: float = Field(..., ge=0, le=5, description="ATR止损乘数必须在0-5之间")

class Levels(BaseModel):
    profit: float = Field(..., gt=0, le=10, description="止盈金额必须大于0")
    close_percent: float = Field(..., gt=0, le=1, description="止盈百分比必须大于0")

# 部分止盈
class PartialTakeProfit(BaseModel):
    enabled: bool
    levels: List[Levels]

# 止盈配置
class TakeProfit(BaseModel):
    enabled: bool
    profit_target: float = Field(..., gt=0, le=10, description="盈利目标必须在0-10之间")
    partial_take_profit: PartialTakeProfit

# 每日限制
class DailyLimits(BaseModel):
    max_daily_loss: float = Field(..., ge=0, description="最大日亏损金额必须大于0")
    max_daily_loss_percent: float = Field(..., ge=0, le=100, description="最大日亏损百分比必须在0-100之间")
    max_daily_trades: int = Field(..., ge=0, description="最大日交易数必须大于0")

# 风险配置
class RiskConfig(BaseModel):
    position_management: PositionManagement
    stop_loss: StopLoss
    take_profit: TakeProfit
    daily_limits: DailyLimits

# API配置
class Api(BaseModel):
    api_key: str = Field(..., min_length=1, description="API密钥不能为空")
    api_secret: str = Field(..., min_length=1, description="API密钥不能为空")
    sandbox_mode: bool = True

    @field_validator('sandbox_mode')
    @classmethod
    def validate_sandbox_mode(cls, v):
        if not v:
            warnings.warn("生产模式启用，请确保API密钥安全", UserWarning)
        return v

# 网络配置
class Network(BaseModel):
    timeout: int = Field(..., ge=1, le=60, description="超时时间必须在1-60秒之间")
    retries: int = Field(..., ge=0, le=10, description="重试次数必须在0-10之间")
    rate_limit: int = Field(..., ge=1, le=1000, description="速率限制必须在1-1000之间")

# 交易所配置
class ExchangeConfig(BaseModel):
    exchange_name: str = Field(..., pattern="^(binance|okx|huobi|bybit)$", description="交易所必须是 binance/okx/huobi/bybit")
    api: Api
    network: Network

# 技术指标配置
class RSIConfig(BaseModel):
    period: int = Field(..., ge=1, le=100, description="RSI周期必须在1-100之间")
    enabled: bool

class MACDConfig(BaseModel):
    fast_period: int = Field(..., ge=1, le=50, description="MACD快线周期必须在1-50之间")
    slow_period: int = Field(..., ge=1, le=100, description="MACD慢线周期必须在1-100之间")
    signal_period: int = Field(..., ge=1, le=50, description="MACD信号线周期必须在1-50之间")
    enabled: bool

class BollingerBandsConfig(BaseModel):
    period: int = Field(..., ge=1, le=100, description="布林带周期必须在1-100之间")
    std_dev: int = Field(..., ge=1, le=5, description="布林带标准差必须在1-5之间")
    enabled: bool

class ATRConfig(BaseModel):
    period: int = Field(..., ge=1, le=100, description="ATR周期必须在1-100之间")
    enabled: bool

class TechnicalIndicators(BaseModel):
    rsi: RSIConfig
    macd: MACDConfig
    bollinger_bands: BollingerBandsConfig
    atr: ATRConfig

class NormalizationConfig(BaseModel):
    method: str = Field(..., pattern="^(minmax|zscore|robust)$", description="标准化方法必须是 minmax/zscore/robust")
    enabled: bool = True

    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        default_method = "zscore"
        if v != default_method:
            warnings.warn(f"标准化方法使用非默认值 {v}，默认值为 {default_method}", UserWarning)
        return v

class FeatureEngineering(BaseModel):
    technical_indicators: TechnicalIndicators
    normalization: NormalizationConfig

class DataConfig(BaseModel):
    data_source: str = Field(..., pattern="^(exchange|database|csv)$", description="数据源必须是 exchange/database/csv")
    proxy: Optional[str] = Field(None, description="代理设置（如有需要）")
    sandbox: bool = Field(default=False, description="交易所数据沙盒模式，默认使用实盘数据,只在data_source为exchange时该参数有效")
    timeframe: str = Field(..., pattern="^(1m|5m|15m|1h|4h|1d|1M|5M|15M|1H|4H|1D)$", description="时间框架必须是 1m/5m/15m/1h/4h/1d/1M/5M/15M/1H/4H/1D")
    historical_days: int = Field(..., ge=1, le=3650, description="历史天数必须在1-3650之间")
    update_interval: int = Field(..., ge=1, le=3600, description="更新间隔必须在1-3600秒之间")
    feature_engineering: FeatureEngineering

    @field_validator('sandbox')
    @classmethod
    def validate_sandbox(cls, v, info):
        """验证沙盒模式"""
        data_source = info.data.get('data_source')
        if v and data_source != 'exchange':
            warnings.warn("沙盒模式这一参数只在data_source为exchange时有效", UserWarning)
        return v

    @field_validator('historical_days')
    @classmethod
    def validate_historical_days(cls, v):
        if v < 30:
            warnings.warn("历史数据天数较少，可能影响模型性能", UserWarning)
        return v

# 交易策略配置
class SignalGeneration(BaseModel):
    confidence_threshold: float = Field(..., ge=0, le=1, description="置信度阈值必须在0-1之间")
    min_signal_strength: float = Field(..., ge=0, le=1, description="最小信号强度必须在0-1之间")
    confirmation_period: int = Field(..., ge=0, le=10, description="确认周期必须在0-10之间")

class VolatilityFilter(BaseModel):
    enabled: bool
    max_volatility: float = Field(..., gt=0, le=1, description="最大波动率必须在0-1之间")

class EntryConditions(BaseModel):
    ai_signal_enabled: bool = True
    technical_confirmation: bool = True
    market_regime_filter: bool = True
    volatility_filter: VolatilityFilter

class TimeBasedExit(BaseModel):
    enabled: bool = False
    max_holding_hours: int = Field(..., ge=1, le=720, description="最大持仓时间必须在1-720小时之间")

class ExitConditions(BaseModel):
    ai_exit_signal: bool = True
    time_based_exit: TimeBasedExit
    technical_exit: bool = True

class StrategyConfig(BaseModel):
    strategy_name: str = Field(..., min_length=1, description="策略名称不能为空")
    signal_generation: SignalGeneration
    entry_conditions: EntryConditions
    exit_conditions: ExitConditions

# 监控与日志配置
class LoggingConfig(BaseModel):
    level: str = Field(..., pattern="^(DEBUG|INFO|WARNING|ERROR)$", description="日志级别必须是 DEBUG/INFO/WARNING/ERROR")
    file_path: str
    console_output: bool = True

class PerformanceMonitoring(BaseModel):
    enabled: bool = True
    metrics: List[str]
    report_interval: int = Field(..., ge=1, le=720, description="报告间隔必须在1-720小时之间")

class EmailAlerts(BaseModel):
    enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = Field(ge=1, le=65535, description="SMTP端口必须在1-65535之间")
    sender_email: str = ""
    sender_password: str = ""
    receiver_emails: List[str] = []

class WechatAlerts(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""

class AlertsConfig(BaseModel):
    email_alerts: EmailAlerts
    wechat_alerts: WechatAlerts
    alert_conditions: List[str]

class MonitoringConfig(BaseModel):
    logging: LoggingConfig
    performance: PerformanceMonitoring
    alerts: AlertsConfig

# 备份与恢复配置
class AutoBackup(BaseModel):
    enabled: bool = True
    interval_hours: int = Field(..., ge=1, le=720, description="备份间隔必须在1-720小时之间")
    keep_backups: int = Field(..., ge=1, le=100, description="保留备份数必须在1-100之间")

class BackupConfig(BaseModel):
    auto_backup: AutoBackup
    backup_items: List[str]
    backup_path: str

# 高级配置
class ParallelProcessing(BaseModel):
    enabled: bool = True
    max_workers: int = Field(..., ge=1, le=64, description="最大工作线程数必须在1-64之间")

class MemoryManagement(BaseModel):
    max_memory_usage: str = Field(..., pattern="^[0-9]+[MG]$", description="内存使用格式如 2G, 512M")
    clear_cache_hours: int = Field(..., ge=1, le=24, description="清理缓存间隔必须在1-24小时之间")

class AdvancedConfig(BaseModel):
    parallel_processing: ParallelProcessing
    memory_management: MemoryManagement
    debug_mode: bool = False
    random_seed: int = Field(..., ge=0, le=9999, description="随机种子必须在0-9999之间")

# 主配置类
class TradingConfig(BaseModel):
    user_config: UserConfig
    modeL_config: ModelConfig
    risk_config: RiskConfig
    exchange_config: ExchangeConfig
    data_config: DataConfig
    strategy_config: StrategyConfig
    monitoring_config: MonitoringConfig
    backup_config: BackupConfig
    advanced_config: AdvancedConfig
    config_version: str = Field(..., pattern="^[0-9]+\\.[0-9]+\\.[0-9]+$", description="配置版本格式必须为 x.x.x")

    @classmethod
    def from_yaml(cls, file_path: str = None) -> 'TradingConfig':
        """从YAML文件创建配置实例"""
        if file_path is None:
            file_path = ConfigManager().config_path
        else:
            file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        return validate_config(config_data)
    
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self._find_config_file()
    
    def _find_config_file(self) -> Path:
        """自动查找配置文件"""
        # 从当前文件开始向上查找
        current_dir = Path(__file__).parent
        search_paths = []
        
        for i in range(5):  # 最多向上5层
            config_dir = current_dir / 'config'
            config_file = config_dir / 'config.yaml'
            search_paths.append(config_file)
            
            if config_file.exists():
                return config_file
            
            # 检查当前目录
            current_config = current_dir / 'config.yaml'
            search_paths.append(current_config)
            if current_config.exists():
                return current_config
            
            if current_dir.parent == current_dir:  # 到达根目录
                break
                
            current_dir = current_dir.parent
        
        # 如果自动查找失败，提供有用的错误信息
        print("❌ 自动查找配置文件失败，尝试了以下路径:")
        for path in search_paths:
            exists = "✓" if path.exists() else "✗"
            print(f"  {exists} {path.absolute()}")
        
        raise FileNotFoundError(
            "请确保配置文件存在，或使用 ConfigManager('path/to/config.yaml') 指定路径"
        )
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置数据"""
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def validate_config(self) -> TradingConfig:
        """验证配置"""
        config_data = self.load_config()
        return validate_config(config_data)

def validate_config(config_data: dict) -> TradingConfig:
    """
    验证配置数据并返回配置对象
    """
    try:
        warnings.simplefilter("always")
        config = TradingConfig(**config_data)
        print("✅ 配置验证通过!")
        return config
        
    except ValidationError as e:
        print("❌ 配置验证失败:")
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error['loc'])
            print(f"  字段: {field}")
            print(f"  错误: {error['msg']}")
            print("  " + "-" * 50)
        raise
    
    except Exception as e:
        print(f"❌ 配置验证过程中发生未知错误: {e}")
        raise

# 使用示例
if __name__ == '__main__':
    try:
        # 方法1: 自动查找
        config = TradingConfig.from_yaml()
        
        # 方法2: 使用管理器
        # manager = ConfigManager()
        # config = manager.validate_config()
        
        print("🎉 配置验证完成!")
        print(f"版本: {config.config_version}")
        print(f"交易对: {config.user_config.trading_pairs}")
        
    except Exception as e:
        print(f"💥 配置处理失败: {e}")