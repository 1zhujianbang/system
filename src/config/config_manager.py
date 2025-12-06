import yaml
from pydantic import BaseModel, ValidationError, field_validator, Field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import warnings

# 用户配置
class UserConfig(BaseModel):
    # 用户账户配置
    user_id: str = Field(..., description="用户唯一标识")
    api_key: str = Field(..., description="API密钥")
    api_secret: str = Field(..., description="API密钥密码")
    
    # 风险偏好配置
    risk_preference: str = Field(..., pattern="^(conservative|moderate|aggressive)$", description="分析风险偏好: conservative(保守), moderate(中性), aggressive(激进)")
    
    # 市场数据配置
    symbols: List[str] = Field(..., min_length=1, description="要分析的市场符号列表")
    data_sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="数据源配置")
    update_interval: int = Field(30, ge=1, description="数据更新间隔（秒）")
    historical_data_length: int = Field(1000, ge=100, description="历史数据长度")

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

# 市场分析风险参数配置
class MarketRiskParams(BaseModel):
    # 波动率分析配置
    volatility: Dict[str, Any] = Field(default_factory=dict, description="波动率分析配置")
    
    # 相关性分析配置
    correlation: Dict[str, Any] = Field(default_factory=dict, description="相关性分析配置")

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
class MarketAnalysisConfig(BaseModel):
    user_config: UserConfig
    models_config: ModelConfig
    data_config: DataConfig
    monitoring_config: MonitoringConfig
    backup_config: BackupConfig
    advanced_config: AdvancedConfig
    market_risk_params: Optional[MarketRiskParams] = Field(default_factory=MarketRiskParams, description="市场分析风险参数")
    config_version: str = Field(..., pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$", description="配置版本格式必须为 x.x.x")

    @classmethod
    def from_yaml(cls, file_path: str = None) -> 'MarketAnalysisConfig':
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
    
    def validate_config(self) -> MarketAnalysisConfig:
        """验证配置"""
        config_data = self.load_config()
        return validate_config(config_data)

def validate_config(config_data: dict) -> MarketAnalysisConfig:
    """
    验证配置数据并返回配置对象
    """
    try:
        warnings.simplefilter("always")
        config = MarketAnalysisConfig(**config_data)
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
        config = MarketAnalysisConfig.from_yaml()
        
        # 方法2: 使用管理器
        # manager = ConfigManager()
        # config = manager.validate_config()
        
        print("🎉 配置验证完成!")
        print(f"版本: {config.config_version}")
        print(f"市场符号: {config.user_config.symbols}")
        
    except Exception as e:
        print(f"💥 配置处理失败: {e}")