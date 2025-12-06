# src/config/__init__.py
"""
配置管理模块
提供完整的交易系统配置管理功能
"""

from .config_manager import (
    # 主配置类
    MarketAnalysisConfig,
    ConfigManager,
    validate_config,
    
    # 配置子类
    UserConfig,
    DataConfig,
    ModelConfig,
    MonitoringConfig,
    BackupConfig,
    AdvancedConfig,
    
    # 技术指标配置
    TechnicalIndicators,
    RSIConfig,
    MACDConfig,
    BollingerBandsConfig,
    ATRConfig,
    FeatureEngineering,
    NormalizationConfig,
    
    # 模型配置子类
    HyperParameters,
    
    # 监控配置子类
    LoggingConfig,
    PerformanceMonitoring,
    AlertsConfig,
    EmailAlerts,
    WechatAlerts,
    
    # 备份配置子类
    AutoBackup,
    
    # 高级配置子类
    ParallelProcessing,
    MemoryManagement
)

# 定义 __all__ 来控制导入行为
__all__ = [
    # 主配置类
    'MarketAnalysisConfig',
    'ConfigManager', 
    'validate_config',
    
    # 主要配置类
    'UserConfig',
    'DataConfig',
    'ModelConfig',
    'MonitoringConfig',
    'BackupConfig',
    'AdvancedConfig',
    
    # 技术指标
    'TechnicalIndicators',
    'RSIConfig',
    'MACDConfig', 
    'BollingerBandsConfig',
    'ATRConfig',
    'FeatureEngineering',
    'NormalizationConfig',
    
    # 模型配置
    'HyperParameters'
]

# 包版本信息
__version__ = "1.0.0"