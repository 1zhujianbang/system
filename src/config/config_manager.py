import yaml
from pydantic import BaseModel, ValidationError, Field
from pathlib import Path
from typing import List, Optional, Dict, Any

# 用户配置（仅保留核心字段）
class UserConfig(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    api_key: str = Field(..., description="API密钥")
    api_secret: str = Field(..., description="API密钥密码")
    risk_preference: str = Field(..., pattern="^(conservative|moderate|aggressive)$", description="分析风险偏好: conservative(保守), moderate(中性), aggressive(激进)")
    symbols: List[str] = Field(..., min_length=1, description="要分析的市场符号列表")

# 模型配置（保留加载所需字段）
class ModelConfig(BaseModel):
    model_name: str = Field(..., min_length=1, description="模型名称不能为空")
    model_type: str = Field(..., pattern="^(LSTM|lstm|GRU|gru|transformer|Transformer|CNN|cnn|Ensemble|ensemble)$", description="模型类型必须是 LSTM/lstm/GRU/gru/transformer/Transformer/CNN/cnn/Ensemble/ensemble")
    data_window: int = Field(..., ge=10, le=10000, description="数据窗口大小必须在10-10000之间")
    prediction_target: str = Field(..., min_length=1, description="预测目标不能为空")
    prediction_horizon: int = Field(..., ge=1, le=100, description="预测范围必须在1-100之间")
    features: List[str] = Field(..., min_length=1, description="至少需要一个特征")

# 数据配置（精简至当前使用字段）
class DataConfig(BaseModel):
    data_source: str = Field(..., pattern="^(exchange|database|csv)$", description="数据源必须是 exchange/database/csv")
    proxy: Optional[str] = Field(None, description="代理设置（如有需要）")
    timeframe: str = Field(..., pattern="^(1m|5m|15m|1h|4h|1d|1M|5M|15M|1H|4H|1D)$", description="时间框架必须是 1m/5m/15m/1h/4h/1d/1M/5M/15M/1H/4H/1D")
    historical_days: int = Field(..., ge=1, le=3650, description="历史天数必须在1-3650之间")
    update_interval: int = Field(..., ge=1, le=3600, description="更新间隔必须在1-3600秒之间")

# 主配置类（仅保留核心模块依赖字段）
class MarketAnalysisConfig(BaseModel):
    user_config: UserConfig
    models_config: ModelConfig
    data_config: DataConfig
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