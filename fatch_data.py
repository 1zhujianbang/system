from src.config.config_manager import UserConfig, DataConfig
from src.data.data_collector import MarketClient
from src.config.config_manager import MarketAnalysisConfig

config = MarketAnalysisConfig.from_yaml("config/config.yaml")
user_config = config.user_config
data_config = config.data_config

# 创建客户端
client = MarketClient(user_config,data_config)

# 批量导出
results = client.batch_export_historical_data(
    instIds=["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT"],
    output_dir="./models/data/1H/",
    bar="1H",  # 1小时线
    years=5,
    include_technical_indicators=True
)

results = client.batch_export_historical_data(
    instIds=["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT"],
    output_dir="./models/data/1D/",
    bar="1D",  # 1日线
    years=5,
    include_technical_indicators=True
)

results = client.batch_export_historical_data(
    instIds=["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT"],
    output_dir="./models/data/1W/",
    bar="1W",  # 周线数据
    years=5,
    include_technical_indicators=True
)