from src.config.config_manager import TradingConfig
from src.agents.trading_agent import TradingAgent

def main():
    try:
        # 方法1: 自动加载配置
        config = TradingConfig.from_yaml()
        
        # 方法2: 指定配置文件路径
        # config = TradingConfig.from_yaml('config/user_config.yaml')
        
        # 创建交易Agent
        agent = TradingAgent(config)
        agent.initialize()
        
        print(agent.get_status())
        print("✅ 交易系统启动成功!")
        
        # 进入主循环
        # agent.run()
        
    except Exception as e:
        print(f"💥 系统启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()