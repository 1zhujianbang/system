import asyncio
from src.config.config_manager import MarketAnalysisConfig
from src.agents.market_analysis_agent import MarketAnalysisAgent

async def main():
    agent = None
    try:
        # 方法1: 自动加载配置
        config = MarketAnalysisConfig.from_yaml()
        
        # 方法2: 指定配置文件路径
        # config = MarketAnalysisConfig.from_yaml('config/user_config.yaml')
        
        # 创建市场分析智能体
        agent = MarketAnalysisAgent(config)
        await agent.initialize()
        
        print(agent.get_status())
        print("✅ 市场分析系统启动成功!")

        return 0
        
    except Exception as e:
        print(f"💥 市场分析系统启动失败: {e}")
        return 1
    
    finally:
        # 关闭所有资源
        if agent:
            await agent.cleanup()
            print("🎯 所有资源已关闭")

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
        exit(0)