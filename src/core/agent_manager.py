"""
Agent管理器 (BE-01)
提供Agent生命周期管理和协作功能
"""

import asyncio
from typing import Dict, List, Any, Optional
from .agent_base import BaseAgent, AgentRegistry, AgentStatus
from .logging import LoggerManager
from ..utils.llm_utils import AsyncExecutor


class AgentManager:
    """
    Agent管理器
    负责Agent的创建、生命周期管理和协作
    """

    def __init__(self):
        self.logger = LoggerManager.get_logger("AgentManager")
        self.agents: Dict[str, BaseAgent] = {}
        self.executor = AsyncExecutor()

    async def create_agent(self, agent_name: str, agent_class_name: str, **kwargs) -> Optional[str]:
        """
        创建Agent实例

        Args:
            agent_name: Agent实例名称
            agent_class_name: Agent类名
            **kwargs: Agent初始化参数

        Returns:
            Agent实例ID，失败返回None
        """
        try:
            agent_class = AgentRegistry.get_agent_class(agent_class_name)
            if not agent_class:
                self.logger.error(f"未找到Agent类: {agent_class_name}")
                return None

            # 创建Agent实例
            agent = agent_class(**kwargs)
            agent_id = f"{agent_class_name}_{agent_name}"

            # 初始化Agent
            success = await agent.initialize()
            if not success:
                self.logger.error(f"Agent {agent_id} 初始化失败")
                return None

            # 注册到管理器
            self.agents[agent_id] = agent
            self.logger.info(f"Agent {agent_id} 创建成功")
            return agent_id

        except Exception as e:
            self.logger.error(f"创建Agent失败: {e}")
            return None

    async def execute_agent(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """
        执行Agent任务

        Args:
            agent_id: Agent实例ID
            **kwargs: 执行参数

        Returns:
            执行结果
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return {"status": "error", "error": f"Agent {agent_id} 不存在"}

        try:
            result = await agent.execute(**kwargs)
            return result
        except Exception as e:
            self.logger.error(f"执行Agent {agent_id} 失败: {e}")
            return {"status": "error", "error": str(e)}

    async def execute_agents_parallel(self, agent_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        并行执行多个Agent

        Args:
            agent_configs: Agent执行配置列表
                格式: [{"agent_id": "agent1", "params": {...}}, ...]

        Returns:
            执行结果列表
        """
        async def execute_single(config):
            agent_id = config["agent_id"]
            params = config.get("params", {})
            return await self.execute_agent(agent_id, **params)

        # 使用AsyncExecutor进行并发执行
        results = await self.executor.run_concurrent_tasks(
            tasks=[lambda c=config: execute_single(c) for config in agent_configs],
            concurrency=len(agent_configs)  # 所有任务并发执行
        )

        return results

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent状态"""
        agent = self.agents.get(agent_id)
        if agent:
            return agent.get_status()
        return None

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """列出所有Agent状态"""
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self.agents.items()
        }

    async def pause_agent(self, agent_id: str) -> bool:
        """暂停Agent"""
        agent = self.agents.get(agent_id)
        if agent:
            await agent.pause()
            return True
        return False

    async def resume_agent(self, agent_id: str) -> bool:
        """恢复Agent"""
        agent = self.agents.get(agent_id)
        if agent:
            await agent.resume()
            return True
        return False

    async def stop_agent(self, agent_id: str) -> bool:
        """停止Agent"""
        agent = self.agents.get(agent_id)
        if agent:
            await agent.stop()
            return True
        return False

    async def save_agent_states(self) -> Dict[str, str]:
        """
        保存所有Agent状态

        Returns:
            Agent状态映射 {agent_id: state_json}
        """
        states = {}
        for agent_id, agent in self.agents.items():
            try:
                state = agent.save_state()
                states[agent_id] = state
            except Exception as e:
                self.logger.error(f"保存Agent {agent_id} 状态失败: {e}")

        self.logger.info(f"已保存 {len(states)} 个Agent状态")
        return states

    async def load_agent_states(self, states: Dict[str, str]):
        """
        加载Agent状态

        Args:
            states: Agent状态映射 {agent_id: state_json}
        """
        loaded_count = 0
        for agent_id, state_json in states.items():
            agent = self.agents.get(agent_id)
            if agent:
                try:
                    agent.load_state(state_json)
                    loaded_count += 1
                except Exception as e:
                    self.logger.error(f"加载Agent {agent_id} 状态失败: {e}")

        self.logger.info(f"已加载 {loaded_count} 个Agent状态")

    def get_registered_agents(self) -> Dict[str, Dict[str, Any]]:
        """获取所有注册的Agent类信息"""
        return AgentRegistry.list_agents()

    async def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理Agent管理器资源")

        # 停止所有Agent
        stop_tasks = []
        for agent_id, agent in self.agents.items():
            if agent.status != AgentStatus.STOPPED:
                stop_tasks.append(self.stop_agent(agent_id))

        if stop_tasks:
            await asyncio.gather(*stop_tasks)

        # 清空Agent列表
        self.agents.clear()
        self.logger.info("Agent管理器资源清理完成")


class AgentWorkflow:
    """
    Agent工作流
    支持定义Agent执行序列和依赖关系
    """

    def __init__(self, manager: AgentManager):
        self.manager = manager
        self.logger = LoggerManager.get_logger("AgentWorkflow")
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, agent_id: str, params: Dict[str, Any], depends_on: Optional[List[str]] = None):
        """
        添加工作流步骤

        Args:
            agent_id: Agent实例ID
            params: 执行参数
            depends_on: 依赖的步骤ID列表
        """
        step_id = f"step_{len(self.steps)}"
        self.steps.append({
            "id": step_id,
            "agent_id": agent_id,
            "params": params,
            "depends_on": depends_on or [],
            "status": "pending"
        })

    async def execute(self) -> Dict[str, Any]:
        """
        执行工作流

        Returns:
            执行结果
        """
        self.logger.info(f"开始执行Agent工作流，共 {len(self.steps)} 步")

        results = {}
        completed_steps = set()

        for step in self.steps:
            step_id = step["id"]

            # 检查依赖
            if not all(dep in completed_steps for dep in step["depends_on"]):
                self.logger.warning(f"步骤 {step_id} 依赖未满足，跳过")
                continue

            # 执行步骤
            agent_id = step["agent_id"]
            params = step["params"]

            self.logger.info(f"执行步骤 {step_id}: Agent {agent_id}")
            result = await self.manager.execute_agent(agent_id, **params)

            results[step_id] = result
            if result.get("status") == "success":
                completed_steps.add(step_id)
                step["status"] = "completed"
            else:
                step["status"] = "failed"
                self.logger.error(f"步骤 {step_id} 执行失败")

        self.logger.info("Agent工作流执行完成")
        return {
            "status": "completed",
            "results": results,
            "completed_steps": len(completed_steps),
            "total_steps": len(self.steps)
        }


# 全局Agent管理器实例
_global_manager = None

def get_agent_manager() -> AgentManager:
    """获取全局Agent管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = AgentManager()
    return _global_manager
