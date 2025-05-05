from typing import Dict, List, Any
from .base_agent import BaseAgent, AgentState
import asyncio

class AgentCoordinator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = asyncio.Queue()
        
    def register_agent(self, agent: BaseAgent) -> None:
        self.agents[agent.config.name] = agent
        
    async def dispatch_task(self, task: Dict[str, Any]) -> None:
        target_agent = self._select_agent(task)
        if target_agent:
            await self.task_queue.put((target_agent, task))
            
    def _select_agent(self, task: Dict[str, Any]) -> BaseAgent:
        # Agent selection logic based on task requirements and agent state
        available_agents = [a for a in self.agents.values() 
                          if a.state == AgentState.IDLE]
        if available_agents:
            return max(available_agents, key=lambda x: x.config.priority)
        return None
