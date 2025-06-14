from typing import Dict, Any, Optional
import asyncio
from dataclasses import dataclass
from enum import Enum
import logging

class AgentState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"

@dataclass
class AgentConfig:
    name: str
    priority: int = 1
    max_concurrent_tasks: int = 1
    timeout: float = 60.0

class BaseAgent:
    """Base class for all agents with common functionality."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState.IDLE
        self.current_tasks = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self):
        """Initialize agent resources and connections."""
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
    def shutdown(self):
        """Clean up resources when shutting down."""
        self.logger.info(f"Shutting down {self.__class__.__name__}")
        
    def update_config(self, new_config: Dict[str, Any]):
        """Update agent configuration."""
        self.config = AgentConfig(**new_config)
        
    async def process(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a task and return results"""
        raise NotImplementedError
        
    async def handle_error(self, error: Exception) -> None:
        """Handle any errors during processing"""
        self.state = AgentState.ERROR
        # Add error handling logic
