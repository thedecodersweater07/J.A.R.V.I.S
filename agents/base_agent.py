from typing import Dict, Any
import logging

class BaseAgent:
    """Base class for all agents with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self):
        """Initialize agent resources and connections."""
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
    def shutdown(self):
        """Clean up resources when shutting down."""
        self.logger.info(f"Shutting down {self.__class__.__name__}")
        
    def update_config(self, new_config: Dict[str, Any]):
        """Update agent configuration."""
        self.config.update(new_config)
