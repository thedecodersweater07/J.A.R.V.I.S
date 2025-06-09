import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BaseScreen(ABC):
    def __init__(self):
        self.parent = None
        self.is_visible = False
        self.initialized = False
        self.width = 800
        self.height = 600
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup UI elements - override in subclasses"""
        pass
        
    @abstractmethod
    def init(self) -> bool:
        """Initialize the screen"""
        pass
        
    @abstractmethod
    def render(self, frame_data: Dict[str, Any]) -> None:
        """Render the screen"""
        pass
        
    @abstractmethod
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        pass
        
    def show(self):
        """Show this screen"""
        self.is_visible = True
        
    def hide(self):
        """Hide this screen"""
        self.is_visible = False
        
    def add_message(self, message: str, is_user: bool = False):
        """Add message to screen - override in subclasses"""
        pass
        
    def cleanup(self) -> None:
        """Clean up resources"""
        self.initialized = False
        
    def set_parent(self, parent):
        """Set the parent screen manager"""
        self.parent = parent
