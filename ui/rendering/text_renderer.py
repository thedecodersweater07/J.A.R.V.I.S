from typing import Dict, Any
import logging
from .renderer_base import RendererBase

logger = logging.getLogger(__name__)

class TextRenderer(RendererBase):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.width = config.get("width", 80)
        self.height = config.get("height", 24)
        
    def init(self) -> bool:
        """Initialize text display"""
        try:
            print("\033[2J\033[H")  # Clear screen
            return True
        except Exception as e:
            logger.error(f"Text renderer initialization failed: {e}")
            return False
            
    def render(self, frame_data: Dict[str, Any]) -> None:
        """Render frame in text mode"""
        print("\033[H")  # Move cursor to home
        print("-" * self.width)
        
        if "chat_history" in frame_data:
            self._render_chat(frame_data["chat_history"])
        if "status" in frame_data:
            self._render_status(frame_data["status"])
            
    def _render_chat(self, history):
        for msg in history[-10:]:  # Show last 10 messages
            print(f"{msg['sender']}: {msg['text']}")
            
    def _render_status(self, status):
        print("-" * self.width)
        print(f"Status: {status}")
        
    def cleanup(self) -> None:
        """Clean up text display"""
        print("\033[2J\033[H")  # Clear screen
        
    def begin_frame(self):
        """Start frame rendering"""
        print("\033[H")  # Move cursor to home
        
    def end_frame(self):
        """Finish frame rendering"""
        print()  # Add newline after frame
