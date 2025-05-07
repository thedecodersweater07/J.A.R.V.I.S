from typing import List, Dict, Any
import logging
from pathlib import Path
from contextlib import contextmanager
from core.logging import get_logger, logger
from core.constants import OPENGL_AVAILABLE
from core.config import ConfigValidator
from core.session import SessionManager
from core.command.command_parser import CommandParser
from core.command.executor import CommandExecutor
from core.config.models import ChatConfig, UIConfig

import imgui
from .base_screen import BaseScreen
from ..components.chat_input import ChatInput
from ..components.chat_history import ChatHistory
from ..components.status_bar import StatusBar

class ChatScreen(BaseScreen):
    def __init__(self, llm_pipeline):
        super().__init__()
        self.llm = llm_pipeline
        self.config = ChatConfig()
        self.ui_config = UIConfig()
        self.chat_input = ChatInput(self.config.input_placeholder)
        self.chat_history = ChatHistory(max_messages=self.config.history_max_messages)
        self.status_bar = StatusBar()
        self.initialized = False
        
    def init(self) -> bool:
        """Initialize chat screen"""
        try:
            if not self.chat_input.init() or not self.chat_history.init():
                return False
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Chat screen initialization failed: {e}")
            return False
            
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle input events"""
        if not self.initialized:
            return
            
        if input_data.get("type") == "quit":
            self.cleanup()
            return
            
        if input_data.get("type") == "message":
            self._process_message(input_data["content"])
            
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.chat_input:
                self.chat_input.cleanup()
            if self.chat_history:
                self.chat_history.cleanup()
            if self.status_bar:
                self.status_bar.cleanup()
        except Exception as e:
            logger.error(f"Error during chat screen cleanup: {e}")
        finally:
            self.initialized = False
            
    def render(self, frame_data: Dict[str, Any]) -> None:
        imgui.set_next_window_size(self.ui_config.width, self.ui_config.height)
        imgui.begin(self.config.window_title, 
                   flags=imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
        
        # Chat history area with scrolling
        self.chat_history.render()
        
        # Input area with auto-focus
        if self.chat_input.render():
            message = self.chat_input.get_text()
            self._process_message(message)
            
        # Status bar
        self.status_bar.render(self.llm.get_status())
        
        imgui.end()

    def _process_message(self, message: str) -> None:
        try:
            self.chat_history.add_user_message(message)
            response = self.llm.process(message)
            self.chat_history.add_assistant_message(response)
        except Exception as e:
            self.status_bar.show_error(f"Error processing message: {e}")
