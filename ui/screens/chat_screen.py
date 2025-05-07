from typing import List, Dict, Any
import imgui
from .base_screen import BaseScreen
from ..components.chat_input import ChatInput
from ..components.chat_history import ChatHistory
from ..components.status_bar import StatusBar

class ChatScreen(BaseScreen):
    def __init__(self, llm_pipeline):
        super().__init__()
        self.llm = llm_pipeline
        self.chat_input = ChatInput()
        self.chat_history = ChatHistory()
        self.status_bar = StatusBar()
        
    def render(self, frame_data: Dict[str, Any]) -> None:
        imgui.set_next_window_size(self.width, self.height)
        imgui.begin("JARVIS Chat", flags=imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
        
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
