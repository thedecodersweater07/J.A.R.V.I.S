import imgui
from typing import Dict, Any
import logging
from .base_screen import BaseScreen

logger = logging.getLogger(__name__)

class MainScreen(BaseScreen):
    def __init__(self, container=None):
        super().__init__(container)
        self.initialized = False
        self.width = 800
        self.height = 600

    def init(self) -> bool:
        self.initialized = True
        return True

    def render(self, frame_data: Dict[str, Any]) -> None:
        imgui.set_next_window_size(self.width, self.height)
        imgui.set_next_window_centered()
        imgui.begin("JARVIS Main Interface", flags=imgui.WINDOW_NO_COLLAPSE)

        imgui.text("Welcome to JARVIS")
        
        if imgui.button("Chat", width=120):
            self.switch_screen("chat")
            
        imgui.same_line()
        if imgui.button("Settings", width=120):
            self.switch_screen("settings")
            
        imgui.same_line()
        if imgui.button("Data", width=120):
            self.switch_screen("data")

        imgui.end()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        pass
