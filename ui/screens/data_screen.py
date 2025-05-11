from .base_screen import BaseScreen
from db import DatabaseManager
import imgui
from typing import Dict, Any
import logging
import glfw
from OpenGL import GL

logger = logging.getLogger(__name__)

class DataScreen(BaseScreen):
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.metrics = {}
        self.initialized = False
        self.setup_ui()

    def init(self) -> bool:
        try:
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize data screen: {e}")
            return False

    def render(self, frame_data: Dict[str, Any]) -> None:
        imgui.begin("Data Viewer", flags=imgui.WINDOW_NO_COLLAPSE)
        
        if imgui.button("Refresh Data"):
            self.load_data()
            
        imgui.separator()
        
        # Display metrics
        if self.metrics:
            for key, value in self.metrics.items():
                imgui.text(f"{key}: {value}")
        else:
            imgui.text_colored("No data available", 0.5, 0.5, 0.5)
            
        imgui.end()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        if not self.initialized:
            return
            
        if input_data.get("type") == "refresh":
            self.load_data()
    
    def setup_ui(self):
        pass  # Not using tkinter elements anymore
        
    def load_data(self):
        try:
            self.metrics = self.db.load_metrics()
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            self.metrics = {}
