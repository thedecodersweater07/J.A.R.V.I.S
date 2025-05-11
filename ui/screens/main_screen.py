import imgui
from typing import Dict, Any
import logging
import glfw
from OpenGL import GL
from .base_screen import BaseScreen

logger = logging.getLogger(__name__)

class MainScreen(BaseScreen):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.width = 800
        self.height = 600

    def init(self) -> bool:
        self.initialized = True
        return True

    def render(self, frame_data: Dict[str, Any]) -> None:
        # Vermijd ImGui-operaties als er geen geldig frame is
        import imgui
        
        try:
            # Centreer het venster met GLFW
            self._center_window()
            
            # Controleer of er een geldige ImGui-context is
            if not imgui.get_current_context():
                logger.error("No valid ImGui context available")
                return
            
            # Eenvoudige rendering zonder complexe styling
            imgui.set_next_window_size(self.width, self.height)
            imgui.begin("JARVIS Main Interface", flags=imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
            
            # Inhoud van het venster
            imgui.text("Welcome to JARVIS")
            imgui.separator()
            imgui.text("Kies een optie om te starten:")
            imgui.spacing()
            
            # Knoppen
            if imgui.button("Chat", width=140):
                self.switch_screen("chat")
            
            imgui.same_line()
            if imgui.button("Instellingen", width=140):
                self.switch_screen("settings")
                
            imgui.same_line()
            if imgui.button("Data", width=140):
                self.switch_screen("data")
            
            imgui.spacing()
            imgui.text("Nova Industries 2025")
            
            # Altijd afsluiten
            imgui.end()
            
        except Exception as e:
            logger.error(f"MainScreen render error: {e}")
    
    def _center_window(self):
        """Centreer het GLFW-venster op het scherm"""
        try:
            window = glfw.get_current_context() if hasattr(glfw, 'get_current_context') else None
            if window:
                monitor = glfw.get_primary_monitor()
                if monitor:
                    vidmode = glfw.get_video_mode(monitor)
                    if vidmode:
                        x = int((vidmode.size.width - self.width) / 2)
                        y = int((vidmode.size.height - self.height) / 2)
                        glfw.set_window_pos(window, x, y)
        except Exception as e:
            logger.error(f"Error centering window: {e}")

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        pass
