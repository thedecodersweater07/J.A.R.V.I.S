import logging
from typing import Dict, Any
import imgui
from .screens.base_screen import BaseScreen
from .screens.login_screen import LoginScreen
from .screens.chat_screen import ChatScreen
from .screens.settings_screen import SettingsScreen
from ui.rendering import RendererFactory, RenderMode
from security.auth.auth_service import AuthService
from security.config.security_config import SecurityConfig
from core.session import SessionManager
from core.constants import SECURITY
from llm.pipeline import LLMPipeline

logger = logging.getLogger(__name__)

class Screen:
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS Interface"):
        self.width = width
        self.height = height
        self.title = title
        self.renderer = None
        self.render_mode = RenderMode.OPENGL
        self.is_initialized = False
        self.auth_service = None
        self.session_manager = SessionManager()
        self.llm_pipeline = LLMPipeline()
        self.active_screen = None
        self.screens = {}
        
    def init(self) -> bool:
        try:
            self._init_imgui()
            # Try OpenGL first
            self.renderer = RendererFactory.create_renderer(
                RenderMode.OPENGL,
                {"width": self.width, "height": self.height, "title": self.title}
            )
            self.is_initialized = self.renderer.init()
        except ImportError as e:
            logger.warning(f"OpenGL not available: {e}, falling back to text mode")
            self.render_mode = RenderMode.TEXT
            self.renderer = RendererFactory.create_renderer(
                RenderMode.TEXT,
                {"width": self.width, "height": self.height}
            )
            self.is_initialized = self.renderer.init()
        
        self._init_security()
        self._init_screens()
        return self.is_initialized

    def _init_imgui(self):
        """Initialize ImGui context"""
        imgui.create_context()
        imgui.set_current_context(imgui.get_current_context())
        
    def _init_security(self):
        security_config = SecurityConfig(
            jwt_secret=SECURITY["jwt_secret"],
            token_expiry_hours=SECURITY["token_expiry_hours"],
            max_login_attempts=SECURITY["max_login_attempts"],
            lockout_duration_minutes=SECURITY["lockout_duration_minutes"]
        )
        self.auth_service = AuthService(security_config)
        
    def _init_screens(self):
        self.register_screen("login", LoginScreen(self.auth_service))
        self.register_screen("chat", ChatScreen(self.llm_pipeline))
        self.register_screen("settings", SettingsScreen())
        self.active_screen = "login"
        
    def register_screen(self, name: str, screen: 'BaseScreen'):
        self.screens[name] = screen
        
    def switch_screen(self, screen_name: str):
        if screen_name in self.screens:
            if self.session_manager.validate_session() or screen_name == "login":
                self.active_screen = screen_name
                
    def render(self, frame_data: Dict[str, Any]) -> None:
        if not self.is_initialized:
            return
            
        current_screen = self.screens.get(self.active_screen)
        if current_screen:
            with self.renderer.frame():
                current_screen.render(frame_data)
                
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        current_screen = self.screens.get(self.active_screen)
        if current_screen:
            current_screen.handle_input(input_data)

    def cleanup(self) -> None:
        if self.renderer:
            self.renderer.cleanup()
        self.is_initialized = False
        self.session_manager.cleanup()
        try:
            if imgui.get_current_context():
                imgui.destroy_context()
        except Exception as e:
            logger.error(f"Error destroying ImGui context: {e}")
