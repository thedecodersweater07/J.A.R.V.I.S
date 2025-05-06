import logging
from typing import Dict, Any, Optional
import imgui
from contextlib import contextmanager
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

class UIState:
    def __init__(self):
        self.imgui_initialized = False
        self.context = None
        self.renderer = None
        self.is_initialized = False

class Screen:
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS Interface"):
        self.width = width
        self.height = height
        self.title = title
        self.ui_state = UIState()
        self.auth_service = None
        self.session_manager = SessionManager()
        self.llm_pipeline = LLMPipeline()
        self.active_screen = "login"
        self.screens = {}
        self.interrupt_received = False

    def init(self) -> bool:
        try:
            if not self._init_imgui():
                logger.error("Failed to initialize ImGui")
                return False

            # Try OpenGL first
            self.ui_state.renderer = RendererFactory.create_renderer(
                RenderMode.OPENGL,
                {"width": self.width, "height": self.height, "title": self.title}
            )
            self.ui_state.is_initialized = self.ui_state.renderer.init()
        except ImportError as e:
            logger.warning(f"OpenGL not available: {e}, falling back to text mode")
            self.ui_state.renderer = RendererFactory.create_renderer(
                RenderMode.TEXT,
                {"width": self.width, "height": self.height}
            )
            self.ui_state.is_initialized = self.ui_state.renderer.init()
        except Exception as e:
            logger.error(f"Screen initialization failed: {e}")
            return False

        self._init_security()
        self._init_screens()
        return self.ui_state.is_initialized

    def _init_imgui(self) -> bool:
        """Initialize ImGui with proper error handling"""
        try:
            if not self.ui_state.imgui_initialized:
                self.ui_state.context = imgui.create_context()
                imgui.set_current_context(self.ui_state.context)
                self.ui_state.imgui_initialized = True
            return True
        except Exception as e:
            logger.error(f"ImGui initialization failed: {e}")
            return False

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

    @contextmanager
    def frame(self):
        """Context manager for ImGui frames"""
        try:
            if self.ui_state.renderer and self.ui_state.imgui_initialized:
                self.ui_state.renderer.process_inputs()
                imgui.new_frame()
                yield
                imgui.render()
                self.ui_state.renderer.render(imgui.get_draw_data())
        except Exception as e:
            logger.error(f"Frame error: {e}")
            raise

    def render(self, frame_data: Dict[str, Any]) -> None:
        """Improved render with error handling"""
        if not self.ui_state.is_initialized:
            return

        try:
            with self.frame():
                current_screen = self.screens.get(self.active_screen)
                if current_screen:
                    current_screen.render(frame_data)
        except Exception as e:
            logger.error(f"Render error: {e}")

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        current_screen = self.screens.get(self.active_screen)
        if current_screen:
            current_screen.handle_input(input_data)

    def add_message(self, message: str, is_user: bool) -> None:
        """Add message to current screen"""
        try:
            current_screen = self.screens.get(self.active_screen)
            if current_screen and hasattr(current_screen, 'add_message'):
                current_screen.add_message(message, is_user)
        except Exception as e:
            logger.error(f"Error adding message: {e}")

    def cleanup(self) -> None:
        """Enhanced cleanup with proper ImGui context handling"""
        try:
            if self.ui_state.renderer:
                self.ui_state.renderer.cleanup()
                self.ui_state.renderer = None

            if self.ui_state.imgui_initialized:
                if imgui.get_current_context() == self.ui_state.context:
                    imgui.set_current_context(None)
                    imgui.destroy_context(self.ui_state.context)
                self.ui_state.context = None
                self.ui_state.imgui_initialized = False

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            self.ui_state.is_initialized = False
            self.session_manager.cleanup()
