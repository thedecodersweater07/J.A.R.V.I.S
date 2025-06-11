import logging
from typing import Dict, Any
import time


try:
    import imgui
    IMGUI_AVAILABLE = True
except ImportError:
    IMGUI_AVAILABLE = False
    # Initialize logger first
    logger = logging.getLogger(__name__)
    logger.warning("ImGui not available. Some UI features will be disabled")


# Initialize logger at module level
logger = logging.getLogger(__name__)

class UIState:
    def __init__(self):
        self.imgui_manager = None
        self.renderer = None
        self.is_initialized = False
        self.context = None

class Screen:
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS"):
        self.width = width
        self.height = height
        self.title = title
        self.ui_state = UIState()
        self.should_quit = False
        self.active_screen = "main"
        self.screens = {}
        self.llm = None
        self.model_manager = None
        self.api_client = None
        # Add state for window handling
        self.window_created = False
        self.window_error = None
        self.frame_count = 0
        self.last_frame_time = time.time()

        if not IMGUI_AVAILABLE:
            logger.warning("Screen initialized without ImGui support")
            
    def init(self) -> bool:
        """Initialize the screen and renderers"""
        try:
            if not IMGUI_AVAILABLE:
                logger.error("ImGui required but not available")
                return self._fallback_init()

            # Initialize GLFW
            import glfw
            if not glfw.init():
                logger.error("Could not initialize GLFW")
                return self._fallback_init()

            # Configure GLFW window hints
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.RESIZABLE, True)
            glfw.window_hint(glfw.VISIBLE, True)  # Explicitly set window visibility
            glfw.window_hint(glfw.FOCUSED, True)  # Window should be focused
            
            # Create window
            self.window = glfw.create_window(self.width, self.height, self.title, None, None)
            if not self.window:
                glfw.terminate()
                logger.error("Failed to create GLFW window")
                return self._fallback_init()

            # Make context current
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)  # Enable vsync

            # Center window on screen
            monitor = glfw.get_primary_monitor()
            if monitor:
                mode = glfw.get_video_mode(monitor)
                if mode:
                    screen_width = mode.size.width
                    screen_height = mode.size.height
                    window_x = (screen_width - self.width) // 2
                    window_y = (screen_height - self.height) // 2
                    glfw.set_window_pos(self.window, window_x, window_y)

            # Initialize ImGui
            imgui.create_context()
            self.ui_state.imgui_manager = self._setup_imgui_impl()
            
            if not self.ui_state.imgui_manager:
                logger.error("Failed to initialize ImGui manager")
                return False

            # Show window after initialization
            glfw.show_window(self.window)
            
            # Initialize screens after window is ready
            self._init_screens()
            
            self.window_created = True
            self.ui_state.is_initialized = True
            
            return True

        except Exception as e:
            logger.error(f"Screen initialization failed: {e}", exc_info=True)
            return self._fallback_init()

    def _check_opengl_glfw(self) -> bool:
        """Check OpenGL and GLFW availability"""
        try:
            import OpenGL.GL
            import glfw
            return True
        except ImportError as e:
            self.logger.error(f"Required package not found: {e}")
            return False

    def render(self, frame_data: Dict[str, Any]) -> None:
        """Render current frame"""
        if not self.window_created or not self.ui_state.is_initialized:
            return

        try:
            import glfw
            import OpenGL.GL as gl

            if glfw.window_should_close(self.window):
                self.should_quit = True
                return

            # Poll events and start new frame
            glfw.poll_events()
            
            # Make sure window is visible
            if not glfw.get_window_attrib(self.window, glfw.VISIBLE):
                glfw.show_window(self.window)

            # Process frame
            self.ui_state.imgui_manager.process_inputs()
            imgui.new_frame()

            # Clear buffer
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            # Render active screen
            current_screen = self.screens.get(self.active_screen)
            if current_screen:
                current_screen.render(frame_data)

            # End frame
            imgui.render()
            self.ui_state.imgui_manager.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        except Exception as e:
            logger.error(f"Render error: {e}")
            self._attempt_recovery()

    def _attempt_recovery(self):
        """Attempt to recover from rendering errors"""
        try:
            if self.ui_state.imgui_manager:
                self.ui_state.imgui_manager.cleanup()
            if self.ui_state.renderer:
                self.ui_state.renderer.cleanup()
                
            # Reinitialize
            self.init()
        except Exception as e:
            logger.error(f"Recovery failed: {e}")

    def cleanup(self) -> None:
        """Clean up resources in correct order"""
        try:
            logger.info("Starting screen cleanup...")
            
            # Clean up ImGui manager first
            if hasattr(self.ui_state, 'imgui_manager') and self.ui_state.imgui_manager:
                try:
                    self.ui_state.imgui_manager.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up ImGui manager: {e}")
                self.ui_state.imgui_manager = None
                
            # Then clean up renderer
            if hasattr(self.ui_state, 'renderer') and self.ui_state.renderer:
                try:
                    self.ui_state.renderer.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up renderer: {e}")
                self.ui_state.renderer = None
                
            self.ui_state.is_initialized = False
            self.window_created = False
            
            logger.info("Screen cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during screen cleanup: {e}")

    def _init_screens(self):
        """Initialize different screen views"""
        from ui.screens.main_screen import MainScreen
        from ui.screens.settings_screen import SettingsScreen
        from ui.screens.login_screen import LoginScreen
        from ui.screens.chat_screen import ChatScreen
        from security.auth.auth_service import AuthService
        from security.config import SECURITY_CONFIG
        
        # Initialize auth service with config
        auth_service = AuthService(security_config=SECURITY_CONFIG)
        
        self.screens = {
            'login': LoginScreen(auth_service),
            'main': MainScreen(),
            'settings': SettingsScreen(),
            'chat': ChatScreen(self.llm) if self.llm else None
        }
        
        # Start with login screen
        self.active_screen = "login"
        
    def set_llm(self, llm):
        """Set LLM component"""
        self.llm = llm
        
    def set_model_manager(self, model_manager):
        """Set model manager component"""
        self.model_manager = model_manager
        
    def set_api_client(self, api_client):
        """Set API client component"""
        self.api_client = api_client

    def process_frame(self, data: dict, timeout_ms: int = 0) -> bool:
        """Process a single frame with improved error handling"""
        try:
            import glfw
            
            if not self.ui_state.is_initialized:
                return False

            # Process GLFW events
            glfw.poll_events()
            if glfw.window_should_close(self.window):
                self.should_quit = True
                return False

            # Start new ImGui frame
            self.ui_state.imgui_manager.process_inputs()
            imgui.new_frame()

            # Render active screen
            try:
                current_screen = self.screens.get(self.active_screen)
                if current_screen:
                    current_screen.render(data)
            except Exception as e:
                logger.error(f"Error rendering screen: {e}")

            # End frame
            imgui.render()
            self.ui_state.imgui_manager.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

            # Frame timing
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_frame_time > 1.0:
                fps = self.frame_count / (current_time - self.last_frame_time)
                logger.debug(f"FPS: {fps:.1f}")
                self.frame_count = 0
                self.last_frame_time = current_time

            return not self.should_quit

        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            self.should_quit = True
            return False

    def shutdown(self):
        """Clean shutdown of screen components"""
        if self.ui_state.renderer:
            self.ui_state.renderer.cleanup()

    def _fallback_init(self) -> bool:
        """Initialize fallback text mode"""
        try:
            from .rendering.text_renderer import TextRenderer
            self.ui_state.renderer = TextRenderer()
            if self.ui_state.renderer.init():
                self.ui_state.is_initialized = True
                self._init_screens()
                return True
            return False
        except Exception as e:
            logger.error(f"Fallback initialization failed: {e}")
            return False

    def _setup_imgui_impl(self):
        """Set up ImGui implementation"""
        try:
            from imgui.integrations.glfw import GlfwRenderer
            if not self.window:
                logger.error("No window available for ImGui setup")
                return None

            impl = GlfwRenderer(self.window)
            
            # Configure ImGui style
            style = imgui.get_style()
            style.window_rounding = 5.0
            style.frame_rounding = 3.0
            style.scrollbar_rounding = 3.0
            
            return impl
        except Exception as e:
            logger.error(f"ImGui setup failed: {e}")
            return None