import logging
import time
import glfw

from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

# Try to import imgui, but don't fail if not available
try:
    import imgui
    IMGUI_AVAILABLE = True
except ImportError:
    IMGUI_AVAILABLE = False

logger = logging.getLogger(__name__)

class ImGuiManager:
    """
    Manages ImGui context and frame state to prevent frame errors
    Directly integrates with GLFW to ensure proper frame synchronization
    """
    
    def __init__(self):
        if not IMGUI_AVAILABLE:
            raise ImportError("ImGui is not available. Please install with: pip install imgui")
        self.impl = None
        self.context = None
        self.window = None
        self.is_initialized = False
        self.frame_active = False
        self.last_frame_time = 0
        self.frame_count = 0
        self.window_stack = []
        self.tab_bar_stack = []
        
    def init(self, window_handle=None) -> bool:
        """Initialize ImGui context with direct GLFW integration"""
        try:
            from imgui.integrations.glfw import GlfwRenderer
            
            # Store window handle
            self.window = window_handle
            
            # Create fresh ImGui context
            self.context = imgui.create_context()
            imgui.set_current_context(self.context)
            
            if self.window:
                self.impl = GlfwRenderer(self.window)
                self._setup_style()
                
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"ImGui initialization error: {e}")
            return False
            
    def _setup_style(self):
        """Configure ImGui style"""
        style = imgui.get_style()
        style.window_rounding = 5.0
        style.frame_rounding = 3.0
        style.scrollbar_rounding = 5.0
        style.grab_rounding = 3.0
        style.frame_border_size = 1.0
        
        # Set dark theme with better contrast
        imgui.style_colors_dark()
        colors = style.colors
        colors[imgui.COLOR_TEXT] = (0.95, 0.95, 0.95, 1.00)
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.10, 0.10, 0.12, 1.00)
        colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.15, 0.15, 0.17, 1.00)
    
    def set_window(self, window_handle) -> bool:
        """Set or update the GLFW window handle"""
        try:
            from imgui.integrations.glfw import GlfwRenderer
            
            self.window = window_handle
            
            # Reinitialize the GLFW implementation
            if self.impl:
                self.impl.shutdown()
                
            self.impl = GlfwRenderer(window_handle)
            return True
        except Exception as e:
            logger.error(f"Error setting window handle: {e}")
            return False
    
    def process_inputs(self):
        """Process GLFW inputs for ImGui"""
        if not self.is_initialized or not self.impl:
            return
            
        try:
            self.impl.process_inputs()
        except Exception as e:
            logger.error(f"Error processing inputs: {e}")
    
    def begin_frame(self) -> bool:
        """Begin a new ImGui frame with direct GLFW integration"""
        if not self.is_initialized or not self.impl:
            return False
            
        try:
            # Check for valid context
            if not imgui.get_current_context():
                logger.error("No valid ImGui context in begin_frame")
                return False
                
            # If a frame is already active, do nothing
            if self.frame_active:
                return True
                
            # Rate limiting (max 100 fps)
            current_time = time.time()
            if current_time - self.last_frame_time < 0.01:
                time.sleep(0.01 - (current_time - self.last_frame_time))
            
            # Process inputs and start a new frame
            self.impl.process_inputs()
            imgui.new_frame()
            
            # Update state
            self.frame_active = True
            self.frame_count += 1
            self.last_frame_time = current_time
            self.window_stack = []
            self.tab_bar_stack = []
            return True
            
        except Exception as e:
            logger.error(f"Error in begin_frame: {e}")
            return False
    
    def end_frame(self) -> bool:
        """End the current ImGui frame with direct GLFW integration"""
        if not self.is_initialized or not self.impl:
            return False
            
        try:
            # Close any open windows or tab bars
            self._close_open_elements()
            
            # Only end the frame if one is active
            if self.frame_active:
                imgui.render()
                self.impl.render(imgui.get_draw_data())
                self.frame_active = False
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error in end_frame: {e}")
            self.frame_active = False
            return False
    
    def _close_open_elements(self):
        """Close any open ImGui elements (windows, tab bars, etc.)"""
        import imgui
        
        # Close tab bars first (inner elements)
        while self.tab_bar_stack:
            try:
                imgui.end_tab_bar()
            except Exception as e:
                logger.error(f"Error closing tab bar: {e}")
            self.tab_bar_stack.pop()
            
        # Then close windows (outer elements)
        while self.window_stack:
            try:
                imgui.end()
            except Exception as e:
                logger.error(f"Error closing window: {e}")
            self.window_stack.pop()
    
    def begin_window(self, name: str, **kwargs) -> bool:
        """Begin a new ImGui window with tracking"""
        try:
            result = imgui.begin(name, **kwargs)
            self.window_stack.append(name)
            return result
        except Exception as e:
            logger.error(f"Error beginning window '{name}': {e}")
            return False
    
    def end_window(self) -> None:
        """End the current ImGui window with tracking"""
        try:
            imgui.end()
            if self.window_stack:
                self.window_stack.pop()
        except Exception as e:
            logger.error(f"Error ending window: {e}")
    
    def begin_tab_bar(self, name: str, **kwargs) -> bool:
        """Begin a new ImGui tab bar with tracking"""
        try:
            result = imgui.begin_tab_bar(name, **kwargs)
            if result:
                self.tab_bar_stack.append(name)
            return result
        except Exception as e:
            logger.error(f"Error beginning tab bar '{name}': {e}")
            return False
    
    def end_tab_bar(self) -> None:
        """End the current ImGui tab bar with tracking"""
        try:
            imgui.end_tab_bar()
            if self.tab_bar_stack:
                self.tab_bar_stack.pop()
        except Exception as e:
            logger.error(f"Error ending tab bar: {e}")
    
    def render_frame(self, render_function, frame_data: Dict[str, Any] = None) -> bool:
        """Render a complete frame with proper GLFW integration"""
        if not self.is_initialized or not self.impl:
            return False
            
        if frame_data is None:
            frame_data = {}
            
        # Add self to frame data so components can access ImGui utilities
        frame_data["imgui_manager"] = self
        
        try:
            # Begin frame
            if not self.begin_frame():
                return False
                
            # Call the render function
            render_function(frame_data)
            
            # End frame and render
            self.end_frame()
            return True
            
        except Exception as e:
            logger.error(f"Error in render_frame: {e}")
            # Ensure frame is properly closed on error
            if self.frame_active:
                try:
                    self._close_open_elements()
                    imgui.end_frame()
                    self.frame_active = False
                except Exception:
                    pass
            return False
    
    @contextmanager
    def frame(self):
        """Safe context manager for ImGui frames with direct GLFW integration"""
        frame_started = False
        try:
            # Start a new frame
            frame_started = self.begin_frame()
            
            # Yield control to the caller
            yield
            
        finally:
            # Always end the frame if we started one
            if frame_started:
                self._close_open_elements()
                self.end_frame()
                
    @contextmanager
    def window(self, name: str, **kwargs):
        """Safe context manager for ImGui windows"""
        try:
            self.begin_window(name, **kwargs)
            yield
        finally:
            self.end_window()
    
    @contextmanager
    def tab_bar(self, name: str, **kwargs):
        """Safe context manager for ImGui tab bars"""
        try:
            if self.begin_tab_bar(name, **kwargs):
                yield
                self.end_tab_bar()
            else:
                yield
        except Exception as e:
            logger.error(f"Error in tab_bar context manager: {e}")
            yield
                
    def set_next_window_centered(self):
        """Center the next window on screen - replacement for missing ImGui function"""
        try:
            import glfw
            
            # Get window size
            if self.window:
                window_width, window_height = glfw.get_window_size(self.window)
                # Center the window
                imgui.set_next_window_position(
                    window_width // 2, 
                    window_height // 2,
                    imgui.COND_ALWAYS,
                    pivot_x=0.5, 
                    pivot_y=0.5
                )
                return True
            return False
        except Exception as e:
            logger.error(f"ImGui frame error: {e}")
            return False
            
    def cleanup(self):
        """Clean up ImGui resources"""
        try:
            # End any active frame
            if self.frame_active:
                try:
                    self._close_open_elements()
                    imgui.end_frame()
                    self.frame_active = False
                except Exception as e:
                    logger.error(f"Error ending frame during cleanup: {e}")
            
            # Shutdown the GLFW implementation met veilige OpenGL cleanup
            if self.impl:
                try:
                    # Controleer of de OpenGL functies beschikbaar zijn voordat we ze aanroepen
                    from OpenGL import GL
                    if hasattr(GL, 'glDeleteVertexArrays') and bool(GL.glDeleteVertexArrays):
                        self.impl.shutdown()
                    else:
                        # Alternatieve cleanup zonder glDeleteVertexArrays
                        logger.warning("OpenGL glDeleteVertexArrays not available, using alternative cleanup")
                        if hasattr(self.impl, '_imgui_io'):
                            self.impl._imgui_io = None
                        if hasattr(self.impl, '_font_texture'):
                            self.impl._font_texture = None
                except ImportError:
                    logger.warning("OpenGL module not available, skipping GLFW implementation shutdown")
                except Exception as e:
                    logger.error(f"Error shutting down GLFW implementation: {e}")
                self.impl = None
                
            # Destroy the ImGui context
            if self.context and imgui.get_current_context() == self.context:
                try:
                    imgui.destroy_context(self.context)
                except Exception as e:
                    logger.error(f"Error destroying ImGui context: {e}")
                
            # Reset state
            self.context = None
            self.window = None
            self.is_initialized = False
            self.frame_active = False
            
            logger.info("ImGui cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up ImGui: {e}")