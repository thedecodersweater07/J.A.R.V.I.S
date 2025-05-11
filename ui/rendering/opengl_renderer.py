import os
import sys
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .renderer_base import RendererBase

logger = logging.getLogger(__name__)

class InputEvent:
    """Simple class to represent input events"""
    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)

class OpenGLRenderer(RendererBase):
    """OpenGL-based renderer using GLFW and ImGui"""
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS"):
        self.width = width
        self.height = height
        self.title = title
        self.window = None
        self.impl = None
        self.is_initialized = False
        self.input_events = []
        self._init_failed = False
        
    def init(self) -> bool:
        """Initialize the OpenGL renderer with proper error handling"""
        try:
            # Import required libraries
            import glfw
            import imgui
            from imgui.integrations.glfw import GlfwRenderer
            import OpenGL.GL as gl
            from .imGUI_manager import ImGuiManager
            
            # Initialize GLFW
            if not glfw.init():
                logger.error("Could not initialize GLFW")
                self._init_failed = True
                return False
                
            # Configure GLFW
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            if sys.platform == 'darwin':  # Special handling for MacOS
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
            
            # Create window
            self.window = glfw.create_window(
                self.width, self.height, self.title, None, None
            )
            
            if not self.window:
                logger.error("Could not create GLFW window")
                glfw.terminate()
                self._init_failed = True
                return False
                
            # Make context current
            glfw.make_context_current(self.window)
            
            # Enable vsync
            glfw.swap_interval(1)
            
            # We don't need the old ImGui integration anymore
            # self.impl = GlfwRenderer(self.window)
            
            # Set callbacks
            glfw.set_key_callback(self.window, self._key_callback)
            glfw.set_window_size_callback(self.window, self._resize_callback)
            
            self.is_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Required OpenGL libraries not available: {e}")
            self._init_failed = True
            return False
        except Exception as e:
            logger.error(f"OpenGL initialization error: {e}")
            self._init_failed = True
            return False
            
    def _key_callback(self, window, key, scancode, action, mods):
        """GLFW key callback"""
        try:
            import glfw
            
            # Standard key handling
            if action == glfw.PRESS:
                if key == glfw.KEY_ESCAPE:
                    self.input_events.append(InputEvent("keydown", key="escape"))
                elif key == glfw.KEY_Q and (mods & glfw.MOD_CONTROL):
                    self.input_events.append(InputEvent("quit"))
        except Exception as e:
            logger.error(f"Error in key callback: {e}")
                
    def _resize_callback(self, window, width, height):
        """GLFW window resize callback"""
        try:
            import OpenGL.GL as gl
            
            self.width = width
            self.height = height
            gl.glViewport(0, 0, width, height)
        except Exception as e:
            logger.error(f"Error in resize callback: {e}")
            
    def process_inputs(self) -> None:
        """Process accumulated input events"""
        try:
            import glfw
            
            if not self.is_initialized or not self.window:
                return
                
            # Poll for events
            glfw.poll_events()
            
            # Check for window close flag
            if glfw.window_should_close(self.window):
                self.input_events.append(InputEvent("quit"))
                
            # Update ImGui inputs
            if self.impl:
                self.impl.process_inputs()
                
        except Exception as e:
            logger.error(f"Error processing inputs: {e}")
            
    def get_input_events(self) -> List[InputEvent]:
        """Get and clear accumulated input events"""
        events = self.input_events.copy()
        self.input_events.clear()
        return events
            
    def render(self, frame_data: Dict[str, Any] = None) -> None:
        """Render ImGui frame"""
        try:
            import OpenGL.GL as gl
            import glfw
            
            if not self.is_initialized or not self.window:
                return
                
            # Clear the framebuffer
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Render ImGui
            if self.impl and frame_data:
                self.impl.render(frame_data.get('imgui_draw_data'))
                
            # Swap buffers
            glfw.swap_buffers(self.window)
            
        except Exception as e:
            logger.error(f"Render error: {e}")
            
    def window_should_close(self) -> bool:
        """Check if window should close"""
        try:
            import glfw
            return glfw.window_should_close(self.window) if self.window else True
        except Exception:
            return True
            
    def process_events(self) -> None:
        """Process window events"""
        try:
            import glfw
            if self.is_initialized and self.window:
                glfw.poll_events()
        except Exception as e:
            logger.error(f"Error processing events: {e}")
            
    def begin_frame(self):
        """Setup for frame rendering"""
        try:
            import imgui
            if self.is_initialized and self.impl:
                self.impl.process_inputs()
                imgui.new_frame()
        except Exception as e:
            logger.error(f"Error beginning frame: {e}")
            
    def end_frame(self):
        """Cleanup after frame rendering"""
        try:
            import imgui
            if self.is_initialized:
                imgui.render()
                self.render({'imgui_draw_data': imgui.get_draw_data()})
        except Exception as e:
            logger.error(f"Error ending frame: {e}")
            
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            import glfw
            
            # If init failed, we might not need full cleanup
            if self._init_failed:
                logger.info("Skipping full cleanup due to initialization failure")
                return
                
            # Clean up ImGui implementation
            if self.impl is not None:
                try:
                    self.impl.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down ImGui implementation: {e}")
                self.impl = None
                
            # Clean up GLFW resources
            if self.window is not None:
                try:
                    glfw.destroy_window(self.window)
                except Exception as e:
                    logger.error(f"Error destroying window: {e}")
                self.window = None
                
            # Terminate GLFW
            try:
                glfw.terminate()
            except Exception as e:
                logger.error(f"Error terminating GLFW: {e}")
                
            self.is_initialized = False
            logger.info("OpenGL renderer cleaned up successfully")
            
        except ImportError:
            # If GLFW is not available, just log and continue
            logger.info("GLFW not available during cleanup")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")