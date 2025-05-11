import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ImGuiManager:
    """
    Manages ImGui context and frame state to prevent frame errors
    """
    
    def __init__(self):
        self.context = None
        self.is_initialized = False
        self.current_frame_started = False
        self.last_frame_time = 0
        self.frame_count = 0
        
    def init(self) -> bool:
        """Initialize ImGui context"""
        try:
            import imgui
            
            # Create context if not already created
            if not imgui.get_current_context():
                self.context = imgui.create_context()
                imgui.set_current_context(self.context)
                
                # Set default style
                style = imgui.get_style()
                style.window_rounding = 5.0
                style.frame_rounding = 3.0
                style.scrollbar_rounding = 5.0
                
                # Dark theme
                imgui.style_colors_dark()
                
                self.is_initialized = True
                logger.info("ImGui initialized successfully")
                return True
            else:
                self.context = imgui.get_current_context()
                self.is_initialized = True
                logger.info("ImGui context already exists, reusing")
                return True
                
        except ImportError:
            logger.error("ImGui not available")
            return False
        except Exception as e:
            logger.error(f"ImGui initialization error: {e}")
            return False
            
    @contextmanager
    def frame(self):
        """Safe context manager for ImGui frames"""
        if not self.is_initialized:
            yield
            return
            
        frame_started = False
        try:
            import imgui
            
            # Rate limiting to avoid frame issues
            current_time = time.time()
            if current_time - self.last_frame_time < 0.01:  # Max 100 fps
                yield
                return
                
            # Start frame if not already in a frame
            ctx = imgui.get_current_context()
            if ctx and not self.current_frame_started:
                imgui.new_frame()
                frame_started = True
                self.current_frame_started = True
                self.frame_count += 1
                
            yield
            
            # End frame if we started one
            if frame_started:
                try:
                    # Make sure to call end_frame before render
                    imgui.end_frame()
                    imgui.render()
                    self.current_frame_started = False
                    self.last_frame_time = current_time
                except Exception as e:
                    logger.error(f"Error ending ImGui frame: {e}")
                    # Attempt recovery
                    try:
                        if self.current_frame_started:
                            imgui.end_frame()
                            self.current_frame_started = False
                    except Exception as e2:
                        logger.error(f"Failed to recover from frame error: {e2}")
                        self.current_frame_started = False
                        
        except Exception as e:
            logger.error(f"ImGui frame error: {e}")
            if not frame_started:
                yield
            
            # Try to recover from bad frame state
            try:
                if self.current_frame_started:
                    imgui.end_frame()
                    self.current_frame_started = False
            except Exception:
                self.current_frame_started = False
                
    def cleanup(self):
        """Clean up ImGui resources"""
        try:
            import imgui
            
            # End any active frame
            if self.current_frame_started:
                try:
                    imgui.end_frame()
                except Exception as e:
                    logger.error(f"Error ending frame during cleanup: {e}")
                self.current_frame_started = False
                
            # Destroy context if it's ours
            if self.context and imgui.get_current_context() == self.context:
                try:
                    imgui.destroy_context(self.context)
                except Exception as e:
                    logger.error(f"Error destroying ImGui context: {e}")
                
            self.context = None
            self.is_initialized = False
            logger.info("ImGui cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up ImGui: {e}")