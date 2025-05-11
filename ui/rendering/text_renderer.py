import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from .renderer_base import RendererBase

logger = logging.getLogger(__name__)

class InputEvent:
    """Simple class to represent input events"""
    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)

class TextRenderer(RendererBase):
    """Text-based renderer for fallback mode"""
    
    def __init__(self, width: int = 80, height: int = 24, title: str = "JARVIS"):
        self.width = width
        self.height = height
        self.title = title
        self.is_initialized = False
        self.input_events = []
        self.should_close = False
        self.command_listener = None
        self.frame_buffer = []
        self.last_frame_time = time.time()
        self.frame_rate = 30  # Target frame rate
        self.frame_time = 1.0 / self.frame_rate
        
    def init(self) -> bool:
        """Initialize the text renderer"""
        try:
            # Print header
            print(f"\n{'=' * self.width}")
            print(f"{self.title.center(self.width)}")
            print(f"{'=' * self.width}\n")
            
            # Initialize command listener on separate thread
            self._init_command_listener()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Text renderer initialization error: {e}")
            return False
            
    def _init_command_listener(self):
        """Initialize command listener on separate thread"""
        try:
            import threading
            
            def listener():
                while not self.should_close:
                    try:
                        cmd = input("> ")
                        if cmd.lower() in ["quit", "exit", "q"]:
                            self.input_events.append(InputEvent("quit"))
                            self.should_close = True
                        else:
                            self.input_events.append(InputEvent("command", text=cmd))
                    except EOFError:
                        self.input_events.append(InputEvent("quit"))
                        self.should_close = True
                        break
                    except KeyboardInterrupt:
                        self.input_events.append(InputEvent("quit"))
                        self.should_close = True
                        break
                    except Exception as e:
                        logger.error(f"Error in command listener: {e}")
            
            self.command_listener = threading.Thread(target=listener, daemon=True)
            self.command_listener.start()
            
        except Exception as e:
            logger.error(f"Command listener initialization error: {e}")
            
    def process_inputs(self) -> None:
        """Process accumulated input events"""
        # Input processing is done in the listener thread
        pass
            
    def get_input_events(self) -> List[InputEvent]:
        """Get and clear accumulated input events"""
        events = self.input_events.copy()
        self.input_events.clear()
        return events
            
    def render(self, frame_data: Dict[str, Any] = None) -> None:
        """Render text frame"""
        try:
            # Limit frame rate
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            
            if elapsed < self.frame_time:
                # Skip rendering if too soon
                return
                
            self.last_frame_time = current_time
            
            # For text mode, we just render a status message
            if len(self.frame_buffer) > 0:
                for line in self.frame_buffer:
                    print(line)
                self.frame_buffer.clear()
                
            # If frame_data contains any text messages, print them
            if frame_data and 'text_messages' in frame_data:
                for message in frame_data['text_messages']:
                    print(message)
                
        except Exception as e:
            logger.error(f"Text render error: {e}")
            
    def add_to_buffer(self, text: str):
        """Add text to the frame buffer"""
        self.frame_buffer.append(text)
            
    def window_should_close(self) -> bool:
        """Check if window should close"""
        return self.should_close
            
    def process_events(self) -> None:
        """Process window events"""
        # Nothing to do for text mode
        pass
            
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.should_close = True
            print("\nExiting...\n")
            self.is_initialized = False
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def begin_frame(self):
        """Setup for frame rendering"""
        # Nothing to do for text mode
        pass
        
    def end_frame(self):
        """Cleanup after frame rendering"""
        # Render any pending text in the buffer
        self.render()