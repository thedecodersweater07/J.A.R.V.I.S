import imgui
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..api_client import APIClient
from ..themes.theme_manager import ThemeManager
from .base_screen import BaseScreen

logger = logging.getLogger(__name__)

class AIInterfaceScreen(BaseScreen):
    """
    Advanced AI Interface Screen
    Provides a modern UI for interacting with the JARVIS AI system
    """
    
    def __init__(self, api_client: APIClient):
        super().__init__()
        self.api_client = api_client
        self.theme = ThemeManager()
        self.input_text = ""
        self.conversation = []
        self.is_processing = False
        self.current_request_type = "text"
        self.request_types = ["text", "nlp", "ml", "full"]
        self.request_type_names = {
            "text": "Text Response",
            "nlp": "NLP Analysis",
            "ml": "ML Prediction",
            "full": "Full Analysis"
        }
        self.last_response = None
        self.error_message = None
        self.status_message = "Ready"
        self.show_settings = False
        self.imgui_manager = None
        self.width = 800
        self.height = 600
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup UI elements"""
        # Register callbacks with API client
        self.api_client.add_callback("on_response", self._handle_response)
        self.api_client.add_callback("on_error", self._handle_error)
        
    def init(self) -> bool:
        """Initialize screen"""
        self.initialized = True
        return True
        
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle input events"""
        pass  # Input handled through imgui interface
        
    def render(self, frame_data: Dict[str, Any]) -> None:
        """Render the AI interface screen"""
        try:
            # Get ImGui manager from frame data
            self.imgui_manager = frame_data.get("imgui_manager")
            
            # Set window size and position
            imgui.set_next_window_size(self.width, self.height)
            
            # Begin window
            imgui.begin("JARVIS AI Interface", flags=imgui.WINDOW_NO_RESIZE)
            
            # Render main UI components
            self._render_header()
            self._render_conversation()
            self._render_input_area()
            self._render_status_bar()
            
            # Render settings popup if active
            if self.show_settings:
                self._render_settings_popup()
                
            # End window
            imgui.end()
            
        except Exception as e:
            logger.error(f"Error rendering AI interface: {e}")
            
    def _render_header(self):
        """Render header with title and controls"""
        # Header with title
        imgui.begin_group()
        
        # Title
        imgui.push_font(imgui.get_font_index("ImGui-Large"))
        self.theme.text_centered("JARVIS AI Assistant")
        imgui.pop_font()
        
        # Subtitle with status
        status_color = (0.2, 0.8, 0.2, 1.0) if self.api_client.is_authenticated else (0.8, 0.2, 0.2, 1.0)
        imgui.push_style_color(imgui.COLOR_TEXT, *status_color)
        self.theme.text_centered("Status: " + ("Connected" if self.api_client.is_authenticated else "Disconnected"))
        imgui.pop_style_color()
        
        # User info if authenticated
        if self.api_client.is_authenticated and self.api_client.user_info:
            self.theme.text_centered(f"User: {self.api_client.user_info.get('username', 'Unknown')}")
        
        # Settings button
        imgui.same_line(imgui.get_window_width() - 100)
        if imgui.button("Settings", width=80):
            self.show_settings = not self.show_settings
            
        imgui.separator()
        imgui.end_group()
        
    def _render_conversation(self):
        """Render conversation history"""
        # Calculate conversation area height (leave space for input and status)
        conversation_height = self.height - 200
        
        # Begin conversation area with scrolling
        if imgui.begin_child("conversation", width=0, height=conversation_height, border=True):
            # Render each message in the conversation
            for i, message in enumerate(self.conversation):
                self._render_message(message, i)
                
            # Auto-scroll to bottom when new messages arrive
            if len(self.conversation) > 0 and self.conversation[-1].get("is_new", False):
                imgui.set_scroll_here_y(1.0)  # 1.0 = bottom
                self.conversation[-1]["is_new"] = False
                
        imgui.end_child()
        
    def _render_message(self, message: Dict[str, Any], index: int):
        """Render a single message in the conversation"""
        is_user = message.get("is_user", False)
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")
        
        # Message container
        imgui.push_id(str(index))
        
        # Style based on sender
        if is_user:
            # User message (right-aligned)
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.2, 0.4, 0.8, 0.3)
            imgui.push_style_var(imgui.STYLE_CHILD_ROUNDING, 10.0)
            
            # Right-align by calculating width
            text_width = min(imgui.calc_text_size(content).x + 40, imgui.get_window_width() * 0.7)
            imgui.same_line(imgui.get_window_width() - text_width - 20)
            
        else:
            # AI message (left-aligned)
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.3, 0.3, 0.3, 0.3)
            imgui.push_style_var(imgui.STYLE_CHILD_ROUNDING, 10.0)
            imgui.same_line(20)
        
        # Message bubble
        bubble_width = min(imgui.calc_text_size(content).x + 40, imgui.get_window_width() * 0.7)
        if imgui.begin_child(f"message_{index}", width=bubble_width, height=0, border=False):
            # Header with sender and timestamp
            sender = "You" if is_user else "JARVIS"
            imgui.text(f"{sender} - {timestamp}")
            imgui.separator()
            
            # Message content
            imgui.text_wrapped(content)
            
        imgui.end_child()
        
        # Reset styles
        imgui.pop_style_var()
        imgui.pop_style_color()
        
        # Add spacing between messages
        imgui.dummy(0, 10)
        
        imgui.pop_id()
        
    def _render_input_area(self):
        """Render input area with text input and send button"""
        imgui.begin_group()
        
        # Request type selector
        imgui.push_item_width(150)
        current_type = self.request_types.index(self.current_request_type)
        changed, new_type = imgui.combo(
            "Request Type", 
            current_type,
            [self.request_type_names[t] for t in self.request_types]
        )
        if changed:
            self.current_request_type = self.request_types[new_type]
        imgui.pop_item_width()
        
        # Text input
        imgui.push_item_width(imgui.get_window_width() - 180)
        changed, self.input_text = imgui.input_text(
            "##input", 
            self.input_text, 
            2048,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        )
        imgui.pop_item_width()
        
        # Send button
        imgui.same_line()
        send_clicked = imgui.button("Send", width=100)
        
        # Process input on Enter or Send button click
        if (changed or send_clicked) and self.input_text.strip() and not self.is_processing:
            self._send_message()
            
        imgui.end_group()
        
    def _render_status_bar(self):
        """Render status bar at bottom of screen"""
        imgui.separator()
        
        # Status message
        status_text = "Processing..." if self.is_processing else self.status_message
        
        # Error message takes precedence
        if self.error_message:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.2, 0.2, 1.0)
            imgui.text(f"Error: {self.error_message}")
            imgui.pop_style_color()
        else:
            imgui.text(status_text)
            
        # Show processing spinner
        if self.is_processing:
            imgui.same_line()
            self._render_spinner()
            
    def _render_spinner(self):
        """Render a simple spinner animation"""
        t = int(time.time() * 4) % 4
        spinner = "|/-\\"[t]
        imgui.text(spinner)
        
    def _render_settings_popup(self):
        """Render settings popup"""
        # Center popup
        if self.imgui_manager:
            self.imgui_manager.set_next_window_centered()
            
        if imgui.begin_popup_modal("Settings", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Server Settings")
            imgui.separator()
            
            # Server URL
            imgui.text("Server URL:")
            imgui.same_line()
            imgui.text(self.api_client.base_url)
            
            # Authentication status
            auth_status = "Authenticated" if self.api_client.is_authenticated else "Not authenticated"
            imgui.text(f"Status: {auth_status}")
            
            # Server health
            if imgui.button("Check Server Health"):
                health = self.api_client.check_health()
                self.status_message = f"Server health: {health.get('status', 'unknown')}"
                
            imgui.separator()
            
            # Close button
            if imgui.button("Close"):
                self.show_settings = False
                imgui.close_current_popup()
                
            imgui.end_popup()
        elif self.show_settings:
            imgui.open_popup("Settings")
            
    def _send_message(self):
        """Send message to AI"""
        if not self.input_text.strip() or not self.api_client.is_authenticated:
            return
            
        # Add user message to conversation
        user_message = {
            "is_user": True,
            "content": self.input_text,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_new": True
        }
        self.conversation.append(user_message)
        
        # Store message text and clear input
        query = self.input_text
        self.input_text = ""
        
        # Set processing state
        self.is_processing = True
        self.error_message = None
        self.status_message = f"Processing {self.current_request_type} request..."
        
        # Send async request
        self.api_client.async_query_ai(
            query=query,
            request_type=self.current_request_type
        )
        
    def _handle_response(self, response: Dict[str, Any]):
        """Handle AI response"""
        self.is_processing = False
        self.last_response = response
        
        # Extract response content
        content = response.get("response", "")
        if isinstance(content, dict):
            # Format structured response
            if "combined" in content:
                # Use combined response from full analysis
                content = content["combined"]
            else:
                # Format dictionary as text
                content = self._format_dict_response(content)
        
        # Add AI message to conversation
        ai_message = {
            "is_user": False,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_new": True,
            "metadata": {
                "request_id": response.get("request_id"),
                "processing_time": response.get("processing_time", 0)
            }
        }
        self.conversation.append(ai_message)
        
        # Update status
        processing_time = response.get("processing_time", 0)
        self.status_message = f"Response received in {processing_time:.2f}s"
        
    def _handle_error(self, error_message: str):
        """Handle error from API client"""
        self.is_processing = False
        self.error_message = error_message
        self.status_message = "Error occurred"
        
    def _format_dict_response(self, data: Dict[str, Any]) -> str:
        """Format dictionary response as text"""
        lines = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively format nested dictionaries
                nested_value = self._format_dict_response(value)
                lines.append(f"{key}:\n{nested_value}")
            else:
                lines.append(f"{key}: {value}")
                
        return "\n".join(lines)
        
    def cleanup(self) -> None:
        """Cleanup resources"""
        # Remove callbacks
        self.api_client.remove_callback("on_response", self._handle_response)
        self.api_client.remove_callback("on_error", self._handle_error)
        self.initialized = False
