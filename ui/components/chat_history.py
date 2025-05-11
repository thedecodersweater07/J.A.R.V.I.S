import imgui
import time
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class ChatHistory:
    def __init__(self, max_messages: int = 1000):
        self.messages: List[Dict] = []
        self.max_messages = max_messages
        self.should_scroll = False
        self.last_message_time = 0
        self.theme = {
            "user_bg": (0.1, 0.2, 0.1, 0.7),
            "user_text": (0.8, 1.0, 0.8, 1.0),
            "assistant_bg": (0.1, 0.1, 0.2, 0.7),
            "assistant_text": (0.8, 0.8, 1.0, 1.0),
            "timestamp": (0.5, 0.5, 0.5, 0.8),
            "separator": (0.3, 0.3, 0.3, 1.0)
        }
        
    def init(self) -> bool:
        """Initialize the chat history component"""
        try:
            self._load_history()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ChatHistory: {e}")
            return False
        
    def render(self):
        """Render chat messages with improved styling and auto-scroll"""
        # Begin child window with a border
        imgui.push_style_var(imgui.STYLE_CHILD_ROUNDING, 5.0)
        imgui.push_style_var(imgui.STYLE_FRAME_BORDER_SIZE, 1.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.3, 0.3, 0.3)
        
        imgui.begin_child("ChatHistory", 0, -60, border=True)
        
        # Render messages
        if self.messages:
            for i, msg in enumerate(self.messages):
                self._render_message(msg, i)
                
            # Auto-scroll naar beneden als er een nieuw bericht is
            if self.should_scroll:
                imgui.set_scroll_here_y(1.0)  # 1.0 = onderaan
                self.should_scroll = False
        else:
            # Toon een welkomstbericht als er geen berichten zijn
            imgui.push_font(imgui.get_io().fonts.fonts[0])
            text_width = imgui.calc_text_size("Welkom bij JARVIS. Hoe kan ik je helpen?").x
            window_width = imgui.get_window_width()
            imgui.set_cursor_pos_x((window_width - text_width) * 0.5)
            imgui.set_cursor_pos_y(imgui.get_window_height() * 0.4)
            imgui.text_colored("Welkom bij JARVIS. Hoe kan ik je helpen?", 0.7, 0.7, 0.7)
            imgui.pop_font()
            
        imgui.end_child()
        imgui.pop_style_color()
        imgui.pop_style_var(2)
        
    def _render_message(self, msg: Dict, index: int):
        """Render een individueel bericht met verbeterde stijl"""
        is_user = msg.get("is_user", False)
        text = msg.get("text", "")
        timestamp = msg.get("timestamp", "")
        
        # Bereid kleuren voor
        if is_user:
            bg_color = self.theme["user_bg"]
            text_color = self.theme["user_text"]
            sender = "You"
        else:
            bg_color = self.theme["assistant_bg"]
            text_color = self.theme["assistant_text"]
            sender = "JARVIS"
            
        # Voeg wat ruimte toe tussen berichten
        if index > 0:
            imgui.dummy(0, 5)
            
        # Bereken padding en breedte
        window_width = imgui.get_window_width()
        text_width = min(window_width * 0.85, 600)  # Max breedte van het bericht
        padding = 10
        
        # Begin een gekleurde groep voor het bericht
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *bg_color)
        imgui.push_style_var(imgui.STYLE_CHILD_ROUNDING, 8.0)
        
        # Maak unieke ID voor dit bericht
        child_id = f"msg_{index}"
        
        # Bereken de juiste positie (links voor assistant, rechts voor user)
        if is_user:
            imgui.set_cursor_pos_x(window_width - text_width - padding)
        else:
            imgui.set_cursor_pos_x(padding)
            
        # Begin het berichtvenster
        imgui.begin_child(child_id, text_width, 0, border=False)
        
        # Toon afzender en tijdstempel
        imgui.push_style_color(imgui.COLOR_TEXT, *text_color)
        imgui.text(f"{sender}")
        if timestamp:
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_TEXT, *self.theme["timestamp"])
            imgui.text(f"({timestamp})")
            imgui.pop_style_color()
        
        # Toon berichtinhoud met word wrapping
        imgui.push_text_wrap_pos(text_width - padding * 2)
        imgui.text_wrapped(text)
        imgui.pop_text_wrap_pos()
        imgui.pop_style_color()  # Text color
        
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color()  # Background color
        
    def add_user_message(self, text: str):
        """Voeg een gebruikersbericht toe aan de geschiedenis"""
        self.messages.append({
            "text": text, 
            "is_user": True,
            "timestamp": self._get_timestamp()
        })
        self._trim_history()
        self.should_scroll = True
        self.last_message_time = time.time()
        self._save_history()
        
    def add_assistant_message(self, text: str):
        """Voeg een assistentbericht toe aan de geschiedenis"""
        self.messages.append({
            "text": text, 
            "is_user": False,
            "timestamp": self._get_timestamp()
        })
        self._trim_history()
        self.should_scroll = True
        self.last_message_time = time.time()
        self._save_history()
    
    def _get_timestamp(self) -> str:
        """Genereer een tijdstempel voor berichten"""
        return time.strftime("%H:%M:%S")
        
    def _trim_history(self):
        """Beperk de geschiedenis tot het maximum aantal berichten"""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
    def _save_history(self):
        """Sla de chatgeschiedenis op"""
        # Implementatie voor het opslaan van geschiedenis kan hier worden toegevoegd
        pass
        
    def _load_history(self):
        """Laad de chatgeschiedenis"""
        # Implementatie voor het laden van geschiedenis kan hier worden toegevoegd
        pass
        
    def clear(self):
        """Wis alle berichten uit de geschiedenis"""
        self.messages = []
        self._save_history()
        
    def cleanup(self):
        """Ruim resources op"""
        self._save_history()
