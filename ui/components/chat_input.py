import imgui
import glfw
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChatInput:
    def __init__(self, placeholder: str = "Type your message..."):
        self.input_text = ""
        self.history: List[str] = []
        self.history_index = -1
        self.placeholder = placeholder
        self.is_focused = False
        self.last_key_up = False
        self.last_key_down = False
        
    def init(self) -> bool:
        """Initialize the chat input component"""
        try:
            # Laad eventuele opgeslagen chatgeschiedenis
            self._load_history()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ChatInput: {e}")
            return False
        
    def render(self) -> bool:
        """Render chat input with history support and improved focus handling."""
        # Bereken de juiste breedte voor het inputveld
        input_width = imgui.get_window_width() - 120
        if input_width < 100:
            input_width = imgui.get_window_width() * 0.7  # Fallback voor smalle vensters
            
        imgui.push_item_width(input_width)

        # Maak een duidelijk zichtbaar inputveld
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (8, 8))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.05, 0.05, 0.05)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, 0.1, 0.1, 0.1)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 0.15, 0.15, 0.15)

        # Controleer of we focus moeten instellen
        if not self.is_focused:
            imgui.set_keyboard_focus_here()
            self.is_focused = True

        # Verwerk invoer met verbeterde flags
        flags = imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | imgui.INPUT_TEXT_CALLBACK_ALWAYS
        changed, self.input_text = imgui.input_text(
            "##chat_input",
            self.input_text,
            2048,  # Verhoogde buffer grootte
            flags
        )

        # Herstel stijl
        imgui.pop_style_color(3)
        imgui.pop_style_var()

        # Verwerk geschiedenis navigatie met pijltjestoetsen
        self._handle_history_navigation()

        # Toon placeholder als er geen tekst is
        cursor_pos = imgui.get_cursor_pos()
        if not self.input_text and not imgui.is_item_active():
            imgui.set_cursor_pos((cursor_pos[0] - input_width + 10, cursor_pos[1] - 30))
            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5)
            imgui.text(self.placeholder)
            imgui.pop_style_color()
            imgui.set_cursor_pos(cursor_pos)

        # Verzendknop met verbeterde stijl
        imgui.same_line()
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.4, 0.8)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.5, 0.9)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.1, 0.3, 0.7)
        send_clicked = imgui.button("Send")
        imgui.pop_style_color(3)

        # Verwerk verzending
        if changed or send_clicked:
            if self.input_text.strip():
                self.history.append(self.input_text)
                result = self.input_text
                self.input_text = ""
                self.history_index = -1
                self._save_history()  # Sla geschiedenis op
                return True

        return False
    
    def _handle_history_navigation(self):
        """Handle up/down arrow keys for history navigation"""
        # Controleer of pijltje omhoog is ingedrukt
        key_up = imgui.is_key_pressed(glfw.KEY_UP)
        if key_up and not self.last_key_up and len(self.history) > 0:
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.input_text = self.history[-(self.history_index+1)]
        self.last_key_up = key_up
        
        # Controleer of pijltje omlaag is ingedrukt
        key_down = imgui.is_key_pressed(glfw.KEY_DOWN)
        if key_down and not self.last_key_down:
            if self.history_index > 0:
                self.history_index -= 1
                self.input_text = self.history[-(self.history_index+1)]
            elif self.history_index == 0:
                self.history_index = -1
                self.input_text = ""
        self.last_key_down = key_down
        
    def _save_history(self):
        """Save chat history to memory"""
        # Beperk geschiedenis tot laatste 50 berichten
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def _load_history(self):
        """Load chat history from memory"""
        # Implementatie voor het laden van geschiedenis kan hier worden toegevoegd
        pass
        
    def get_text(self) -> str:
        """Get the current input text, trimmed of whitespace"""
        return self.input_text.strip()
    
    def cleanup(self):
        """Clean up resources"""
        self._save_history()
