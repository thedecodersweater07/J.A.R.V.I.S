import imgui
from typing import Optional, List

class ChatInput:
    def __init__(self, placeholder: str = "Type your message..."):
        self.input_text = ""
        self.history: List[str] = []
        self.history_index = -1
        self.placeholder = placeholder
        
    def render(self) -> bool:
        """Render chat input with history support"""
        imgui.push_item_width(imgui.get_window_width() - 120)
        
        # Show placeholder text if input is empty
        if not self.input_text and not imgui.is_item_active():
            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5)
            imgui.text(self.placeholder)
            imgui.pop_style_color()
        
        changed, self.input_text = imgui.input_text(
            "##chat_input", 
            self.input_text,
            1024,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        )
        
        imgui.same_line()
        send_clicked = imgui.button("Send")
        
        if changed or send_clicked:
            if self.input_text.strip():
                self.history.append(self.input_text)
                result = self.input_text
                self.input_text = ""
                self.history_index = -1
                return True
                
        return False
        
    def get_text(self) -> str:
        return self.input_text.strip()
