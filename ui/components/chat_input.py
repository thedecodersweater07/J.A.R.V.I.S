import imgui
from typing import Optional

class ChatInput:
    def __init__(self):
        self.input_text = ""
        self.history: List[str] = []
        self.history_index = -1
        
    def render(self) -> bool:
        """Render chat input with history support"""
        imgui.push_item_width(imgui.get_window_width() - 120)
        
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
