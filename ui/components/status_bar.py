import imgui
from typing import Dict, Any, Optional

class StatusBar:
    def __init__(self):
        self.error: Optional[str] = None
        
    def render(self, status: Dict[str, Any]):
        imgui.set_cursor_pos_y(imgui.get_window_height() - 30)
        if self.error:
            imgui.text_colored(self.error, 1, 0, 0)
        else:
            imgui.text(f"Status: {status.get('status', 'Ready')}")
            
    def show_error(self, error: str):
        self.error = error
