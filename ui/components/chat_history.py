import imgui
from typing import List, Dict

class ChatHistory:
    def __init__(self):
        self.messages: List[Dict] = []
        
    def render(self):
        imgui.begin_child("ChatHistory", 0, -60, border=True)
        for msg in self.messages:
            if msg["is_user"]:
                imgui.text_colored("You: " + msg["text"], 0.2, 0.7, 0.2)
            else:
                imgui.text_colored("JARVIS: " + msg["text"], 0.2, 0.2, 0.7)
        imgui.end_child()
        
    def add_user_message(self, text: str):
        self.messages.append({"text": text, "is_user": True})
        
    def add_assistant_message(self, text: str):
        self.messages.append({"text": text, "is_user": False})
