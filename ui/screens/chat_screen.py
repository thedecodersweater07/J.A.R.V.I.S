import imgui
from typing import List, Dict, Any
from .base_screen import BaseScreen
from llm.pipeline import LLMPipeline

class ChatScreen(BaseScreen):
    def __init__(self, llm_pipeline: LLMPipeline):
        super().__init__()
        self.llm = llm_pipeline
        self.input_text = ""
        self.chat_history: List[Dict[str, str]] = []
        
    def init(self) -> bool:
        self.initialized = True
        return True
        
    def render(self, frame_data: Dict[str, Any]) -> None:
        imgui.begin("JARVIS Chat", flags=imgui.WINDOW_NO_COLLAPSE)
        
        # Chat history
        for msg in self.chat_history:
            if msg["type"] == "user":
                imgui.text_colored("You: " + msg["text"], 0.2, 0.7, 0.2)
            else:
                imgui.text_colored("JARVIS: " + msg["text"], 0.2, 0.2, 0.7)
        
        # Input field
        changed, self.input_text = imgui.input_text(
            "Input", self.input_text, 1024,
            flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        )
        
        if changed and self.input_text.strip():
            self._process_input(self.input_text)
            self.input_text = ""
            
        imgui.end()
        
    def _process_input(self, text: str):
        self.chat_history.append({"type": "user", "text": text})
        response = self.llm.process(text)
        self.chat_history.append({"type": "assistant", "text": response})
        
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        if "text" in input_data:
            self._process_input(input_data["text"])
