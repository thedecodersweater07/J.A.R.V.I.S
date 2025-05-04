from typing import Callable, Optional
import queue
import logging

class TextInputHandler:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.callbacks = []
        self.logger = logging.getLogger(__name__)
        
    def register_callback(self, callback: Callable[[str], None]):
        """Registreer callback voor tekstinvoer verwerking"""
        self.callbacks.append(callback)
        
    def handle_input(self, text: str) -> bool:
        """Verwerk binnenkomende tekstinvoer"""
        try:
            self.input_queue.put(text)
            for callback in self.callbacks:
                callback(text)
            return True
        except Exception:
            return False

    def get_next_input(self) -> Optional[str]:
        """Haal volgende tekstinvoer op uit queue"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

class TextInput:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def listen(self) -> str:
        """Get input from text"""
        try:
            text = input("JARVIS > ")
            return text.strip()
        except EOFError:
            return ""
        except KeyboardInterrupt:
            return ""
