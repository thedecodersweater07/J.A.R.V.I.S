import logging
from typing import Optional
from datetime import datetime

class ConversationScreen:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def display_header(self):
        print("\n" + "="*50)
        print(f"JARVIS Conversation Interface - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
    def display_prompt(self):
        print("\nJARVIS > ", end='')
        
    def display_response(self, response: str):
        print(f"JARVIS: {response}")
        
    def display_error(self, error: Optional[str] = None):
        if error:
            print(f"Error: {error}")
        print("JARVIS: Sorry, er is een fout opgetreden.")
        
    def clear_screen(self):
        print("\n" * 100)  # Simple screen clearing
