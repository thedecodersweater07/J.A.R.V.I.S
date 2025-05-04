from typing import Dict, Optional
from datetime import datetime
import random

class ResponseGenerator:
    def __init__(self):
        self.templates = {
            "greeting": [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I help?"
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Until next time!"
            ]
        }

    def generate(self, 
                base_response: str, 
                context: Dict,
                intent: Optional[str] = None,
                sentiment: Optional[str] = None) -> str:
                
        # Use template if available
        if intent in self.templates:
            response = random.choice(self.templates[intent])
        else:
            response = base_response

        # Add context-aware modifications
        if context["topic"] != "general":
            response = f"Regarding {context['topic']}, {response}"

        # Add sentiment-aware modifications
        if sentiment == "negative":
            response = "I understand your concern. " + response
        elif sentiment == "positive":
            response = "I'm glad to hear that! " + response

        return response
