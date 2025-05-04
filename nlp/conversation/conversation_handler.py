import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import random
from datetime import datetime
from nlp.language_processor import LanguageProcessor  # Updated import

@dataclass
class ConversationHandler:
    language_processor: LanguageProcessor
    conversation_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        logging.debug(f"Initialized ConversationHandler with ID: {self.conversation_id}")

    def start_conversation(self, user_id: str) -> str:
        self.conversation_id = f"{user_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.conversation_history = []
        logging.info(f"Started new conversation with ID: {self.conversation_id}")
        return self.conversation_id

    def process_message(self, message: str) -> str:
        logging.debug(f"Processing message: {message}")
        response = self.language_processor.process(message)
        self.conversation_history.append({"user": message, "bot": response})
        logging.debug(f"Updated conversation history: {self.conversation_history}")
        return response

    def end_conversation(self) -> None:
        logging.info(f"Ending conversation with ID: {self.conversation_id}")
        self.conversation_id = None
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        logging.debug(f"Retrieving conversation history: {self.conversation_history}")
        return self.conversation_history