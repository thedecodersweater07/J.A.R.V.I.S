from typing import Optional
from sqlalchemy.orm import Session
from .core import LLMCore
from .memory import ConversationMemory

class InferenceEngine:
    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session
        self.llm = LLMCore(db_session=db_session)
        self.memory = ConversationMemory(db_session) if db_session else None
        
    def process_input(self, text: str) -> str:
        context = self.memory.get_recent_context() if self.memory else []
        
        # Enhance input with context
        enhanced_input = self._prepare_input_with_context(text, context)
        
        return self.llm.generate_response(enhanced_input)
        
    def _prepare_input_with_context(self, text: str, context: list) -> str:
        if not context:
            return text
            
        context_str = "\n".join([
            f"Previous: {c['input']} -> {c['response']}"
            for c in context[-2:] # Use last 2 conversations for context
        ])
        
        return f"{context_str}\nCurrent: {text}"
