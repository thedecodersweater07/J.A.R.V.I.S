from typing import List, Dict
from sqlalchemy.orm import Session
from ..db.models import Conversation

class ConversationMemory:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.context_window = 5
        
    def get_recent_context(self) -> List[Dict]:
        recent_conversations = self.db_session.query(Conversation)\
            .order_by(Conversation.timestamp.desc())\
            .limit(self.context_window)\
            .all()
        
        return [
            {
                "input": conv.user_input,
                "response": conv.response,
                "timestamp": conv.timestamp
            }
            for conv in recent_conversations
        ]
