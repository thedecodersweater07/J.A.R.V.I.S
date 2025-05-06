import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, dict] = {}
        self.session_timeout = timedelta(hours=12)
        self.jwt_secret = "your-secure-secret-key"  # Should be in config
        
    def create_session(self, user_id: str, user_data: dict) -> str:
        session_token = jwt.encode({
            'user_id': user_id,
            'exp': datetime.utcnow() + self.session_timeout,
            'data': user_data
        }, self.jwt_secret, algorithm='HS256')
        
        self.active_sessions[session_token] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_active': datetime.utcnow(),
            'data': user_data
        }
        return session_token

    def validate_session(self, token: Optional[str] = None) -> bool:
        if not token or token not in self.active_sessions:
            return False
            
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            session = self.active_sessions[token]
            session['last_active'] = datetime.utcnow()
            return True
        except jwt.ExpiredSignatureError:
            self.end_session(token)
            return False
            
    def end_session(self, token: str) -> None:
        self.active_sessions.pop(token, None)
        
    def cleanup(self) -> None:
        expired = [
            token for token, session in self.active_sessions.items()
            if datetime.utcnow() - session['last_active'] > self.session_timeout
        ]
        for token in expired:
            self.end_session(token)
