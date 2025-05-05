from typing import Optional, Dict
import jwt
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, user_manager, secret_key: str):
        self.user_manager = user_manager
        self.secret_key = secret_key
        self.token_expiry = timedelta(hours=12)
        
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        user = self.user_manager.verify_credentials(username, password)
        if not user:
            return None
            
        token_data = {
            "sub": user["id"],
            "username": username,
            "role": user["role"],
            "exp": datetime.utcnow() + self.token_expiry
        }
        
        return jwt.encode(token_data, self.secret_key, algorithm="HS256")
        
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
