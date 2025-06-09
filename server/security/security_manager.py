"""
Security Manager for JARVIS Server
Integrates core security components with the server
"""
import os
import logging
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from passlib.context import CryptContext

logger = logging.getLogger("jarvis-server.security")

class SecurityManager:
    """
    Security Manager for JARVIS Server
    Handles authentication, authorization, and security integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the security manager with configuration"""
        self.logger = logging.getLogger(__name__)
        if not config:
            self.logger.warning("No config provided, using defaults")
            config = {}
            
        self.config = config
        self.jwt_secret = config.get('jwt_secret', os.getenv('JWT_SECRET', 'fallback-secret-key'))
        self.token_expiry = config.get('token_expiry_hours', 12)
        self.users = {}
        self.failed_attempts = {}
        
        # Load secret keys from environment or config
        self.api_key = os.getenv("API_KEY", self.config.get("api_key", "your-api-key"))
        
        # Rate limiting settings
        self.rate_limit = self.config.get("rate_limit", 100)  # requests per minute
        self.rate_limit_window = self.config.get("rate_limit_window", 60)  # seconds
        
        # Initialize rate limiting storage
        self.request_counts = {}
        
        if 'fallback' in self.jwt_secret:
            self.logger.warning("Using fallback JWT secret - this is not secure for production!")

        logger.info("Security Manager initialized")
        self.add_default_user()  # Ensure default user is created
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), 
                             hashed_password.encode('utf-8'))
        
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(hours=12))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.jwt_secret, algorithm="HS256")
        
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            username: str = payload.get("sub")
            role: str = payload.get("role", "user")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return {"username": username, "role": role}
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
            
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.utcnow().timestamp()
        client_requests = self.request_counts.get(client_id, [])
        
        # Remove old requests
        client_requests = [req for req in client_requests 
                         if req > now - self.rate_limit_window]
        
        # Check limit
        if len(client_requests) >= self.rate_limit:
            return False
            
        # Update requests
        client_requests.append(now)
        self.request_counts[client_id] = client_requests
        return True
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.jwt_secret, algorithm="HS256")  # Use jwt_secret instead of SECRET_KEY
        
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token"""
        try:
            if hasattr(self, 'token_blacklist') and token in self.token_blacklist:
                return None
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])  # Use jwt_secret instead of SECRET_KEY
            return payload
        except jwt.PyJWTError:
            return None
    
    def add_default_user(self):
        """Add default admin user if no users exist"""
        try:
            if not self.users:
                password_hash = self.get_password_hash("admin")
                self.users["admin"] = {
                    "id": "admin_default",
                    "username": "admin",
                    "password_hash": password_hash,
                    "role": "admin",
                    "is_active": True
                }
                logger.info("Added default admin user")
        except Exception as e:
            logger.error(f"Error adding default user: {e}")
