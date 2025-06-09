"""
Simplified Security Manager for JARVIS Server
"""
import os
import json
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger("jarvis.security")

class SecurityManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.jwt_secret = os.getenv('JWT_SECRET', 'jarvis-secret-key-2025')
        self.token_expiry = self.config.get('token_expiry_hours', 24)
        
        # Persistent storage
        self.data_file = Path("data/users.json")
        self.session_file = Path("data/sessions.json")
        self._ensure_data_dir()
        
        # Load data
        self.users = self._load_users()
        self.sessions = self._load_sessions()
        self.failed_attempts = {}
        
        # Create default admin if needed
        self._create_default_admin()
        
        logger.info("Security Manager initialized")
    
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        self.data_file.parent.mkdir(exist_ok=True)
    
    def _load_users(self) -> Dict:
        """Load users from persistent storage"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading users: {e}")
        return {}
    
    def _save_users(self):
        """Save users to persistent storage"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _load_sessions(self) -> Dict:
        """Load active sessions"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    sessions = json.load(f)
                # Clean expired sessions
                now = datetime.utcnow().timestamp()
                return {k: v for k, v in sessions.items() if v.get('expires', 0) > now}
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
        return {}
    
    def _save_sessions(self):
        """Save active sessions"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _create_default_admin(self):
        """Create default admin user"""
        if not self.users:
            admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123')
            self.users["admin"] = {
                "id": "admin_001",
                "username": "admin",
                "password_hash": self._hash_password(admin_pass),
                "role": "admin",
                "is_active": True,
                "created": datetime.utcnow().isoformat()
            }
            self._save_users()
            logger.info("Default admin user created")
    
    def _hash_password(self, password: str) -> str:
        """Hash password"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, hash_str: str) -> bool:
        """Verify password"""
        return bcrypt.checkpw(password.encode('utf-8'), hash_str.encode('utf-8'))
    
    def authenticate(self, username: str, password: str, ip_address: str = None) -> Dict:
        """Authenticate user"""
        # Check if user exists
        user = self.users.get(username)
        if not user or not user.get("is_active"):
            return {"success": False, "error": "Invalid credentials"}
        
        # Check password
        if not self._verify_password(password, user["password_hash"]):
            return {"success": False, "error": "Invalid credentials"}
        
        # Create token
        token_data = {
            "sub": username,
            "role": user["role"],
            "user_id": user["id"],
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry)
        }
        token = jwt.encode(token_data, self.jwt_secret, algorithm="HS256")
        
        # Store session
        session_id = f"{username}_{datetime.utcnow().timestamp()}"
        self.sessions[session_id] = {
            "username": username,
            "token": token,
            "ip_address": ip_address,
            "created": datetime.utcnow().isoformat(),
            "expires": (datetime.utcnow() + timedelta(hours=self.token_expiry)).timestamp()
        }
        self._save_sessions()
        
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"]
            }
        }
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            # First check if token exists in active sessions
            for session in self.sessions.values():
                if session.get("token") == token:
                    # Verify JWT
                    payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
                    user = self.users.get(payload["sub"])
                    if user:
                        return {
                            "id": user["id"],
                            "username": user["username"],
                            "role": user["role"],
                            "is_active": user["is_active"]
                        }
            return None
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def logout(self, token: str):
        """Logout user by removing session"""
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session.get("token") == token:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        self._save_sessions()
    
    def create_user(self, username: str, password: str, role: str = "user") -> Dict:
        """Create new user"""
        if username in self.users:
            return {"success": False, "error": "User already exists"}
        
        self.users[username] = {
            "id": f"user_{len(self.users)}",
            "username": username,
            "password_hash": self._hash_password(password),
            "role": role,
            "is_active": True,
            "created": datetime.utcnow().isoformat()
        }
        self._save_users()
        
        return {
            "success": True,
            "user": {
                "id": self.users[username]["id"],
                "username": username,
                "role": role
            }
        }
    
    def get_config(self) -> Dict:
        """Get security configuration"""
        return {
            "rate_limit": self.config.get("rate_limit", 100),
            "rate_window": self.config.get("rate_window", 60)
        }