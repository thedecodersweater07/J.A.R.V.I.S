"""
Centralized Authentication and User Management
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import jwt
from jwt.exceptions import PyJWTError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

# --- Configuration ---
class SecurityConfig:
    JWT_SECRET: str = os.getenv("JWT_SECRET", "a-secure-secret-key-that-should-be-in-env")
    ALGORITHM: str = "HS256"
    TOKEN_EXPIRE_HOURS: int = 24
    PASSWORD_MIN_LENGTH: int = 8

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")
logger = logging.getLogger("jarvis.security")

# --- Pydantic Models ---
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)

class UserCreate(UserBase):
    password: str = Field(..., min_length=SecurityConfig.PASSWORD_MIN_LENGTH)
    role: str = Field(default="user", pattern="^(admin|user|guest)$")

class User(UserBase):
    id: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: User

# --- In-memory User Store (for demonstration) ---
# In a real application, this would be a database.
user_store: Dict[str, Dict[str, Any]] = {}

# --- Core Logic ---
class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

    @staticmethod
    def create_access_token(data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=SecurityConfig.TOKEN_EXPIRE_HOURS)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SecurityConfig.JWT_SECRET, algorithm=SecurityConfig.ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            payload = jwt.decode(token, SecurityConfig.JWT_SECRET, algorithms=[SecurityConfig.ALGORITHM])
            return payload
        except PyJWTError as e:
            logger.warning(f"JWT Error: {e}")
            return None

class UserManager:
    @staticmethod
    def create_user(username: str, password: str, role: str = "user") -> Dict[str, Any]:
        if username in user_store:
            raise ValueError("Username already exists")
        
        user_id = str(len(user_store) + 1)
        new_user = {
            "id": user_id,
            "username": username,
            "hashed_password": AuthManager.hash_password(password),
            "role": role,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }
        user_store[username] = new_user
        return new_user

    @staticmethod
    def get_user(username: str) -> Optional[Dict[str, Any]]:
        return user_store.get(username)

    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        user = UserManager.get_user(username)
        if user and AuthManager.verify_password(password, user["hashed_password"]):
            user["last_login"] = datetime.utcnow()
            return user
        return None

# --- Dependency for protected routes ---
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = AuthManager.verify_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
        
    user = UserManager.get_user(username)
    if user is None:
        raise credentials_exception
        
    return user

async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Admin access required"
        )
    return current_user

# --- Initialize default users ---
def create_default_users():
    try:
        if not UserManager.get_user("admin"):
            UserManager.create_user("admin", "admin_password", "admin")
            logger.info("Default 'admin' user created.")
        if not UserManager.get_user("user"):
            UserManager.create_user("user", "user_password", "user")
            logger.info("Default 'user' user created.")
    except ValueError as e:
        logger.warning(f"Default user already exists: {e}")

# This function will be called on startup
create_default_users()

