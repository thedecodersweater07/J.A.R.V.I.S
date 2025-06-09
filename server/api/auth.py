"""
Authentication API Routes
Handles user authentication and authorization
"""
import os
import logging
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

# Setup logging
logger = logging.getLogger("jarvis-server.auth")

# Models
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str
    role: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# Create router
router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={404: {"description": "Not found"}},
)

# In-memory user store for development
user_store = {}
failed_attempts = {}

# Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    import bcrypt
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, 
                        secret_key: str = "your-secret-key-should-be-in-env-vars") -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=12))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret_key, algorithm="HS256")

def get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user from store"""
    if username in user_store:
        return user_store[username]
    return None

# Add a default admin user for testing
def add_default_user():
    """Add a default admin user for testing"""
    if not user_store:
        user_store["admin"] = {
            "id": "1",
            "username": "admin",
            "password_hash": get_password_hash("admin"),
            "role": "admin",
            "is_active": True
        }
        logger.info("Added default admin user")

# Routes
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint"""
    # Check for too many failed attempts
    if form_data.username in failed_attempts:
        attempts = failed_attempts[form_data.username]
        if attempts["count"] >= 3:
            lockout_time = attempts["timestamp"] + timedelta(minutes=15)
            if datetime.utcnow() < lockout_time:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account locked due to too many failed attempts"
                )
            else:
                # Reset counter after lockout period
                failed_attempts[form_data.username] = {"count": 0, "timestamp": datetime.utcnow()}
    
    # Get user and verify credentials
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        # Track failed attempt
        if form_data.username not in failed_attempts:
            failed_attempts[form_data.username] = {"count": 0, "timestamp": datetime.utcnow()}
        
        failed_attempts[form_data.username]["count"] += 1
        failed_attempts[form_data.username]["timestamp"] = datetime.utcnow()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Reset failed attempts on successful login
    if form_data.username in failed_attempts:
        failed_attempts.pop(form_data.username)
    
    # Create access token
    access_token_expires = timedelta(hours=12)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires,
        secret_key=os.environ.get("JWT_SECRET", "your-secret-key-should-be-in-env-vars")
    )
    
    # Update last login
    user["last_login"] = datetime.utcnow()
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"]
    }

@router.post("/register", response_model=Token)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # Check if username already exists
    if user_data.username in user_store:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    user_id = str(len(user_store) + 1)
    new_user = {
        "id": user_id,
        "username": user_data.username,
        "password_hash": get_password_hash(user_data.password),
        "role": user_data.role,
        "is_active": True,
        "created_at": datetime.utcnow()
    }
    
    # Add to store
    user_store[user_data.username] = new_user
    
    # Create access token
    access_token_expires = timedelta(hours=12)
    access_token = create_access_token(
        data={"sub": new_user["username"], "role": new_user["role"]},
        expires_delta=access_token_expires,
        secret_key=os.environ.get("JWT_SECRET", "your-secret-key-should-be-in-env-vars")
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": new_user["id"],
        "username": new_user["username"],
        "role": new_user["role"]
    }

# Initialize default user
add_default_user()
