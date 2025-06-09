from pydantic import BaseModel, Field
from typing import Optional

class UserCreate(BaseModel):
    """User creation model with validation"""
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$"
    )
    password: str = Field(
        ..., 
        min_length=8,
        max_length=100
    )
    role: str = Field(
        default="user",
        pattern="^(admin|user|guest)$"
    )

    model_config = {
        "json_schema_extra": {  # Updated from schema_extra
            "example": {
                "username": "john_doe",
                "password": "SecurePass123!",
                "role": "user"
            }
        }
    }

class UserResponse(BaseModel):
    id: str
    username: str
    role: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str 
    role: str

class SecurityStatusResponse(BaseModel):
    status: str
    active_users: int
    locked_accounts: int
    suspicious_ips: int
