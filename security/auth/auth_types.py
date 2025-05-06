"""Type definitions for authentication module"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AuthResult:
    """Authentication result"""
    success: bool
    error: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None

@dataclass
class LoginResponse:
    """Login response containing auth result and token"""
    success: bool
    token: Optional[str] = None
    error: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
