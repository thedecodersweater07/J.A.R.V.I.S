from dataclasses import dataclass
from typing import Dict, List

@dataclass
class SecurityConfig:
    jwt_secret: str
    token_expiry_hours: int = 12
    password_min_length: int = 8
    max_login_attempts: int = 3
    lockout_duration_minutes: int = 15
    allowed_roles: List[str] = ("admin", "user", "guest")
    role_permissions: Dict[str, List[str]] = {
        "admin": ["all"],
        "user": ["read", "write", "execute"],
        "guest": ["read"]
    }
