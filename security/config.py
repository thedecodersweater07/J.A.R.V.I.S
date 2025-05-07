from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    token_expiry_hours: int = 12
    max_login_attempts: int = 3
    lockout_duration_minutes: int = 15
    password_min_length: int = 8
    jwt_secret: str = "your-secure-secret-key"  # Should be loaded from env
    hash_rounds: int = 12

# Default config instance
SECURITY_CONFIG = SecurityConfig()
