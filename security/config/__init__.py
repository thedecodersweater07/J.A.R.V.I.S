"""Security configuration management module"""
from .security_config import SecurityConfig

# Create default config instance
SECURITY_CONFIG = SecurityConfig(
    jwt_secret="your-secure-secret-key"  # Should be loaded from env
)

__all__ = ['SecurityConfig', 'SECURITY_CONFIG']
