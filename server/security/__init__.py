"""
JARVIS Server Security Module
Integrates the core security components with the server
"""
from .auth import create_auth_router, get_auth_router
from .auth import AuthService
from .middleware import SecurityMiddleware


__all__ = [
    'create_auth_router',
    'SecurityMiddleware'
]
#     content={"detail": "Your IP has been blocked due to suspicious activity."},
#         forwarded = request.headers.get("X-Forwarded-For")
from .auth import AuthService

__all__ += ['AuthService']
from .middleware import SecurityMiddleware
__all__ += ['SecurityMiddleware']

