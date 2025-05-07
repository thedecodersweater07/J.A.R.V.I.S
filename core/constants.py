"""Core constants used throughout JARVIS"""
import platform
import sys

# OpenGL availability check
try:
    if platform.system() == "Windows":
        import win32gui
        import win32con
    import OpenGL.GL as gl
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

# Security constants (moved from existing code)
SECURITY = {
    "jwt_secret": "your-secret-key",
    "token_expiry_hours": 24,
    "max_login_attempts": 3,
    "lockout_duration_minutes": 15
}
