"""System-wide constants and configuration values"""

SECURITY = {
    "jwt_secret": "your-256-bit-secure-key-here",  # TODO: Move to environment variable
    "token_expiry_hours": 12,
    "max_login_attempts": 3,
    "lockout_duration_minutes": 15
}
