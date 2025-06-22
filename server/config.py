"""
J.A.R.V.I.S. Configuration
-------------------------
Central configuration management for the J.A.R.V.I.S. server.
"""

import os
import logging
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings managed by Pydantic."""
    
    # Basic settings
    debug: bool = False
    environment: str = "development"
    version: str = "2.0.0"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging
    log_level: int = logging.INFO
    
    # Security
    secret_key: str = os.getenv("JARVIS_SECRET_KEY", "your-super-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
    
    # Static files
    static_dir: str = "web/static"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Environment-specific settings
if settings.environment == "production":
    settings.debug = False
    settings.cors_origins = ["https://your-production-domain.com"]
    settings.log_level = logging.WARNING
elif settings.environment == "testing":
    settings.debug = True
    settings.port = 8001
    settings.log_level = logging.DEBUG
