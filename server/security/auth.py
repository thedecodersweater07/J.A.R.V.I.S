"""
Authentication and Authorization Module for JARVIS Server
Provides secure authentication and authorization functions
"""
import os
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt

logger = logging.getLogger("jarvis-server.security.auth")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

class AuthHandler:
    """
    Authentication and Authorization Handler
    Provides functions for user authentication and authorization
    """
    
    def __init__(self, security_manager):
        """Initialize with security manager"""
        self.security_manager = security_manager
        logger.info("AuthHandler initialized")
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
        """
        Get current user from JWT token
        
        Args:
            token: JWT token from Authorization header
            
        Returns:
            User information
            
        Raises:
            HTTPException: If token is invalid
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        user = self.security_manager.verify_token(token)
        if not user:
            logger.warning("Invalid token or user not found")
            raise credentials_exception
            
        return user
    
    async def get_current_active_user(self, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        """
        Check if user is active
        
        Args:
            current_user: User information from get_current_user
            
        Returns:
            User information if active
            
        Raises:
            HTTPException: If user is inactive
        """
        if not current_user.get("is_active", False):
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    async def get_admin_user(self, current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
        """
        Check if user has admin role
        
        Args:
            current_user: User information from get_current_active_user
            
        Returns:
            User information if admin
            
        Raises:
            HTTPException: If user is not admin
        """
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to perform this action"
            )
        return current_user
    
    async def check_permission(self, permission: str, current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
        """
        Check if user has specific permission
        
        Args:
            permission: Required permission
            current_user: User information from get_current_active_user
            
        Returns:
            User information if has permission
            
        Raises:
            HTTPException: If user doesn't have permission
        """
        if not self.security_manager.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized: missing {permission} permission"
            )
        return current_user
    
    async def login(self, form_data: OAuth2PasswordRequestForm, request: Request) -> Dict[str, Any]:
        """
        Authenticate user and return token
        
        Args:
            form_data: OAuth2 form data with username and password
            request: FastAPI request object for IP address
            
        Returns:
            Authentication result with token
        """
        client_ip = self._get_client_ip(request)
        
        result = self.security_manager.authenticate(
            username=form_data.username,
            password=form_data.password,
            ip_address=client_ip
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.get("error", "Invalid credentials"),
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "access_token": result["token"],
            "token_type": "bearer",
            "user_id": result["user"]["id"],
            "username": result["user"]["username"],
            "role": result["user"]["role"]
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# Function to create dependencies for use in FastAPI routes
def create_auth_dependencies(security_manager):
    """
    Create authentication dependencies for FastAPI routes
    
    Args:
        security_manager: Security manager instance
        
    Returns:
        Dictionary with auth dependencies
    """
    auth_handler = AuthHandler(security_manager)
    
    return {
        "get_current_user": auth_handler.get_current_user,
        "get_current_active_user": auth_handler.get_current_active_user,
        "get_admin_user": auth_handler.get_admin_user,
        "check_permission": auth_handler.check_permission,
        "login": auth_handler.login,
        "oauth2_scheme": oauth2_scheme
    }
