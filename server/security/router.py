"""
Security Router for JARVIS Server
Provides security-related API endpoints
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from server.models.security import UserCreate, UserResponse, TokenResponse, SecurityStatusResponse

logger = logging.getLogger("jarvis-server.security.router")

# Create router factory function
def create_security_router(security_manager, auth_handler):
    if not security_manager or not auth_handler:
        raise ValueError("Security manager and auth handler are required")
        
    router = APIRouter(
        prefix="/security",
        tags=["security"],
        responses={
            404: {"description": "Not found"},
            500: {"description": "Internal server error"},
            401: {"description": "Unauthorized"},
            403: {"description": "Forbidden"}
        },
    )
    
    @router.post("/token", response_model=TokenResponse)
    async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        request: Request = None
    ):
        """Login and get access token"""
        try:
            return await auth_handler.login(form_data, request)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Login error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred during login"
            )
    
    @router.post("/users", response_model=UserResponse)
    async def create_user(
        user: UserCreate,
        current_user: Dict[str, Any] = Depends(auth_handler.get_admin_user)
    ):
        """Create a new user (admin only)"""
        result = security_manager.create_user(
            username=user.username,
            password=user.password,
            role=user.role
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to create user")
            )
        
        return result["user"]
    
    @router.get("/status", response_model=SecurityStatusResponse)
    async def get_security_status(
        current_user: Dict[str, Any] = Depends(auth_handler.get_admin_user)
    ):
        """Get security status (admin only)"""
        # Count active users
        active_users = sum(1 for user in security_manager.users.values() if user.get("is_active", True))
        
        # Count locked accounts
        locked_accounts = sum(1 for username, attempts in security_manager.failed_attempts.items() 
                             if attempts["count"] >= security_manager.max_login_attempts)
        
        # Count suspicious IPs
        suspicious_ips = len(getattr(security_manager, "blocked_ips", set()))
        
        return {
            "status": "active",
            "active_users": active_users,
            "locked_accounts": locked_accounts,
            "suspicious_ips": suspicious_ips
        }
    
    @router.post("/unlock/{username}")
    async def unlock_account(
        username: str,
        current_user: Dict[str, Any] = Depends(auth_handler.get_admin_user)
    ):
        """Unlock a locked account (admin only)"""
        if username in security_manager.failed_attempts:
            security_manager.failed_attempts[username] = {"count": 0, "timestamp": 0}
            logger.info(f"Account {username} unlocked by admin {current_user['username']}")
            return {"status": "success", "message": f"Account {username} unlocked"}
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {username} not found or not locked"
        )
    
    @router.get("/me", response_model=UserResponse)
    async def get_current_user_info(
        current_user: Dict[str, Any] = Depends(auth_handler.get_current_active_user)
    ):
        """Get current user information"""
        return {
            "id": current_user["id"],
            "username": current_user["username"],
            "role": current_user["role"]
        }
    
    return router
