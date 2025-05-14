"""
Security Router for JARVIS Server
Provides security-related API endpoints
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

logger = logging.getLogger("jarvis-server.security.router")

# Models
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class UserResponse(BaseModel):
    id: str
    username: str
    role: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str
    role: str

class SecurityStatusResponse(BaseModel):
    status: str
    active_users: int
    locked_accounts: int
    suspicious_ips: int

# Create router factory function
def create_security_router(security_manager, auth_handler):
    """
    Create a security router with the given security manager and auth handler
    
    Args:
        security_manager: Security manager instance
        auth_handler: Authentication handler instance
        
    Returns:
        FastAPI router with security endpoints
    """
    router = APIRouter(
        prefix="/security",
        tags=["security"],
        responses={404: {"description": "Not found"}},
    )
    
    @router.post("/token", response_model=TokenResponse)
    async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        request: Request = None
    ):
        """Login and get access token"""
        return await auth_handler.login(form_data, request)
    
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
