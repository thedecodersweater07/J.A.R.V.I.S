"""
Simplified Authentication Handler
"""
import logging
from typing import Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

logger = logging.getLogger("jarvis.auth")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

class AuthHandler:
    def __init__(self, security_manager):
        self.security_manager = security_manager
        logger.info("Auth Handler initialized")
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
        """Get current user from token"""
        user = self.security_manager.verify_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    
    async def get_current_active_user(self, current_user: Dict[str, Any] = Depends(get_current_user)):
        """Check if user is active"""
        if not current_user.get("is_active", False):
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    async def get_admin_user(self, current_user: Dict[str, Any] = Depends(get_current_active_user)):
        """Check admin role"""
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return current_user
    
    async def login(self, form_data: OAuth2PasswordRequestForm, request: Request) -> Dict[str, Any]:
        """Login user"""
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
            "user": result["user"]
        }
    
    async def logout(self, token: str = Depends(oauth2_scheme)):
        """Logout user"""
        self.security_manager.logout(token)
        return {"message": "Logged out successfully"}
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"