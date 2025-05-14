"""
Security Manager for JARVIS Server
Integrates core security components with the server
"""
import os
import logging
import jwt
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import bcrypt
import secrets
import string

# Import core security components
try:
    from security.config.security_config import SecurityConfig
    from security.auth.auth_service import AuthService
    from security.models.user import User
    from security.authentication.identity_verifier import IdentityVerifier
    CORE_SECURITY_AVAILABLE = True
except ImportError:
    CORE_SECURITY_AVAILABLE = False
    logging.warning("Core security components not available, using server-only security")

logger = logging.getLogger("jarvis-server.security")

class SecurityManager:
    """
    Security Manager for JARVIS Server
    Handles authentication, authorization, and security integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the security manager with configuration"""
        self.config = config or {}
        self.users = {}
        self.failed_attempts = {}
        self.lockout_duration = self.config.get("lockout_duration_minutes", 15) * 60  # Convert to seconds
        self.max_login_attempts = self.config.get("max_login_attempts", 3)
        self.token_expiry_hours = self.config.get("token_expiry_hours", 12)
        
        # Get or generate JWT secret
        self.jwt_secret = self.config.get("jwt_secret", os.environ.get("JWT_SECRET", None))
        if not self.jwt_secret:
            # Generate a secure random secret if none is provided
            self.jwt_secret = self._generate_secure_key()
            logger.warning("No JWT secret provided, generated a random one. This will invalidate existing tokens on restart.")
        
        # Initialize core security components if available
        self.auth_service = None
        self.identity_verifier = None
        
        if CORE_SECURITY_AVAILABLE:
            try:
                # Create security config
                security_config = SecurityConfig(
                    jwt_secret=self.jwt_secret,
                    token_expiry_hours=self.token_expiry_hours,
                    max_login_attempts=self.max_login_attempts,
                    lockout_duration_minutes=int(self.lockout_duration / 60)
                )
                
                # Initialize auth service
                self.auth_service = AuthService(security_config)
                
                # Initialize identity verifier
                self.identity_verifier = IdentityVerifier()
                self.identity_verifier.initialize()
                
                logger.info("Core security components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize core security components: {e}")
                self.auth_service = None
                self.identity_verifier = None
        
        logger.info("Security Manager initialized")
    
    def _generate_secure_key(self, length: int = 32) -> str:
        """Generate a secure random key"""
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def authenticate(self, username: str, password: str, ip_address: str = "") -> Dict[str, Any]:
        """
        Authenticate a user and return a token
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            
        Returns:
            Dict with authentication result
        """
        # Check if account is locked
        if self._is_account_locked(username):
            return {
                "success": False,
                "error": "Account is locked due to too many failed attempts"
            }
        
        # Try to authenticate with core auth service if available
        if self.auth_service:
            try:
                result = self.auth_service.authenticate(username, password, ip_address)
                return {
                    "success": result.success,
                    "token": result.token if result.success else None,
                    "user": result.user_info if result.success else None,
                    "error": result.error if not result.success else None
                }
            except Exception as e:
                logger.error(f"Error using core auth service: {e}")
                # Fall back to local authentication
        
        # Local authentication
        user = self._verify_credentials(username, password)
        if not user:
            self._record_failed_attempt(username, ip_address)
            return {
                "success": False,
                "error": "Invalid username or password"
            }
        
        # Generate token
        token = self._generate_token(user)
        
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user.get("id"),
                "username": user.get("username"),
                "role": user.get("role")
            }
        }
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to too many failed attempts"""
        if username not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[username]
        if attempts["count"] >= self.max_login_attempts:
            # Check if lockout period has expired
            lockout_time = attempts["timestamp"] + self.lockout_duration
            if datetime.now().timestamp() < lockout_time:
                return True
            # Reset counter after lockout period
            self.failed_attempts[username] = {"count": 0, "timestamp": datetime.now().timestamp()}
        
        return False
    
    def _record_failed_attempt(self, username: str, ip_address: str = ""):
        """Record a failed authentication attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = {"count": 0, "timestamp": datetime.now().timestamp()}
        
        self.failed_attempts[username]["count"] += 1
        self.failed_attempts[username]["timestamp"] = datetime.now().timestamp()
        self.failed_attempts[username]["last_ip"] = ip_address
        
        logger.warning(f"Failed login attempt for user {username} from IP {ip_address}")
    
    def _verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials"""
        if username not in self.users:
            return None
        
        user = self.users[username]
        try:
            if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
                return user
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
        
        return None
    
    def _generate_token(self, user: Dict[str, Any]) -> str:
        """Generate a JWT token for authenticated user"""
        payload = {
            "sub": user["username"],
            "id": user["id"],
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a JWT token and return user information
        
        Args:
            token: JWT token
            
        Returns:
            User information if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            username = payload.get("sub")
            
            # Check if user exists
            if username in self.users:
                user = self.users[username]
                return {
                    "id": user["id"],
                    "username": user["username"],
                    "role": user["role"],
                    "is_active": user.get("is_active", True)
                }
            
            return None
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def create_user(self, username: str, password: str, role: str = "user") -> Dict[str, Any]:
        """
        Create a new user
        
        Args:
            username: Username
            password: Password
            role: User role
            
        Returns:
            Created user information
        """
        # Check if username already exists
        if username in self.users:
            return {
                "success": False,
                "error": "Username already exists"
            }
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Generate user ID
        user_id = str(len(self.users) + 1)
        
        # Create user
        self.users[username] = {
            "id": user_id,
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        logger.info(f"Created user {username} with role {role}")
        
        return {
            "success": True,
            "user": {
                "id": user_id,
                "username": username,
                "role": role
            }
        }
    
    def add_default_user(self):
        """Add a default admin user for testing"""
        if not self.users:
            self.create_user("admin", "admin", "admin")
            logger.info("Added default admin user")
    
    def check_permission(self, user: Dict[str, Any], required_permission: str) -> bool:
        """
        Check if user has the required permission
        
        Args:
            user: User information
            required_permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        role = user.get("role", "guest")
        
        # Admin has all permissions
        if role == "admin":
            return True
        
        # Check role-based permissions
        role_permissions = {
            "admin": ["all"],
            "user": ["read", "write", "execute"],
            "guest": ["read"]
        }
        
        permissions = role_permissions.get(role, [])
        
        return "all" in permissions or required_permission in permissions
