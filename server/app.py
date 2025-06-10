import os
import sys
import logging
import uvicorn 
import traceback
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root.parent))

from fastapi import FastAPI, Depends, HTTPException, Security, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocket
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import jwt
import bcrypt
import json
import asyncio
import threading
import time
import os
from pathlib import Path
from pydantic import BaseModel
from dataclasses import dataclass, field

# Define AI models
class AIRequest(BaseModel):
    query: str
    request_type: str
    context: Optional[Dict[str, Any]] = None

class AIResponse(BaseModel):
    response: Any
    request_id: str
    processing_time: float
    timestamp: str

# Import security models
from server.models.security import UserCreate, UserResponse, TokenResponse, SecurityStatusResponse
from server.security.security_manager import SecurityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("jarvis-server")

# Initialize FastAPI app
app = FastAPI(
    title="JARVIS Server",
    description="JARVIS AI Assistant Server",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configuration
@dataclass
class SecurityConfig:
    jwt_secret: str = "your-secret-key-should-be-in-env-vars"
    token_expiry_hours: int = 12
    password_min_length: int = 8
    max_login_attempts: int = 3
    lockout_duration_minutes: int = 15
    allowed_roles: List[str] = field(default_factory=lambda: ["admin", "user", "guest"])
    role_permissions: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "admin": ["all"],
            "user": ["read", "write", "execute"],
            "guest": ["read"]
        }
    )

# Initialize security config
security_config = SecurityConfig()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User store (temporary, replace with database)
user_store = {}

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import JARVIS components with error handling
try:
    from security.config.security_config import SecurityConfig
except ImportError:
    logger.warning("SecurityConfig not found, using default")
    from dataclasses import dataclass, field
    from typing import Dict, List
    
    @dataclass
    class SecurityConfig:
        jwt_secret: str
        token_expiry_hours: int = 12
        password_min_length: int = 8
        max_login_attempts: int = 3
        lockout_duration_minutes: int = 15
        allowed_roles: List[str] = field(default_factory=lambda: ["admin", "user", "guest"])
        role_permissions: Dict[str, List[str]] = field(
            default_factory=lambda: {
                "admin": ["all"],
                "user": ["read", "write", "execute"],
                "guest": ["read"]
            }
        )

try:
    from security.models.user import User
except ImportError:
    logger.warning("User model not found, using default")
    class User:
        def __init__(self, username: str, password_hash: str, role: str = "user"):
            import uuid
            self.id = str(uuid.uuid4())
            self.username = username
            self.password_hash = password_hash
            self.role = role
            self.created_at = datetime.now()
            self.last_login = None
            self.status = "active"
            
        def is_active(self):
            return self.status == "active"

# Add default admin user if not exists
if not user_store.get("admin"):
    user_store["admin"] = User(
        username="admin",
        password_hash=bcrypt.hashpw("admin".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        role="admin"
    )

# Initialize optional components
LLMCore = None
ModelManager = None
NLPProcessor = None

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=12)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, security_config.jwt_secret, algorithm="HS256")
DatabaseManager = None

try:
    from llm.core.llm_core import LLMCore
except ImportError:
    logger.warning("LLMCore not found, LLM features will be disabled")

try:
    from ml.model_manager import ModelManager
except ImportError:
    logger.warning("ModelManager not found, ML features will be disabled")

try:
    from nlp.processor import NLPProcessor
except ImportError:
    logger.warning("NLPProcessor not found, NLP features will be disabled")

try:
    from db.manager import DatabaseManager
except ImportError:
    logger.warning("DatabaseManager not found, database features will be disabled")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-server")

# Initialize FastAPI app
app = FastAPI(
    title="JARVIS API Server",
    version="1.0.0",
    description="API server for JARVIS AI Assistant",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

def init_middleware(security_manager=None, api_keys=None):
    """Initialize all middleware before app startup"""
    # Add CORS middleware first
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
        expose_headers=["Content-Disposition"],
        max_age=600,
    )
    
    # Add security middleware
    if security_manager:
        app.add_middleware(SecurityMiddleware, security_manager=security_manager)
    
    # Add API key middleware
    if api_keys:
        app.add_middleware(APIKeyMiddleware, api_keys=api_keys)

# Import security components
try:
    from server.security.middleware import SecurityMiddleware, APIKeyMiddleware
    from server.security.security_manager import SecurityManager
    from server.security.auth import create_auth_dependencies, AuthHandler
    from server.security.router import create_security_router
    SECURITY_COMPONENTS_AVAILABLE = True
    logger.info("Security components available")
except ImportError as e:
    logger.warning(f"Security components not available: {e}")
    SECURITY_COMPONENTS_AVAILABLE = False

# Import API routers
try:
    from server.api import auth, ai, system
    from server.api.dependencies import oauth2_scheme
    HAS_API_ROUTERS = True
except ImportError as e:
    logger.warning(f"API routers not found, using legacy routes: {e}")
    HAS_API_ROUTERS = False
    # OAuth2 scheme for legacy routes
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    # Legacy models
    class Token(BaseModel):
        access_token: str
        token_type: str
        user_id: str
        username: str
        role: str

    class TokenData(BaseModel):
        username: Optional[str] = None
        role: Optional[str] = None

    class UserCreate(BaseModel):
        username: str
        password: str
        role: str = "user"

    class LoginRequest(BaseModel):
        username: str
        password: str

    class TokenRequest(BaseModel):
        grant_type: str
        client_id: Optional[str] = None
        client_secret: Optional[str] = None

    class TokenResponse(BaseModel):
        access_token: str
        token_type: str
        expires_in: int

    class UserResponse(BaseModel):
        id: str
        username: str
        role: str

    class ErrorResponse(BaseModel):
        error: str
        error_description: Optional[str] = None

    class StatusResponse(BaseModel):
        status: str
        version: str

# Global components
security_config = None
db_manager = None
llm_core = None
model_manager = None
nlp_processor = None
security_manager = None
auth_dependencies = None
user_store = {}  # In-memory user store for development
failed_attempts = {}  # Track failed login attempts

# Initialize components
def init_components():
    global security_config, db_manager, llm_core, model_manager, nlp_processor, security_manager, auth_dependencies
    
    # Initialize security components
    if SECURITY_COMPONENTS_AVAILABLE:
        try:
            # Initialize security manager
            security_config = {
                "jwt_secret": os.environ.get("JWT_SECRET", "your-secret-key-should-be-in-env-vars"),
                "token_expiry_hours": 12,
                "max_login_attempts": 3,
                "lockout_duration_minutes": 15
            }
            security_manager = SecurityManager(config=security_config)
            
            # Create auth dependencies
            auth_dependencies = create_auth_dependencies(security_manager)
            
            logger.info("Security components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize security components: {e}")
            logger.error(traceback.format_exc())
            security_manager = None
            auth_dependencies = None
    else:
        # Fall back to basic security config
        security_config = SecurityConfig(
            jwt_secret=os.environ.get("JWT_SECRET", "your-secret-key-should-be-in-env-vars"),
            token_expiry_hours=12,
            max_login_attempts=3,
            lockout_duration_minutes=15
        )
    
    # Initialize database
    try:
        if DatabaseManager:
            db_manager = DatabaseManager()
            logger.info("Database manager initialized")
        else:
            logger.warning("DatabaseManager not available")
            db_manager = None
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.error(traceback.format_exc())
        db_manager = None
    
    # Initialize LLM
    try:
        if LLMCore:
            llm_config = {
                "model": {
                    "name": "gpt2",  # Use a reliable fallback model
                    "type": "transformer"
                }
            }
            llm_core = LLMCore(config=llm_config)
            logger.info("LLM core initialized")
        else:
            logger.warning("LLMCore not available")
            llm_core = None
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        logger.error(traceback.format_exc())
        llm_core = None
    
    # Initialize ML components
    try:
        if ModelManager:
            # ModelManager expects base_path parameter, not config
            model_path = os.environ.get("MODEL_PATH", os.path.abspath("data/models"))
            # Ensure directory exists
            os.makedirs(model_path, exist_ok=True)
            model_manager = ModelManager(base_path=model_path)
            logger.info(f"Model manager initialized with path: {model_path}")
        else:
            logger.warning("ModelManager not available")
            model_manager = None
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
        logger.error(traceback.format_exc())
        model_manager = None
    
    # Initialize NLP processor
    try:
        if NLPProcessor:
            # NLPProcessor expects model_name parameter, not config
            nlp_model = os.environ.get("NLP_MODEL", "nl_core_news_sm")
            nlp_processor = NLPProcessor(model_name=nlp_model)
            logger.info(f"NLP processor initialized with model: {nlp_model}")
        else:
            logger.warning("NLPProcessor not available")
            nlp_processor = None
    except Exception as e:
        logger.error(f"Failed to initialize NLP processor: {e}")
        logger.error(traceback.format_exc())
        nlp_processor = None
        
# Function to get AI components for use in API routes
def get_ai_components():
    """Get AI components for use in API routes"""
    global llm_core, model_manager, nlp_processor
    return llm_core, model_manager, nlp_processor
    
# Helper function for password hashing
def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Add default admin user for testing if using legacy routes
if not HAS_API_ROUTERS:
    # Add the default admin user
    if not user_store:
        user_store["admin"] = User(
            username="admin",
            password_hash=bcrypt.hashpw("admin".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            role="admin"
        )
        logger.info("Added default admin user")

# Helper functions for legacy routes
def get_user(username: str):
    """Get user from store or database"""
    if not username:
        return None
        
    # Try to get from user store
    user = user_store.get(username)
    if user:
        return user
        
    # Try to get from database if available
    
    return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=security_config.token_expiry_hours)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, security_config.jwt_secret, algorithm="HS256")
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Validate JWT token and return user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, security_config.jwt_secret, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=payload.get("role"))
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Check if user is active"""
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Setup static files and templates
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "web")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "web"))

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, error: Optional[str] = None, message: Optional[str] = None):
    """
    Serve the login page with optional error and success messages
    
    Args:
        request: The FastAPI request object
        error: Optional error message to display
        message: Optional success message to display
        
    Returns:
        HTMLResponse: The login page template
    """
    # Check if user is already authenticated
    token = request.cookies.get("access_token") or request.headers.get("authorization", "").replace("Bearer ", "")
    if token:
        try:
            # Verify the token
            payload = jwt.decode(
                token,
                security_config.jwt_secret if hasattr(security_config, 'jwt_secret') else "your-secret-key-should-be-in-env-vars",
                algorithms=["HS256"]
            )
            if payload.get("sub"):
                # User is authenticated, redirect to dashboard
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url="/dashboard")
        except (jwt.PyJWTError, Exception):
            # Token is invalid, clear it and show login page
            pass
    
    # Show login page with any error/message
    return templates.TemplateResponse(
        "login_page.html",
        {
            "request": request,
            "error": error,
            "message": message
        }
    )

@app.post("/token")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    response_class=JSONResponse
):
    """
    Login endpoint that authenticates users and returns JWT tokens
    
    Args:
        request: The FastAPI request object
        form_data: The OAuth2 form data containing username and password
        
    Returns:
        JSONResponse: Contains the access token and user info
    """
    try:
        # Get user from database
        user = get_user(form_data.username)
        
        # Verify user exists and password is correct
        if not user or not verify_password(form_data.password, user.get("password_hash", "")):
            # Log failed login attempt
            logger.warning(f"Failed login attempt for user: {form_data.username}")
            
            # Return to login page with error
            from fastapi.responses import RedirectResponse
            return RedirectResponse(
                url=f"/?error=invalid_credentials",
                status_code=status.HTTP_303_SEE_OTHER
            )
        
        # Update last login time
        if hasattr(user, 'update_last_login'):
            user.update_last_login()
        elif hasattr(user, 'last_login'):
            user["last_login"] = datetime.now()
        
        # Create access token
        access_token_expires = timedelta(hours=security_config.token_expiry_hours if hasattr(security_config, 'token_expiry_hours') else 12)
        access_token = create_access_token(
            data={"sub": user["username"], "role": user.get("role", "user")},
            expires_delta=access_token_expires
        )
        
        # Create response with token
        response = JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": int(access_token_expires.total_seconds()),
                "user": {
                    "username": user["username"],
                    "role": user.get("role", "user")
                }
            }
        )
        
        # Set secure HTTP-only cookie with the token
        response.set_cookie(
            key="access_token",
            value=f"Bearer {access_token}",
            httponly=True,
            max_age=3600 * 12,  # 12 hours
            secure=os.getenv("ENVIRONMENT") == "production",
            samesite="lax"
        )
        
        # Set a separate non-HttpOnly cookie for client-side access if needed
        response.set_cookie(
            key="user_session",
            value=user["username"],
            max_age=3600 * 12,  # 12 hours
            secure=os.getenv("ENVIRONMENT") == "production",
            samesite="lax"
        )
        
        # Redirect to dashboard
        response.headers["Location"] = "/dashboard"
        response.status_code = status.HTTP_303_SEE_OTHER
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return to login page with error
        from fastapi.responses import RedirectResponse
        return RedirectResponse(
            url=f"/?error=server_error",
            status_code=status.HTTP_303_SEE_OTHER
        )

@app.get("/api/verify-token")
async def verify_token(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """
    Verify if the provided token is valid
    
    Args:
        current_user: The authenticated user from the JWT token
        
    Returns:
        dict: Status and user information if token is valid
    """
    return {
        "status": "valid", 
        "user": {
            "username": current_user.get("sub"),
            "role": current_user.get("role", "user")
        }
    }

@app.post("/logout")
async def logout():
    """
    Logout endpoint that clears authentication cookies
    
    Returns:
        JSONResponse: Success message and redirect to login page
    """
    from fastapi.responses import JSONResponse, RedirectResponse
    
    # Create response with success message
    response = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Successfully logged out"}
    )
    
    # Clear authentication cookies
    response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=os.getenv("ENVIRONMENT") == "production",
        samesite="lax"
    )
    response.delete_cookie(
        key="user_session",
        secure=os.getenv("ENVIRONMENT") == "production",
        samesite="lax"
    )
    
    # Redirect to login page
    response.headers["Location"] = "/?message=logged_out"
    response.status_code = status.HTTP_303_SEE_OTHER
    
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """
    Serve the main dashboard
    
    This endpoint verifies the JWT token and serves the dashboard page.
    If the token is invalid or expired, it will redirect to the login page.
    """
    try:
        # Verify the token is still valid
        token = request.cookies.get("access_token") or request.headers.get("authorization", "").replace("Bearer ", "")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
            
        # Verify the token
        try:
            payload = jwt.decode(
                token,
                security_config.jwt_secret if hasattr(security_config, 'jwt_secret') else "your-secret-key-should-be-in-env-vars",
                algorithms=["HS256"]
            )
            username = payload.get("sub")
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
        # Token is valid, serve the dashboard
        response = templates.TemplateResponse("index.html", {
            "request": request,
            "user": {
                "username": current_user.get("sub", ""),
                "role": current_user.get("role", "user")
            }
        })
        
        # Set secure cookie with token
        response.set_cookie(
            key="access_token",
            value=f"Bearer {token}",
            httponly=True,
            max_age=3600 * 12,  # 12 hours
            secure=os.getenv("ENVIRONMENT") == "production",
            samesite="lax"
        )
        
        return response
        
    except HTTPException as he:
        # If there's an authentication error, redirect to login
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/?error=unauthorized")
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/users/", response_model=Dict[str, Any])
async def create_user(user: UserCreate, current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Create new user (admin only)"""
    # Check if user has admin role
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check if username already exists
    if get_user(user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    user_id = f"user_{len(user_store) + 1}"
    
    new_user = {
        "id": user_id,
        "username": user.username,
        "password_hash": hashed_password,
        "role": user.role
    }
    
    # Store user
    user_store[user.username] = new_user
    
    # Also try to store in database if available
    if db_manager:
        try:
            db_manager.create_user(
                user.username,
                hashed_password,
                user.role
            )
        except Exception as e:
            logger.error(f"Database error creating user: {e}")
    
    logger.info(f"User created: {user.username}")
    return {
        "username": user.username,
        "role": user.role,
        "id": user_id
    }

@app.get("/users/me", response_model=Dict[str, Any])
async def read_users_me(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Get current user info"""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"]
    }

@app.post("/ai/query", response_model=AIResponse)
async def process_ai_query(
    request: AIRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Process AI query"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    # Check if required components are initialized
    if request.request_type == "text" and not llm_core:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service not available"
        )
    elif request.request_type == "nlp" and not nlp_processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLP service not available"
        )
    elif request.request_type == "ml" and not model_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML service not available"
        )
    
    # Add user context
    context = request.context or {}
    context["user"] = {
        "id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"]
    }
    
    # Process request based on type
    response = None
    try:
        if request.request_type == "text":
            response = llm_core.generate_response(request.query, context)
        elif request.request_type == "nlp":
            response = nlp_processor.process(request.query, context)
        elif request.request_type == "ml":
            model_name = context.get("model_name", "default")
            response = model_manager.predict(model_name, request.query)
        elif request.request_type == "full":
            # Process with all components
            nlp_result = None
            ml_result = None
            llm_result = None
            
            if nlp_processor:
                try:
                    nlp_result = nlp_processor.process(request.query, context)
                except Exception as e:
                    logger.error(f"NLP processing error: {e}")
            
            if model_manager:
                try:
                    model_name = context.get("model_name", "default")
                    ml_result = model_manager.predict(model_name, request.query)
                except Exception as e:
                    logger.error(f"ML processing error: {e}")
            
            # Add NLP and ML results to context for LLM
            if nlp_result:
                context["nlp_result"] = nlp_result
            if ml_result:
                context["ml_result"] = ml_result
            
            # Process with LLM
            if llm_core:
                llm_result = llm_core.generate_response(request.query, context)
            
            # Combine results
            response = {
                "llm": llm_result,
                "nlp": nlp_result,
                "ml": ml_result,
                "combined": llm_result  # Use LLM result as combined result
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown request type: {request.request_type}"
            )
    except Exception as e:
        logger.error(f"Error processing AI request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Log request
    logger.info(f"AI request processed: {request.request_type} in {processing_time:.2f}s")
    
    return AIResponse(
        response=response,
        request_id=request_id,
        processing_time=processing_time,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "components": {
            "database": db_manager is not None,
            "llm": llm_core is not None,
            "ml": model_manager is not None,
            "nlp": nlp_processor is not None
        },
        "timestamp": datetime.now().isoformat()
    }
    return status

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        init_components()
        logger.info("Server components initialized successfully")
        
        # Include API routers if available
        if HAS_API_ROUTERS:
            app.include_router(auth.router)
            app.include_router(ai.router)
            app.include_router(system.router)
            app.include_router(websocket_handler_router)
            logger.info("API routers registered")
            
        # Include security router if available
        if SECURITY_COMPONENTS_AVAILABLE and security_manager:
            # Create and include security router with proper auth handler
            security_router = create_security_router(
                security_manager=security_manager,
                auth_handler=AuthHandler(security_manager)  # Create new AuthHandler instance
            )
            app.include_router(security_router)
            logger.info("Security router registered")
            
            # Add default admin user
            security_manager.add_default_user()
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(traceback.format_exc())

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down")
    # Add cleanup code here
    try:
        # Close database connections
        if db_manager and hasattr(db_manager, 'close'):
            await db_manager.close()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        logger.error(traceback.format_exc())

def start_server(host="127.0.0.1", port=8000):
    """Start the server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
