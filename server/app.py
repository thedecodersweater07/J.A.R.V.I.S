import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import jwt
import bcrypt
import json
import asyncio
import threading
import time

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
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Optional, Dict, List
    
    @dataclass
    class User:
        id: str
        username: str
        password_hash: str
        role: str = "user"
        created_at: datetime = datetime.now()
        last_login: Optional[datetime] = None
        status: str = "active"
        
        @property
        def is_active(self) -> bool:
            return self.status == "active"

# Try to import optional components
LLMCore = None
ModelManager = None
NLPProcessor = None
DatabaseManager = None

try:
    from llm.core.llm_core import LLMCore
except ImportError:
    logger.warning("LLMCore not found, LLM features will be disabled")

try:
    from ml.models.model_manager import ModelManager
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
app = FastAPI(title="JARVIS API Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
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

class AIRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    request_type: str = "text"  # text, nlp, ml, full

class AIResponse(BaseModel):
    response: Union[str, Dict[str, Any]]
    request_id: str
    processing_time: float
    timestamp: str

# Global components
security_config = None
db_manager = None
llm_core = None
model_manager = None
nlp_processor = None
user_store = {}  # In-memory user store for development
failed_attempts = {}  # Track failed login attempts

# Initialize components
def init_components():
    global security_config, db_manager, llm_core, model_manager, nlp_processor
    
    # Load security config
    security_config = SecurityConfig(
        jwt_secret=os.environ.get("JWT_SECRET", "your-secret-key-should-be-in-env-vars"),
        token_expiry_hours=12,
        max_login_attempts=3,
        lockout_duration_minutes=15
    )
    
    # Initialize database
    try:
        db_manager = DatabaseManager()
        logger.info("Database manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        db_manager = None
    
    # Initialize LLM
    try:
        llm_config = {
            "model": {
                "name": "gpt2",  # Use a reliable fallback model
                "type": "transformer"
            }
        }
        llm_core = LLMCore(config=llm_config)
        logger.info("LLM core initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        llm_core = None
    
    # Initialize ML components
    try:
        model_manager = ModelManager()
        logger.info("Model manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
        model_manager = None
    
    # Initialize NLP processor
    try:
        nlp_processor = NLPProcessor()
        logger.info("NLP processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize NLP processor: {e}")
        nlp_processor = None
    
    # Add default admin user for testing
    add_default_user()

def add_default_user():
    """Add a default admin user for testing"""
    try:
        password_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        user_store["admin"] = {
            "id": "user_0",
            "username": "admin",
            "password_hash": password_hash,
            "role": "admin"
        }
        logger.info("Default admin user created")
    except Exception as e:
        logger.error(f"Failed to create default user: {e}")

# Helper functions
def get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user from store or database"""
    if username in user_store:
        return user_store[username]
    
    if db_manager:
        # Try to get from database
        try:
            user = db_manager.get_user(username)
            if user:
                return {
                    "id": user.id,
                    "username": user.username,
                    "password_hash": user.password_hash,
                    "role": user.role
                }
        except Exception as e:
            logger.error(f"Database error getting user: {e}")
    
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

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to JARVIS API Server", "status": "online"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint"""
    # Check if account is locked
    username = form_data.username
    current_time = datetime.utcnow()
    
    if username in failed_attempts:
        attempts = [attempt for attempt in failed_attempts[username] 
                   if current_time - attempt < timedelta(minutes=security_config.lockout_duration_minutes)]
        failed_attempts[username] = attempts  # Clean up old attempts
        
        if len(attempts) >= security_config.max_login_attempts:
            logger.warning(f"Account locked: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is locked due to too many failed attempts",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Get user and verify password
    user = get_user(username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        # Record failed attempt
        if username not in failed_attempts:
            failed_attempts[username] = []
        failed_attempts[username].append(current_time)
        
        logger.warning(f"Failed login attempt: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Clear failed attempts on successful login
    if username in failed_attempts:
        failed_attempts[username] = []
    
    # Create access token
    access_token_expires = timedelta(hours=security_config.token_expiry_hours)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, 
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {username}")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"]
    }

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
    init_components()
    logger.info("JARVIS API Server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("JARVIS API Server shutting down")

def start_server(host="0.0.0.0", port=8000):
    """Start the server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
