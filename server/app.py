import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager

import jwt
import bcrypt
from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger("jarvis-server")

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize FastAPI app
app = FastAPI(
    title="JARVIS API",
    description="JARVIS AI Assistant API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("DEBUG") else None,
    redoc_url="/redoc" if os.getenv("DEBUG") else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
try:
    templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
    logger.info("Templates initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize templates: {e}")
    templates = None

# Initialize static files
try:
    static_dir = PROJECT_ROOT / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info("Static files mounted at /static")
    else:
        logger.warning(f"Static directory not found: {static_dir}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Initialize Jarvis model with fallback
def initialize_jarvis_model():
    """Initialize the Jarvis model with fallback to dummy model if needed"""
    try:
        # Try to import the actual model
        from models.jarvis import JarvisModel as RealJarvisModel
        
        logger.info("Initializing Jarvis model...")
        model = RealJarvisModel()
        model.initialized = True  # Ensure the model has the initialized flag
        logger.info("Jarvis model initialized successfully")
        return model
        
    except ImportError as e:
        logger.warning(f"Could not import Jarvis model: {e}")
        logger.warning("Falling back to dummy model")
        
        # Create a dummy model if the real one can't be imported
        class DummyJarvisModel:
            def __init__(self):
                self.initialized = True
                self.model_name = "dummy-model"
                
            def process(self, text, **kwargs):
                return {
                    "response": f"Dummy response to: {text}",
                    "success": True,
                    "model": self.model_name,
                    "error": "Running in fallback mode - model initialization failed"
                }
                
        return DummyJarvisModel()
    
    except Exception as e:
        logger.error(f"Failed to initialize Jarvis model: {e}")
        raise

# Initialize the model when the module loads
jarvis_model = initialize_jarvis_model()
app.state.jarvis_model = jarvis_model

def get_jarvis_model():
    """Get the Jarvis model instance"""
    if not hasattr(jarvis_model, 'initialized') or not jarvis_model.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Jarvis model is not available"
        )
    return jarvis_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis.log')
    ]
)
logger = logging.getLogger("jarvis-server")

# Security Configuration
class SecurityConfig:
    JWT_SECRET: str = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-change-in-production")
    ALGORITHM: str = "HS256"
    TOKEN_EXPIRE_HOURS: int = 12
    PASSWORD_MIN_LENGTH: int = 8
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION: int = 15  # minutes

# Pydantic Models
class AIRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    request_type: str = Field(..., regex="^(text|nlp|ml|analysis)$")
    context: Optional[Dict[str, Any]] = None

class AIResponse(BaseModel):
    response: Any
    request_id: str
    processing_time: float
    timestamp: datetime
    success: bool = True

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    role: str = Field(default="user", regex="^(admin|user|guest)$")

class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: Dict[str, str]

# In-memory stores (replace with database in production)
user_store: Dict[str, Dict[str, Any]] = {}
failed_attempts: Dict[str, Dict[str, Any]] = {}

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

# AI Components Registry
class AIComponentRegistry:
    def __init__(self):
        self.components = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize AI components asynchronously"""
        try:
            # Mock AI components - replace with real implementations
            self.components = {
                "text": self._mock_text_processor,
                "nlp": self._mock_nlp_processor,
                "ml": self._mock_ml_processor,
                "analysis": self._mock_analysis_processor
            }
            self.initialized = True
            logger.info("AI components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            raise
    
    async def process_request(self, request_type: str, query: str, context: Dict[str, Any]) -> Any:
        """Process AI request"""
        if not self.initialized:
            raise HTTPException(status_code=503, detail="AI components not initialized")
        
        processor = self.components.get(request_type)
        if not processor:
            raise HTTPException(status_code=400, detail=f"Unknown request type: {request_type}")
        
        return await processor(query, context)
    
    # Mock processors - replace with real AI implementations
    async def _mock_text_processor(self, query: str, context: Dict[str, Any]) -> str:
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Text response for: {query[:50]}..."
    
    async def _mock_nlp_processor(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        return {
            "tokens": len(query.split()),
            "sentiment": "neutral",
            "entities": [],
            "language": "en"
        }
    
    async def _mock_ml_processor(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.3)
        return {
            "prediction": "positive",
            "confidence": 0.85,
            "model": "mock_classifier"
        }
    
    async def _mock_analysis_processor(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.4)
        return {
            "summary": f"Analysis of query with {len(query)} characters",
            "key_points": ["Point 1", "Point 2"],
            "recommendations": ["Recommendation 1"]
        }

# Global AI registry
ai_registry = AIComponentRegistry()

# Authentication utilities
class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            return False
    
    @staticmethod
    def create_access_token(data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=SecurityConfig.TOKEN_EXPIRE_HOURS)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SecurityConfig.JWT_SECRET, algorithm=SecurityConfig.ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SecurityConfig.JWT_SECRET, algorithms=[SecurityConfig.ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

# User management
class UserManager:
    @staticmethod
    def create_user(username: str, password: str, role: str = "user") -> Dict[str, Any]:
        """Create a new user"""
        if username in user_store:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        user_id = f"user_{len(user_store) + 1:04d}"
        user_data = {
            "id": user_id,
            "username": username,
            "password_hash": AuthManager.hash_password(password),
            "role": role,
            "created_at": datetime.utcnow(),
            "last_login": None,
            "status": "active"
        }
        user_store[username] = user_data
        logger.info(f"User created: {username} with role: {role}")
        return user_data
    
    @staticmethod
    def get_user(username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        return user_store.get(username)
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        user = UserManager.get_user(username)
        if not user or not AuthManager.verify_password(password, user["password_hash"]):
            return None
        
        # Update last login
        user["last_login"] = datetime.utcnow()
        return user

# Initialize default users
def create_default_users():
    """Create default admin user"""
    try:
        UserManager.create_user("admin", "admin123", "admin")
        UserManager.create_user("demo", "demo123", "user")
        logger.info("Default users created")
    except HTTPException:
        logger.info("Default users already exist")

# Dependency functions
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    payload = AuthManager.verify_token(token)
    username = payload.get("sub")
    user = UserManager.get_user(username)
    
    if user is None or user.get("status") != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    return user

async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Ensure current user is admin"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting JARVIS server...")
    create_default_users()
    await ai_registry.initialize()
    logger.info("JARVIS server started successfully")
    
    yield
    
    # Shutdown
    logger.info("JARVIS server shutting down...")

# FastAPI app initialization
app = FastAPI(
    title="JARVIS AI Assistant",
    description="Advanced AI Assistant API with authentication and multiple AI capabilities",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Static files setup
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    templates = Jinja2Templates(directory=str(web_dir))

# API Routes
@app.post("/api/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access token"""
    user = UserManager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = AuthManager.create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=SecurityConfig.TOKEN_EXPIRE_HOURS * 3600,
        user={"username": user["username"], "role": user["role"]}
    )

@app.get("/api/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        role=current_user["role"],
        created_at=current_user["created_at"],
        last_login=current_user.get("last_login")
    )

@app.post("/api/users", response_model=UserResponse)
async def create_new_user(
    user: UserCreate,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Create a new user (admin only)"""
    created_user = UserManager.create_user(user.username, user.password, user.role)
    return UserResponse(
        id=created_user["id"],
        username=created_user["username"],
        role=created_user["role"],
        created_at=created_user["created_at"]
    )

@app.get("/api/users", response_model=List[UserResponse])
async def list_users(current_user: Dict[str, Any] = Depends(get_admin_user)):
    """List all users (admin only)"""
    return [
        UserResponse(
            id=user["id"],
            username=user["username"],
            role=user["role"],
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )
        for user in user_store.values()
    ]

@app.post("/api/ai/query", response_model=AIResponse)
async def process_ai_request(
    request: AIRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Process AI query with authentication"""
    import time
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000000)}"
    
    # Add user context
    context = request.context or {}
    context.update({
        "user_id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"],
        "timestamp": datetime.utcnow().isoformat()
    })
    
    try:
        # Process AI request
        response_data = await ai_registry.process_request(
            request.request_type,
            request.query,
            context
        )
        
        processing_time = time.time() - start_time
        
        # Log request in background
        background_tasks.add_task(
            log_ai_request,
            request_id,
            current_user["username"],
            request.request_type,
            processing_time
        )
        
        return AIResponse(
            response=response_data,
            request_id=request_id,
            processing_time=processing_time,
            timestamp=datetime.utcnow(),
            success=True
        )
        
    except Exception as e:
        logger.error(f"AI processing error for user {current_user['username']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI processing failed: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "ai_registry": ai_registry.initialized,
            "user_store": len(user_store) > 0
        },
        "version": "2.0.0"
    }

@app.get("/api/stats")
async def get_stats():
    """Get server statistics"""
    try:
        jarvis = get_jarvis_model()
        return {
            "model_type": "Jarvis",
            "status": "active",
            "is_processing": False,
            "uptime": int((datetime.utcnow() - app.start_time).total_seconds()) if hasattr(app, 'start_time') else 0,
            "components": {
                "llm": hasattr(jarvis, 'llm_service'),
                "nlp": hasattr(jarvis, 'nlp_processor'),
                "ml": hasattr(jarvis, 'model')
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "model_type": "Jarvis",
            "status": "error",
            "error": str(e)
        }

# Background tasks
async def log_ai_request(request_id: str, username: str, request_type: str, processing_time: float):
    """Log AI request for analytics"""
    logger.info(f"AI Request - ID: {request_id}, User: {username}, Type: {request_type}, Time: {processing_time:.3f}s")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )