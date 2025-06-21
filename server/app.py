# Standard library imports
import os
import sys
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
import socket

# Third-party imports
import jwt
from jwt.exceptions import PyJWTError
import bcrypt
from fastapi import (
    FastAPI, Depends, HTTPException, status, Request, 
    BackgroundTasks, Response, WebSocket, WebSocketDisconnect
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

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

# Set up paths
BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"

# Set up Vite build output paths
DIST_DIR = BASE_DIR / "web" / "dist"
ASSETS_DIR = DIST_DIR / "assets"

# Initialize FastAPI app
app = FastAPI(
    title="JARVIS Web Interface",
    description="JARVIS AI Assistant Web Interface",
    version="1.0.0",
    docs_url=None,  # Disable docs in production
    redoc_url=None,  # Disable redoc in production
)

# Serve static files
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# --- STATIC FILE SERVING ---

# Detect production mode (set ENV=production for prod)
IS_PROD = os.environ.get("ENV", "dev").lower() == "production"

if IS_PROD:
    # Serve the Vite build output as the static root
    app.mount(
        "",
        StaticFiles(directory=str(DIST_DIR), html=True),
        name="static-root"
    )
    logger.info(f"Serving static files from {DIST_DIR} (PRODUCTION mode)")
    # Redirect / to /index.html for best compatibility
    @app.get("/")
    async def prod_root():
        return RedirectResponse("/index.html")
else:
    # DEV: Serve from web/ for hot reload and dev
    @app.get("/", response_class=HTMLResponse)
    async def serve_dev_index():
        index_path = BASE_DIR / "web" / "index.html"
        if not index_path.exists():
            return HTMLResponse(content="<h1>index.html not found in web/</h1>", status_code=404)
        return FileResponse(index_path)

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def serve_spa_fallback_dev(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = BASE_DIR / "web" / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(BASE_DIR / "web" / "index.html")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize templates
try:
    if WEB_DIR.exists():
        templates = Jinja2Templates(directory=str(WEB_DIR))
        logger.info(f"Templates initialized from {WEB_DIR}")
    else:
        logger.warning(f"Web directory not found: {WEB_DIR}")
        # Create web directory if it doesn't exist
        WEB_DIR.mkdir(parents=True, exist_ok=True)
        # Create a basic index.html if it doesn't exist
        (WEB_DIR / "index.html").write_text(
            """<!DOCTYPE html>
            <html>
            <head>
                <title>JARVIS Web Interface</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    h1 { color: #2c3e50; }
                </style>
            </head>
            <body>
                <h1>JARVIS Web Interface</h1>
                <p>Welcome to JARVIS AI Assistant</p>
                <p>Place your web interface files in the 'server/web' directory.</p>
            </body>
            </html>""")
        logger.info(f"Created default web interface at {WEB_DIR}/index.html")
        templates = Jinja2Templates(directory=str(WEB_DIR))
except Exception as e:
    logger.error(f"Failed to initialize web interface: {e}")
    raise

def initialize_jarvis_model():
    """
    Initialize the Jarvis model with proper error handling.
    
    Returns:
        An instance of JarvisModel
    """
    try:
        # Add project root to Python path if not already there
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import the model and base classes
        from models.jarvis import JarvisModel, LLMBase, NLPProtocol, NLPBase
        
        # Create a simple LLM implementation
        class SimpleLLM(LLMBase):
            def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
                return {"response": "Simple LLM response", "success": True}
                
            def complete(self, text: str, **kwargs: Any) -> Dict[str, Any]:
                return {"response": text + " [completed]", "success": True}
                
            def predict(self, text: str, **kwargs: Any) -> Dict[str, Any]:
                return {"prediction": "This is a prediction", "confidence": 0.9, "success": True}
                
            def summarize(self, text: str, **kwargs: Any) -> Dict[str, Any]:
                words = text.split()
                summary = ' '.join(words[:30]) + ('...' if len(words) > 30 else '')
                return {
                    "summary": summary,
                    "success": True,
                    "model": "simple-llm"
                }
        
        # Create a simple NLP analyzer
        class SimpleNLPAnalyzer(NLPBase):
            def analyze(self, text: str) -> Dict[str, Any]:
                return {
                    "entities": [{"text": text, "type": "UNKNOWN"}],
                    "sentiment": {"polarity": 0.0, "label": "neutral"},
                    "success": True
                }
                
            def analyze_sentiment(self, text: str) -> Dict[str, Any]:
                return {"sentiment": "neutral", "score": 0.0, "success": True}
                
            def extract_entities(self, text: str) -> List[Dict[str, Any]]:
                return [{"text": text, "type": "UNKNOWN", "start": 0, "end": len(text)}]
                
            def __call__(self, text: str) -> Dict[str, Any]:
                return self.analyze(text)
        
        # Initialize the model with the simple components
        llm = SimpleLLM()
        nlp_analyzer = SimpleNLPAnalyzer()
        
        # Try to import and use real implementations if available
        try:
            from llm.core.llm_core import LLMCore
            llm = LLMCore()
            logger.info("Initialized real LLM")
        except ImportError as e:
            logger.warning(f"Could not import LLM: {e}")
            logger.info("Using simple LLM implementation")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}", exc_info=True)
            logger.info("Falling back to simple LLM implementation due to error")
        
        # Initialize the model
        model = JarvisModel(
            llm=llm,
            nlp_analyzer=nlp_analyzer,
            config={"version": "1.0.0"}
        )
        
        logger.info("Jarvis model initialized successfully")
        return model
        
    except ImportError as e:
        logger.warning(f"Could not import Jarvis model: {e}")
        
        # Create a dummy model as fallback
        class DummyJarvisModel:
            def __init__(self):
                self.model_name = "dummy-model"
                
            def process_input(self, text: str, user_id: str = "default", **kwargs) -> Dict[str, Any]:
                return {
                    "response": "Dummy model response - JARVIS is not properly initialized",
                    "success": False,
                    "error": "Dummy model - JARVIS not properly initialized"
                }
        
        logger.warning("Falling back to dummy model")
        return DummyJarvisModel()
        
    except Exception as e:
        logger.critical(f"Critical error initializing Jarvis model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Jarvis model: {e}")
        logger.warning(f"Could not import Jarvis model: {e}")
        logger.warning("Falling back to dummy model")
        
# Database dependency
def get_db():
    """Dependency for getting database session"""
    from db import init_db
    
    # Initialize the database and get a session
    db = init_db()
    try:
        yield db
    finally:
        db.close()

# Initialize the model when the module loads
jarvis_model = initialize_jarvis_model()
app.state.jarvis_model = jarvis_model

def get_jarvis_model() -> Any:
    """
    Get the Jarvis model instance.
    
    Returns:
        An instance of JarvisModel or a fallback dummy model.
    """
    if not hasattr(app.state, 'jarvis_model'):
        app.state.jarvis_model = initialize_jarvis_model()
    return app.state.jarvis_model

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
    request_type: str = Field(..., pattern="^(text|nlp|ml|analysis)$")
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
    role: str = Field(default="user", pattern="^(admin|user|guest)$")

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
        except PyJWTError:
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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
            
        hashed_password = AuthManager.hash_password(password)
        user_id = str(len(user_store) + 1)
        user_data = {
            "id": user_id,
            "username": username,
            "hashed_password": hashed_password,
            "role": role,
            "created_at": datetime.utcnow(),
            "last_login": None,
            "is_active": True
        }
        user_store[username] = user_data
        return user_data
    
    @staticmethod
    def get_user(username: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        if not username:
            return None
        return user_store.get(username)
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        user = UserManager.get_user(username)
        if not user or not user.get("is_active", True):
            return None
            
        if not AuthManager.verify_password(password, user["hashed_password"]):
            return None
            
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
    
    if user is None or not user.get("is_active", True):
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

# Create web directory if it doesn't exist
web_dir.mkdir(parents=True, exist_ok=True)

# Mount static files from web directory
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

templates = Jinja2Templates(directory=str(web_dir))

# Serve static files directly from web directory
@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    # Skip WebSocket upgrade requests
    if file_path == 'ws' or file_path.startswith('ws/'):
        raise HTTPException(status_code=404, detail="Not Found")
        
    # Handle root path
    if not file_path or file_path == "/":
        file_path = "index.html"
    
    static_file = web_dir / file_path
    
    # Try to serve the requested file
    if static_file.exists() and static_file.is_file():
        return FileResponse(static_file)
    
    # For API routes, return 404
    if file_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    
    # Fall back to index.html for SPA routing
    return FileResponse(web_dir / "index.html")

# Serve index.html for root path
@app.get("/")
async def read_root():
    return await serve_static("")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

# Initialize the connection manager
manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Get the AI model instance
                model = get_jarvis_model()
                
                if not model or not model.initialized:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "content": "AI model not initialized"
                        }),
                        websocket
                    )
                    continue

                # Process the message using the AI model
                response = await model.process_message(message["content"])
                
                # Send the response back
                await manager.send_personal_message(
                    json.dumps({
                        "type": "assistant",
                        "content": response
                    }),
                    websocket
                )
                
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "content": "Invalid message format"
                    }),
                    websocket
                )
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "content": f"Error processing message: {str(e)}"
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(json.dumps({
            "type": "system",
            "content": "A client disconnected"
        }))

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
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Process AI query with authentication and proper error handling.
    
    Args:
        request: The AI request containing query and context
        background_tasks: FastAPI background tasks
        current_user: Authenticated user information
        db: Database session dependency
        
    Returns:
        Dict containing the AI response and metadata
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    start_time = datetime.utcnow()
    request_id = f"req_{int(start_time.timestamp())}_{current_user['username']}"
    
    try:
        # Get the model instance
        model = get_jarvis_model()
        
        # Ensure the model has access to the database session
        if hasattr(model, 'db') and model.db is None:
            model.db = db
        
        # Process the request
        response = model.process_input(
            text=request.query,
            user_id=str(current_user.get('id', 'anonymous')),
            **({} if request.context is None else request.context)
        )
        
        # Ensure response is serializable
        response_data = response.get("response", "I'm sorry, I couldn't process that request.")
        if not isinstance(response_data, (str, int, float, bool, type(None))):
            response_data = str(response_data)
        
        # Log the request in background
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            log_ai_request,
            request_id=request_id,
            username=current_user['username'],
            request_type=request.request_type,
            processing_time=processing_time
        )
        
        return {
            "request_id": request_id,
            "response": response_data,
            "processing_time": processing_time,
            "model": response.get("model", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Log the error in the background
        error_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            log_ai_request,
            request_id=request_id,
            username=current_user['username'],
            request_type=request.request_type,
            processing_time=error_time
        )
        
        # Return a structured error response
        error_detail = {
            "error": "Internal server error",
            "details": error_msg,
            "request_id": request_id,
            "success": False
        }
        
        # Log the full error for debugging
        logger.error(f"Error processing request {request_id}: {error_msg}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
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
            "uptime": int((datetime.utcnow() - getattr(app.state, 'start_time', datetime.utcnow())).total_seconds()),
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

# Cleanup function to close database connections
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup database connections on shutdown"""
    if hasattr(app.state, 'db'):
        try:
            app.state.db.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

# Print local server URL on startup
@app.on_event("startup")
def print_localhost_url():
    port = int(os.environ.get("JARVIS_PORT", 8080))
    print(f"\n[INFO] JARVIS Web UI running at:")
    print(f"  http://localhost:{port}/")
    print(f"  http://127.0.0.1:{port}/\n")

# Include routers
from server.auth import router as auth_router
from server.ai import router as ai_router
from server.system import router as system_router

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(ai_router, prefix="/ai", tags=["ai"])
app.include_router(system_router, prefix="/system", tags=["system"])

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile=None,
        ssl_certfile=None
    )