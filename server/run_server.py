from __future__ import annotations

# Apply compatibility patches first
import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import compatibility patches
from compat import *  # noqa

import argparse
import json
import logging
import traceback
import uvicorn
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union, cast,
    Protocol, TYPE_CHECKING, Mapping, Sequence, Tuple, TypedDict, Literal
)
from dataclasses import dataclass
from enum import Enum

# Import fastapi_framework after environment is loaded
import fastapi_framework

# Import for type checking only
if TYPE_CHECKING:
    from fastapi import FastAPI as FastAPIType
    from fastapi.routing import APIRouter as FastAPIRouterType
    from fastapi.responses import FileResponse as FastAPIFileResponseType

try:
    from fastapi import FastAPI as FastAPIObj, HTTPException, Request, WebSocket, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse as FastAPIFileResponse, JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles as FastAPIStaticFiles
    from fastapi.exceptions import RequestValidationError, WebSocketDisconnect
    from fastapi.routing import APIRouter as FastAPIRouter
    from fastapi.middleware import Middleware
    from starlette.types import ASGIApp as StarletteASGIApp, Receive, Scope, Send
    import uvicorn
    
    # Re-export types for consistency
    ASGIApp = StarletteASGIApp
    APIRouter = FastAPIRouter
    FileResponse = FastAPIFileResponse
    StaticFiles = FastAPIStaticFiles
    FastAPI = FastAPIObj
    
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    
    # Define basic types for type checking
    from typing import Any, Callable, Dict, List, Optional, Type, Union, Awaitable, TypeVar
    
    # Define ASGI protocol types
    Scope = Dict[str, Any]
    
    # Type aliases for ASGI protocol
    Receive = Callable[[], Awaitable[Dict[str, Any]]]
    Send = Callable[[Dict[str, Any]], Awaitable[None]]
    
    # Type alias for ASGI application
    T = TypeVar('T')
    ASGIApp = Callable[[Dict[str, Any], Any, Any], Awaitable[None]]
    ASGIApplication = ASGIApp  # For backward compatibility
    
    # Dummy FastAPI application class with necessary methods
    class FastAPI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.title = str(kwargs.get('title', ''))
            self.description = str(kwargs.get('description', ''))
            self.version = str(kwargs.get('version', '0.1.0'))
            self.docs_url = str(kwargs.get('docs_url', '/docs'))
            self.redoc_url = str(kwargs.get('redoc_url', '/redoc'))
            self._exception_handlers: Dict[Union[Type[Exception], int], Callable] = {}
            self.router = APIRouter()
            self.routes: List[Any] = []
            self._middleware: List[Any] = []
            self._mounts: List[Dict[str, Any]] = []
            
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            # Handle the request by passing it through middleware and router
            await self.router(scope, receive, send)
            
        def _add_middleware(self, middleware: Type[Any], **options: Any) -> None:
            """Add middleware to the application."""
            self._middleware.append((middleware, options))
            
        def _include_router(self, router: Any, *args: Any, **kwargs: Any) -> None:
            """Include a router in the application."""
            self.router.include_router(router, *args, **kwargs)
            
        def _mount(self, path: str, app: ASGIApp, name: str = None) -> None:
            """Mount an ASGI application at the given path."""
            self._mounts.append({'path': path, 'app': app, 'name': name})
            
        def _exception_handler(self, exc_class: Union[Type[Exception], int]) -> Callable:
            """Register an exception handler."""
            def decorator(func: Callable) -> Callable:
                self._exception_handlers[exc_class] = func
                return func
            return decorator
            
        # HTTP method decorators
        def get(self, path: str, *args: Any, **kwargs: Any) -> Callable:
            """Register a GET route."""
            def decorator(func: Callable) -> Callable:
                self.router.get(path, *args, **kwargs)(func)
                return func
            return decorator
            
        def post(self, path: str, *args: Any, **kwargs: Any) -> Callable:
            """Register a POST route."""
            def decorator(func: Callable) -> Callable:
                self.router.post(path, *args, **kwargs)(func)
                return func
            return decorator
            
        def websocket(self, path: str, *args: Any, **kwargs: Any) -> Callable:
            """Register a WebSocket route."""
            def decorator(func: Callable) -> Callable:
                self.router.websocket(path, *args, **kwargs)(func)
                return func
            return decorator
            
        # Alias methods to match FastAPI's interface
        add_middleware = _add_middleware
        include_router = _include_router
        mount = _mount
        exception_handler = _exception_handler
        
        def add_middleware(self, middleware: Type[Any], **options: Any) -> None:
            pass
        
        def include_router(self, router: Any, *args: Any, **kwargs: Any) -> None:
            pass
        
        def mount(self, path: str, app: ASGIApp, name: str = None) -> None:
            pass
        
        def exception_handler(self, exc_class: Union[Type[Exception], int]) -> Callable:
            def decorator(func: Callable) -> Callable:
                if isinstance(exc_class, int):
                    status_code = exc_class
                    # Handle HTTP status code based exception handlers
                    self._exception_handlers[status_code] = func
                else:
                    # Handle exception class based handlers
                    self._exception_handlers[exc_class] = func
                return func
            return decorator
    
    # Dummy request class
    class Request:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.scope: Dict[str, Any] = {}
    
    # Dummy WebSocket class
    class WebSocket:
        async def accept(self, *args: Any, **kwargs: Any) -> None:
            pass
        
        async def close(self, code: int = 1000) -> None:
            pass
    
    # Dummy exception classes
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: Any = None) -> None:
            self.status_code = status_code
            self.detail = detail
    
    class RequestValidationError(Exception):
        pass
    
    class WebSocketDisconnect(Exception):
        def __init__(self, code: int = 1000, reason: str = None) -> None:
            self.code = code
            self.reason = reason
    
    # Dummy middleware class
    class CORSMiddleware(ASGIApp):
        def __init__(self, app: ASGIApp, **kwargs: Any) -> None:
            self.app = app
            super().__init__()
            
        def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
            pass
    
    # Dummy response classes
    class FileResponse:
        def __init__(self, path: Union[str, os.PathLike], *args: Any, **kwargs: Any) -> None:
            # Convert path to string, handling Path objects and None
            self.path = str(Path(path)) if path is not None else ''
            self.status_code = int(kwargs.get('status_code', 200))
            self.headers = dict(kwargs.get('headers', {}))
            self.media_type = 'application/octet-stream'
            self.background = kwargs.get('background')
            
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            # Send a simple 404 response for the fallback implementation
            await send({
                'type': 'http.response.start',
                'status': 404,
                'headers': [
                    (b'content-type', b'text/plain'),
                ]
            })
            await send({
                'type': 'http.response.body',
                'body': b'File not found',
                'more_body': False
            })
    
    class StaticFiles:
        def __init__(self, directory: str, **kwargs: Any) -> None:
            self.directory = Path(directory)
            
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            try:
                path = scope.get('path', '').lstrip('/')
                file_path = self.directory / path
                
                if not file_path.exists() or not file_path.is_file():
                    await self._send_404(send)
                    return
                    
                # Determine content type based on file extension
                content_type = self._get_content_type(file_path)
                
                # Send file content
                with open(file_path, 'rb') as f:
                    content = f.read()
                    
                await send({
                    'type': 'http.response.start',
                    'status': 200,
                    'headers': [
                        (b'content-type', content_type.encode('utf-8')),
                        (b'content-length', str(len(content)).encode('utf-8')),
                    ]
                })
                
                await send({
                    'type': 'http.response.body',
                    'body': content,
                    'more_body': False
                })
                
            except Exception as e:
                logger.error(f"Error serving static file: {e}")
                await self._send_404(send)
        
        async def _send_404(self, send: Send) -> None:
            """Send a 404 response."""
            await send({
                'type': 'http.response.start',
                'status': 404,
                'headers': [
                    (b'content-type', b'text/plain'),
                ]
            })
            await send({
                'type': 'http.response.body',
                'body': b'Not Found',
                'more_body': False
            })
            
        def _get_content_type(self, file_path: Path) -> str:
            """Determine content type based on file extension."""
            ext = file_path.suffix.lower()
            content_types = {
                '.html': 'text/html',
                '.css': 'text/css',
                '.js': 'application/javascript',
                '.json': 'application/json',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
                '.ico': 'image/x-icon',
                '.woff': 'font/woff',
                '.woff2': 'font/woff2',
                '.ttf': 'font/ttf',
                '.eot': 'application/vnd.ms-fontobject',
                '.otf': 'font/otf',
            }
            return content_types.get(ext, 'application/octet-stream')
    
    class JSONResponse:
        def __init__(self, content: Any, status_code: int = 200, **kwargs: Any) -> None:
            self.content = content
            self.status_code = status_code
            self.headers = kwargs.get('headers', {})
            
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            await send({
                'type': 'http.response.start',
                'status': self.status_code,
                'headers': [
                    (k.encode('latin-1'), v.encode('latin-1')) 
                    for k, v in self.headers.items()
                ] + [(b'content-type', b'application/json')]
            })
            
            response_body = json.dumps(self.content).encode('utf-8')
            await send({
                'type': 'http.response.body',
                'body': response_body,
                'more_body': False
            })
            
    class HTMLResponse:
        def __init__(self, content: str, status_code: int = 200, **kwargs: Any) -> None:
            self.content = content
            self.status_code = status_code
            self.headers = kwargs.get('headers', {})
            
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            await send({
                'type': 'http.response.start',
                'status': self.status_code,
                'headers': [
                    (k.encode('latin-1'), v.encode('latin-1')) 
                    for k, v in self.headers.items()
                ] + [(b'content-type', b'text/html; charset=utf-8')]
            })
            
            response_body = self.content.encode('utf-8')
            await send({
                'type': 'http.response.body',
                'body': response_body,
                'more_body': False
            })
    
    # Router class implementing basic routing functionality
    class APIRouter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.routes: List[RouteInfo] = []
            self._prefix: str = str(kwargs.get('prefix', ''))
            self.tags: List[str] = list(kwargs.get('tags', []))
            self.dependencies: List[Any] = list(kwargs.get('dependencies', []))
            self.responses: Dict[Union[int, str], Dict[str, Any]] = dict(
                kwargs.get('responses', {})
            )
        
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            """Handle an incoming ASGI request."""
            path = scope.get('path', '')
            method = scope.get('method', 'GET').upper()
            
            for route in self.routes:
                if route.path == path and route.method == method:
                    try:
                        # Call the endpoint and get the response
                        response = await route.endpoint()
                        
                        # Handle different response types
                        if hasattr(response, '__call__') and hasattr(response, 'status_code'):
                            # If the response is callable (like FileResponse), let it handle the ASGI response
                            await response(scope, receive, send)
                            return
                        elif isinstance(response, dict) or isinstance(response, list):
                            # For JSON-serializable responses
                            await send({
                                'type': 'http.response.start',
                                'status': getattr(response, 'status_code', 200),
                                'headers': [
                                    (b'content-type', b'application/json'),
                                ]
                            })
                            response_body = json.dumps(response).encode('utf-8')
                            await send({
                                'type': 'http.response.body',
                                'body': response_body,
                                'more_body': False
                            })
                            return
                        else:
                            # For other response types, convert to string
                            await send({
                                'type': 'http.response.start',
                                'status': 200,
                                'headers': [
                                    (b'content-type', b'text/plain'),
                                ]
                            })
                            response_body = str(response).encode('utf-8')
                            await send({
                                'type': 'http.response.body',
                                'body': response_body,
                                'more_body': False
                            })
                            return
                            
                    except Exception as e:
                        logger.error(f"Error handling route {path}: {str(e)}", exc_info=True)
                        await self._send_error_response(
                            send, 
                            500, 
                            {"error": "Internal server error"}
                        )
                        return
            
            # No matching route found
            await self._send_error_response(send, 404, {"error": "Not found"})
        
        async def _send_error_response(
            self, 
            send: Send, 
            status_code: int, 
            error_data: Dict[str, Any]
        ) -> None:
            """Helper to send error responses."""
            await send(HttpResponseStartEvent(
                type='http.response.start',
                status=status_code,
                headers=[
                    (b'content-type', b'application/json'),
                ]
            ))
            
            error_body = json.dumps(error_data).encode('utf-8')
            await send(HttpResponseBodyEvent(
                type='http.response.body',
                body=error_body,
                more_body=False
            ))
        
        def include_router(
            self, 
            router: 'APIRouter', 
            prefix: str = "", 
            *args: Any, 
            **kwargs: Any
        ) -> None:
            """Include routes from another router."""
            if hasattr(router, 'routes'):
                # Create a copy of each route with the prefix applied
                for route in router.routes:
                    new_path = f"{prefix}{route.path}" if prefix else route.path
                    self._add_route(
                        path=new_path,
                        endpoint=route.endpoint,
                        methods=route.methods,
                        status_code=route.status_code,
                        tags=route.tags,
                        responses=route.responses,
                        **kwargs
                    )
        
        def _add_route(
            self, 
            path: str, 
            endpoint: Callable[..., Awaitable[Any]], 
            methods: List[str],
            *args: Any, 
            **kwargs: Any
        ) -> None:
            """Add a route with the given HTTP methods."""
            full_path = f"{self._prefix}{path}"
            
            # Extract route-specific overrides
            status_code = kwargs.pop('status_code', 200)
            tags = kwargs.pop('tags', self.tags.copy())
            responses = kwargs.pop('responses', self.responses.copy())
            
            # Create a route for each HTTP method
            for method in methods:
                route_kwargs = {
                    'path': full_path,
                    'endpoint': endpoint,
                    'methods': methods,
                    'method': method.upper(),
                    'path_params': {},
                    'dependencies': self.dependencies.copy(),
                    'tags': tags,
                    'responses': responses,
                    'status_code': status_code,
                }
                # Only pass known parameters to RouteInfo
                known_params = {k: v for k, v in kwargs.items() 
                              if k in {'summary', 'description', 'response_model', 'response_class', 'operation_id'}}
                route = RouteInfo(**route_kwargs, **known_params)
                self.routes.append(route)
        
        def get(self, path: str, *args: Any, **kwargs: Any) -> Callable:
            """Register a GET route."""
            def decorator(func: Callable) -> Callable:
                self._add_route(path, func, ['GET'], *args, **kwargs)
                return func
            return decorator
        
        def post(self, path: str, *args: Any, **kwargs: Any) -> Callable:
            """Register a POST route."""
            def decorator(func: Callable) -> Callable:
                self._add_route(path, func, ['POST'], *args, **kwargs)
                return func
            return decorator
        
        def websocket(self, path: str, *args: Any, **kwargs: Any) -> Callable:
            """Register a WebSocket route."""
            def decorator(func: Callable) -> Callable:
                self._add_route(path, func, ['WEBSOCKET'], *args, **kwargs)
                return func
            return decorator

# Type aliases
T = TypeVar('T')
SecurityManager = TypeVar('SecurityManager', bound=Any)
SecurityMiddleware = TypeVar('SecurityMiddleware', bound=Any)
WebSocketHandler = Any

# ASGI type definitions
class HttpRequestEvent(TypedDict):
    type: Literal['http.request']
    body: bytes
    more_body: bool

class HttpResponseStartEvent(TypedDict):
    type: Literal['http.response.start']
    status: int
    headers: List[Tuple[bytes, bytes]]

class HttpResponseBodyEvent(TypedDict):
    type: Literal['http.response.body']
    body: bytes
    more_body: bool

class WebSocketConnectEvent(TypedDict):
    type: Literal['websocket.connect']

class WebSocketDisconnectEvent(TypedDict):
    type: Literal['websocket.disconnect']
    code: int
    reason: str

class WebSocketReceiveEvent(TypedDict):
    type: Literal['websocket.receive']
    bytes: Optional[bytes]
    text: Optional[str]

class WebSocketSendEvent(TypedDict):
    type: Literal['websocket.send']
    bytes: Optional[bytes]
    text: Optional[str]
    accept: bool

Message = Union[
    HttpRequestEvent,
    HttpResponseStartEvent,
    HttpResponseBodyEvent,
    WebSocketConnectEvent,
    WebSocketDisconnectEvent,
    WebSocketReceiveEvent,
    WebSocketSendEvent
]

Scope = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

@dataclass
class RouteInfo:
    path: str
    endpoint: Callable[..., Awaitable[Any]]
    methods: List[str]
    method: str
    path_params: Dict[str, Any]
    dependencies: List[Any]
    tags: List[str]
    responses: Dict[Union[int, str], Dict[str, Any]]
    summary: Optional[str] = None
    description: Optional[str] = None
    response_model: Optional[Type[Any]] = None
    response_class: Optional[Type[Any]] = None
    status_code: int = 200
    operation_id: Optional[str] = None

# Apply compatibility patches first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import compatibility patches
from compat import *  # noqa

# Type aliases
SecurityManager = Any
SecurityMiddleware = Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-server")

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Configure logging early
def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'server.log')
        ]
    )

# Import modules with comprehensive error handling
def import_with_fallback() -> Dict[str, Any]:
    """Import server modules with fallbacks for missing components"""
    imported_modules: Dict[str, Any] = {
        'start_server': None,
        'security_manager': None,
        'security_middleware': None,
        'auth_router': None,
        'ai_router': None,
        'system_router': None,
        'websocket_router': None,
        'websocket_endpoint': None
    }
    
    # Try importing main server app
    try:
        try:
            from server.app import start_server
            imported_modules['start_server'] = start_server
            logger.info("Successfully imported server app")
        except ImportError as e:
            logger.warning(f"Could not import server.app.start_server: {e}")
            # Try alternative import
            try:
                from app import create_app as start_server
                imported_modules['start_server'] = start_server
                logger.info("Successfully imported alternative server app")
            except ImportError as e:
                logger.warning(f"Could not import app.create_app: {e}")
                # Create a dummy start_server function if neither import works
                def start_server() -> FastAPI:
                    return FastAPI()
                imported_modules['start_server'] = start_server
                logger.info("Created dummy start_server function")
    except Exception as e:
        logger.error(f"Unexpected error importing server app: {e}")
        # Ensure we always have a start_server function
        imported_modules['start_server'] = lambda: FastAPI()
    
    # Try importing security components
    try:
        from server.security.security_manager import SecurityManager as SecurityManagerImpl
        imported_modules['security_manager'] = SecurityManagerImpl
        logger.info("Successfully imported SecurityManager")
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"Could not import SecurityManager: {e}")
    
    try:
        from server.security.middleware import SecurityMiddleware as SecurityMiddlewareImpl
        imported_modules['security_middleware'] = SecurityMiddlewareImpl
        logger.info("Successfully imported SecurityMiddleware")
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"Could not import SecurityMiddleware: {e}")
    
    # Try importing API routers
    try:
        from server.api import auth
        imported_modules['auth_router'] = getattr(auth, 'router', None)
        logger.info("Successfully imported auth router")
    except ImportError as e:
        logger.warning(f"Could not import auth router: {e}")
    
    try:
        from server.api import ai
        imported_modules['ai_router'] = getattr(ai, 'router', None)
        logger.info("Successfully imported ai router")
    except ImportError as e:
        logger.warning(f"Could not import ai router: {e}")
    
    try:
        from server.api import system
        imported_modules['system_router'] = getattr(system, 'router', None)
        logger.info("Successfully imported system router")
    except ImportError as e:
        logger.warning(f"Could not import system router: {e}")
    
    # Try importing WebSocket components
    try:
        from server.websocket_handler import router as websocket_router, websocket_endpoint
        imported_modules['websocket_router'] = websocket_router
        imported_modules['websocket_endpoint'] = websocket_endpoint
        logger.info("Successfully imported WebSocket components")
    except ImportError as e:
        logger.warning(f"WebSocket components not available: {e}")
    
    return imported_modules

# Load configuration
def load_config() -> dict:
    """Load server configuration from file"""
    config_path = Path("config/server.json")
    default_config = {
        "cors_origins": ["*"],
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False,
        "security": {
            "enabled": True,
            "jwt_secret": "your-secret-key-here",
            "jwt_algorithm": "HS256",
            "jwt_expiration": 3600
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            logger.warning(f"Error loading config file: {e}, using defaults")
    else:
        logger.info("Config file not found, using defaults")
        # Create config directory and file
        config_path.parent.mkdir(exist_ok=True)
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file at {config_path}")
        except Exception as e:
            logger.warning(f"Could not create config file: {e}")
    
    return default_config

def create_basic_routes(app: FastAPI):
    """Create basic API routes when full routers are not available"""
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "JARVIS API"}
    
    @app.get("/api/status")
    async def api_status():
        """API status endpoint"""
        return {
            "status": "running",
            "version": "1.0.0",
            "message": "JARVIS API is operational"
        }
    
    @app.post("/api/chat")
    async def chat_fallback():
        """Fallback chat endpoint"""
        return {
            "response": "JARVIS AI system is starting up. Full functionality will be available shortly.",
            "status": "initializing"
        }

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not available. Please install it with: pip install fastapi uvicorn")
    
    # Load configuration
    config = load_config()
    
    # Import modules
    modules = import_with_fallback()
    
    # Get the start_server function
    start_server = modules.get('start_server')
    if not callable(start_server):
        logger.warning("No valid start_server function found, creating default app")
        start_server = FastAPI
    
    # Initialize the app
    try:
        app = start_server()
        if not isinstance(app, FastAPI):
            logger.warning("start_server() did not return a FastAPI instance, creating a new one")
            app = FastAPI(
                title="JARVIS API",
                description="JARVIS AI System API",
                version="1.0.0",
                docs_url="/api/docs",
                redoc_url="/api/redoc"
            )
        else:
            logger.info("Using imported server app")
    except Exception as e:
        logger.error(f"Error creating FastAPI app: {e}")
        app = FastAPI(
            title="JARVIS API",
            description="JARVIS AI System API",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add security middleware if available
    security_middleware = modules.get('security_middleware')
    security_manager = modules.get('security_manager')
    
    if security_middleware and config.get("security", {}).get("enabled", True):
        try:
            manager_instance = security_manager() if security_manager else None
            if manager_instance:
                app.add_middleware(security_middleware, security_manager=manager_instance)
                logger.info("Added security middleware")
        except Exception as e:
            logger.warning(f"Could not add security middleware: {e}")
    
    # Include API routers if available
    routers_added = 0
    for router_name, router in [
        ('auth', modules['auth_router']),
        ('ai', modules['ai_router']),
        ('system', modules['system_router'])
    ]:
        if router:
            try:
                app.include_router(router, prefix=f"/api/{router_name}")
                routers_added += 1
                logger.info(f"Added {router_name} router")
            except Exception as e:
                logger.warning(f"Could not add {router_name} router: {e}")
    
    # Add basic routes if no routers were added
    if routers_added == 0:
        create_basic_routes(app)
        logger.info("Added basic fallback routes")
    
    # Configure WebSocket if available
    if modules['websocket_router']:
        try:
            app.include_router(modules['websocket_router'])
            logger.info("Added WebSocket router")
        except Exception as e:
            logger.warning(f"Could not add WebSocket router: {e}")
    
    websocket_handler = modules.get('websocket_endpoint')
    if websocket_handler:
        try:
            @app.websocket("/ws/{client_id}")
            async def websocket_route(websocket: WebSocket, client_id: str):
                await websocket_handler(websocket, client_id)
            logger.info("Added WebSocket endpoint")
        except Exception as e:
            logger.warning(f"Could not add WebSocket endpoint: {e}")
    
    # Serve static files
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        try:
            app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
            logger.info("Mounted static files")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
    
    # Add root route
    @app.get("/")
    async def read_index():
        """Serve index page or API info"""
        index_path = web_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            return {
                "message": "JARVIS API is running",
                "status": "ok",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/health",
                    "api_status": "/api/status",
                    "docs": "/docs",
                    "redoc": "/redoc"
                }
            }
    
    # Add error handlers
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return {"error": "Endpoint not found", "status": 404}
    
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {exc}")
        return {"error": "Internal server error", "status": 500}
    
    logger.info("FastAPI application created successfully")
    return app

def create_server_app():
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="JARVIS Server",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status": exc.status_code}
        )
        
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "status": 500}
        )
    
    # Static files configuration
    project_root = Path(__file__).parent.parent
    static_dir = project_root / "server" / "web"
    
    # Ensure static directory exists
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # Mount static files
    static_files = StaticFiles(directory=str(static_dir))
    app.mount("/static", static_files, name="static")
    
    # Add a route to serve static files directly (for debugging)
    @app.get("/static/{file_path:path}")
    async def serve_static(file_path: str):
        static_file = static_dir / file_path
        if static_file.exists():
            return FileResponse(static_file)
        raise HTTPException(status_code=404, detail="File not found")
    
    # Create a test index.html if it doesn't exist
    test_index = static_dir / "index.html"
    if not test_index.exists():
        test_index.write_text("<html><body><h1>JARVIS Server is running</h1></body></html>")
    
    # Serve index.html at the root URL
    @app.get("/")
    async def read_root():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(
                path=index_path,
                media_type='text/html',
                headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
            )
        return HTMLResponse(
            content="<html><body><h1>JARVIS Server is running</h1><p>index.html not found in server/web directory</p></body>",
            status_code=404
        )
    
    # Import and include WebSocket router
    websocket_modules = {}
    try:
        # Try to import WebSocket endpoint if available
        from websocket import websocket_endpoint  # type: ignore
        
        @app.websocket("/ws/{client_id}")
        async def websocket_route(websocket: WebSocket, client_id: str) -> None:
            await websocket.accept()
            try:
                await websocket_endpoint(websocket, client_id)  # type: ignore
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                return
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                return
        
        logger.info("WebSocket endpoint registered at /ws/{client_id}")
        
        # Try to import WebSocket manager if available
        try:
            from websocket import manager  # type: ignore
            websocket_modules['websocket_manager'] = manager
            logger.info("WebSocket manager imported successfully")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Could not import WebSocket manager: {e}")
            
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"Could not import WebSocket endpoint: {e}")
    except Exception as e:
        logger.error(f"Error setting up WebSocket: {e}", exc_info=True)

    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "jarvis",
            "version": "1.0.0"
        }
    
    return app

def main():
    """Main server function"""
    # Setup argument parser
    parser = argparse.ArgumentParser(description='JARVIS Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set log level
    log_level = "debug" if args.debug else "info"
    logging.basicConfig(level=log_level.upper())
    
    logger.info(f"Starting JARVIS Server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    
    # Create and run the app
    app = create_server_app()
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=log_level,
        reload=args.debug
    )

if __name__ == "__main__":
    main()