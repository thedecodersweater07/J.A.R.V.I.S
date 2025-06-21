"""
Security Middleware for JARVIS Server
Handles rate limiting, security headers, and request validation.
"""
import time
import logging
from typing import Callable, Dict, Set, List, Any
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger("jarvis.middleware")

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware that provides:
    - Rate limiting
    - Basic request validation
    - Security headers
    - Suspicious pattern detection
    """
    
    def __init__(self, app: ASGIApp, security_manager: Any = None):
        super().__init__(app)
        self.security_manager = security_manager
        self.request_counts: Dict[str, List[float]] = {}
        self.rate_limit = 100  # requests per minute
        self.rate_window = 60  # seconds
        self.blocked_ips: Set[str] = set()
        
        # Suspicious patterns to block
        self.suspicious_patterns = [
            'script', 'select', 'union', 'drop', 'delete',
            '../', '..\\', 'eval\(', 'exec\(', '--', ';',
            '<script', 'onerror', 'onload', 'javascript:'
        ]
        
        # Initialize from security manager if available
        if security_manager and hasattr(security_manager, 'config'):
            config = security_manager.config
            self.rate_limit = config.get("rate_limit", self.rate_limit)
            self.rate_window = config.get("rate_limit_window", self.rate_window)
        else:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            logger.warning("No security manager provided, using default configuration")
        
        logger.info(f"Security middleware initialized (rate limit: {self.rate_limit} req/{self.rate_window}s)")
=======
=======
>>>>>>> Stashed changes
            # Access config directly from security_manager
            config = getattr(self.security_manager, 'config', {})
            self.rate_limit = config.get("rate_limit", 100)
            self.rate_window = config.get("rate_limit_window", 60)  # Note: using rate_limit_window to match SecurityManager
            self.blocked_ips = set()
            
        logger.info("Security middleware initialized")
>>>>>>> Stashed changes
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process incoming requests with security checks"""
        client_ip = self._get_client_ip(request)
        
        # Skip security checks for static files
        if request.url.path.startswith('/static/'):
            return await call_next(request)
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked request from blocked IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"}
            )
        
        # Rate limiting
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests"}
            )
        
        # Check for suspicious patterns
        if self._has_suspicious_patterns(request):
            logger.warning(f"Blocked suspicious request from {client_ip}: {request.url.path}")
            self.blocked_ips.add(client_ip)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Suspicious request detected"}
            )
        
        # Process request
        start_time = time.time()
        try:
            response = await call_next(request)
            self._add_security_headers(response)
            logger.info(f"{client_ip} - {request.method} {request.url.path} - {response.status_code} - {(time.time() - start_time):.3f}s")
            return response
        except Exception as e:
            logger.error(f"Request error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        if x_forwarded_for := request.headers.get("x-forwarded-for"):
            return x_forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        
        # Clean up old timestamps
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                t for t in self.request_counts[client_ip]
                if current_time - t < self.rate_window
            ]
        else:
            self.request_counts[client_ip] = []
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.rate_limit:
            return False
        
        self.request_counts[client_ip].append(current_time)
        return True
    
    def _has_suspicious_patterns(self, request: Request) -> bool:
        """Check for suspicious patterns in request"""
        # Check URL
        path = str(request.url)
        if any(pattern in path.lower() for pattern in self.suspicious_patterns):
            return True
        
        # Check headers
        for header in request.headers.values():
            if any(pattern in header.lower() for pattern in self.suspicious_patterns):
                return True
        
        # Check query parameters
        for param in request.query_params.values():
            if any(pattern in param.lower() for pattern in self.suspicious_patterns):
                return True
        
        return False
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response"""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline';",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for key, value in headers.items():
            response.headers[key] = value

# Alias for backward compatibility
SimpleSecurityMiddleware = SecurityMiddleware