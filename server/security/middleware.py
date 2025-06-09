"""
Simple Security Middleware
"""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("jarvis.middleware")

class SimpleSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, security_manager=None):
        super().__init__(app)
        self.security_manager = security_manager
        self.request_counts = {}
        self.rate_limit = 100  # requests per minute
        self.rate_window = 60  # seconds
        
        # Basic suspicious patterns
        self.suspicious_patterns = [
            'script', 'select', 'union', 'drop', 'delete',
            '../', '..\\', 'eval(', 'exec('
        ]
        
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with basic security checks"""
        client_ip = self._get_client_ip(request)
        
        # Rate limiting
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"}
            )
        
        # Basic security checks
        if self._has_suspicious_patterns(request):
            logger.warning(f"Suspicious request from {client_ip}: {request.url.path}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request"}
            )
        
        # Process request
        start_time = time.time()
        try:
            response = await call_next(request)
            self._add_security_headers(response)
            self._log_request(request, response, time.time() - start_time)
            return response
        except Exception as e:
            logger.error(f"Request error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Simple rate limiting"""
        current_time = time.time()
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {"count": 0, "window_start": current_time}
        elif current_time - self.request_counts[client_ip]["window_start"] > self.rate_window:
            self.request_counts[client_ip] = {"count": 0, "window_start": current_time}
        
        self.request_counts[client_ip]["count"] += 1
        
        if self.request_counts[client_ip]["count"] > self.rate_limit:
            return False
        
        return True
    
    def _has_suspicious_patterns(self, request: Request) -> bool:
        """Check for suspicious patterns"""
        path = request.url.path.lower()
        query = str(request.query_params).lower()
        content = path + " " + query
        
        return any(pattern in content for pattern in self.suspicious_patterns)
    
    def _add_security_headers(self, response: Response):
        """Add basic security headers"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
    
    def _log_request(self, request: Request, response: Response, duration: float):
        """Log requests"""
        if request.url.path not in ["/health", "/favicon.ico"]:
            client_ip = self._get_client_ip(request)
            logger.info(f"{client_ip} - {request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")