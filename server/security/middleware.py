"""
Security Middleware for JARVIS Server
Provides request/response security features
"""
import time
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("jarvis-server.security.middleware")

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for JARVIS server
    Adds security headers, rate limiting, and request validation
    """
    
    def __init__(self, app, security_manager=None):
        super().__init__(app)
        self.security_manager = security_manager
        self.request_counts = {}
        self.rate_limit = 100  # requests per minute
        self.rate_window = 60  # seconds
        self.sensitive_headers = ["authorization", "cookie"]
        self.blocked_ips = set()
        self.suspicious_patterns = [
            "../../", "SELECT ", "UNION ", "<script>", 
            "../etc/passwd", "/bin/bash", "eval(", "exec("
        ]
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and add security features"""
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied"}
            )
        
        # Check for rate limiting
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"}
            )
        
        # Check for suspicious patterns
        if self._has_suspicious_patterns(request):
            self._record_suspicious_activity(client_ip)
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request"}
            )
        
        # Process the request
        start_time = time.time()
        try:
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log request
            self._log_request(request, response, time.time() - start_time)
            
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        
        # Initialize or clean up old entries
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {"count": 0, "window_start": current_time}
        elif current_time - self.request_counts[client_ip]["window_start"] > self.rate_window:
            # Reset window if it has expired
            self.request_counts[client_ip] = {"count": 0, "window_start": current_time}
        
        # Increment count
        self.request_counts[client_ip]["count"] += 1
        
        # Check if limit exceeded
        if self.request_counts[client_ip]["count"] > self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            return False
        
        return True
    
    def _has_suspicious_patterns(self, request: Request) -> bool:
        """Check for suspicious patterns in request"""
        # Check URL path
        path = request.url.path.lower()
        
        # Check query parameters
        query_params = str(request.query_params).lower()
        
        # Combine for pattern matching
        content_to_check = path + " " + query_params
        
        for pattern in self.suspicious_patterns:
            if pattern.lower() in content_to_check:
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return True
        
        return False
    
    def _record_suspicious_activity(self, client_ip: str):
        """Record suspicious activity for monitoring"""
        logger.warning(f"Suspicious activity detected from IP {client_ip}")
        
        # If multiple suspicious activities, consider blocking
        if client_ip in self.request_counts:
            suspicious_count = self.request_counts[client_ip].get("suspicious", 0) + 1
            self.request_counts[client_ip]["suspicious"] = suspicious_count
            
            if suspicious_count >= 5:
                logger.warning(f"Blocking IP {client_ip} due to multiple suspicious activities")
                self.blocked_ips.add(client_ip)
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        # Content Security Policy
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none'"
        
        # Prevent XSS attacks
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # HTTP Strict Transport Security
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    def _log_request(self, request: Request, response: Response, duration: float):
        """Log request details for monitoring"""
        # Don't log health check endpoints to reduce noise
        if request.url.path == "/health" or request.url.path == "/system/health":
            return
        
        client_ip = self._get_client_ip(request)
        method = request.method
        path = request.url.path
        status_code = response.status_code
        
        logger.info(f"{client_ip} - {method} {path} - {status_code} - {duration:.3f}s")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    API Key middleware for JARVIS server
    Validates API keys for external service access
    """
    
    def __init__(self, app, api_keys=None):
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.protected_paths = ["/api/", "/v1/"]
        logger.info("API Key middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and validate API key if needed"""
        # Check if path is protected
        path = request.url.path
        if any(path.startswith(prefix) for prefix in self.protected_paths):
            # Get API key from header or query parameter
            api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            
            # Validate API key
            if not api_key or not self._validate_api_key(api_key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"}
                )
        
        # Process the request
        return await call_next(request)
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key in self.api_keys


class CombinedMiddleware:
    """
    Combined security middleware for JARVIS server
    Integrates security features from multiple middleware classes
    """
    
    def __init__(self, app, security_manager, api_keys=None):
        self.app = app
        self.security_manager = security_manager
        self.api_keys = api_keys or {}
        self.protected_paths = ["/api/", "/v1/"]
        
        # Initialize middleware components
        self.security_middleware = SecurityMiddleware(app, security_manager)
        self.api_key_middleware = APIKeyMiddleware(app, api_keys)
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process the request through combined middleware"""
        # Use security middleware for initial processing
        response = await self.security_middleware.dispatch(request, call_next)
        
        # Further processing with API key middleware if needed
        if response.status_code != 403 and response.status_code != 429:
            response = await self.api_key_middleware.dispatch(request, call_next)
        
        return response
