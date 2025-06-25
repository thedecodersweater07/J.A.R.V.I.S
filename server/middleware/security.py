from fastapi import Request, Response
from starlette.types import ASGIApp, Receive, Scope, Send

async def add_security_headers(request: Request, call_next) -> Response:
    # Skip security headers for WebSocket upgrade requests
    if request.scope.get("type") == "websocket":
        return await call_next(request)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains; preload'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' ws: wss:;"
    return response
