# JARVIS Server Security

Enhanced security components for the JARVIS server that integrate with the core security modules.

## Features

- **Comprehensive Authentication**: Secure user authentication with JWT tokens
- **Advanced Authorization**: Role-based access control with fine-grained permissions
- **Security Middleware**: Protection against common web vulnerabilities
- **Rate Limiting**: Prevents abuse through request rate limiting
- **API Key Authentication**: Secure API access for external services
- **Integration with Core Security**: Uses JARVIS core security components when available

## Components

### Security Manager

The `SecurityManager` class is the central component that manages all security-related functionality:

- User authentication and token generation
- Account lockout after failed attempts
- Permission checking based on user roles
- Integration with core security components

### Security Middleware

Two middleware components provide protection at the request level:

1. **SecurityMiddleware**: 
   - Adds security headers to responses
   - Implements rate limiting
   - Detects and blocks suspicious requests
   - Logs security events

2. **APIKeyMiddleware**:
   - Validates API keys for protected endpoints
   - Enables secure machine-to-machine communication

### Authentication Handler

The `AuthHandler` class provides FastAPI dependencies for route protection:

- `get_current_user`: Extracts and validates user from JWT token
- `get_current_active_user`: Ensures user account is active
- `get_admin_user`: Restricts access to admin users
- `check_permission`: Validates specific permissions

### Security Router

The security router provides API endpoints for security management:

- `/security/token`: Authentication endpoint
- `/security/users`: User management (admin only)
- `/security/status`: Security status information
- `/security/unlock/{username}`: Unlock locked accounts
- `/security/me`: Current user information

## Integration with Core Security

The server security components integrate with the core JARVIS security modules when available:

- Uses `AuthService` for authentication
- Uses `IdentityVerifier` for identity verification
- Uses `SecurityConfig` for configuration

When core components are not available, the server security components provide standalone functionality.

## Usage

### Protecting Routes

```python
from fastapi import Depends
from server.security.auth import create_auth_dependencies

# Create auth dependencies
auth_deps = create_auth_dependencies(security_manager)

@app.get("/protected")
async def protected_route(current_user = Depends(auth_deps["get_current_user"])):
    return {"message": "This is protected", "user": current_user}

@app.get("/admin")
async def admin_route(admin_user = Depends(auth_deps["get_admin_user"])):
    return {"message": "Admin only", "user": admin_user}
```

### Permission-Based Access

```python
@app.get("/write-data")
async def write_data(user = Depends(auth_deps["check_permission"]("write"))):
    return {"message": "You have write permission"}
```

## Environment Variables

- `JWT_SECRET`: Secret key for JWT token signing
- `API_KEYS`: Comma-separated list of valid API keys

## Security Best Practices

1. **Use HTTPS**: Always use HTTPS in production
2. **Rotate JWT Secret**: Regularly rotate the JWT secret key
3. **Limit Token Lifetime**: Use short-lived tokens (default: 12 hours)
4. **Implement Rate Limiting**: Prevents brute force attacks
5. **Validate All Input**: Prevent injection attacks
6. **Use Proper CORS Settings**: Restrict cross-origin requests in production
