# JARVIS Server

The JARVIS Server provides a RESTful API interface to interact with the JARVIS AI Assistant system.

## Features

- **Authentication**: Secure API access with JWT token-based authentication
- **AI Processing**: Process text queries through NLP, ML, and LLM components
- **System Management**: Monitor and control the JARVIS system
- **Modular Architecture**: Well-organized API structure with separate routers

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):

```bash
# Security
export JWT_SECRET=your-secure-jwt-secret-key

# AI Components
export NLP_MODEL=nl_core_news_sm
export MODEL_PATH=/path/to/models
```

## Usage

### Starting the Server

```bash
# Using run_server.py (recommended)
python run_server.py --host 127.0.0.1 --port 8000 --debug

# Using launcher.py
python launcher.py --host 127.0.0.1 --port 8000 --debug

# Directly using app.py
python app.py
```

### API Endpoints

#### Authentication

- `POST /auth/token`: Get access token (login)
- `POST /auth/register`: Register new user

#### AI Processing

- `POST /ai/query`: Process AI query
- `GET /ai/status`: Get AI components status

#### System

- `GET /system/health`: Check system health
- `GET /system/status`: Get detailed system status
- `POST /system/restart`: Restart the system (admin only)
- `GET /system/logs`: Get system logs (admin only)

## Architecture

The server is built with FastAPI and follows a modular architecture:

- `app.py`: Main application entry point
- `launcher.py`: Server launcher with command-line options
- `run_server.py`: Production server runner
- `api/`: API routers and endpoints
  - `auth.py`: Authentication routes
  - `ai.py`: AI processing routes
  - `system.py`: System management routes
  - `dependencies.py`: Shared dependencies

## Development

To extend the API:

1. Create a new router in the `api/` directory
2. Register your router in `app.py`
3. Add appropriate authentication and error handling

## Error Handling

The server includes comprehensive error handling:

- Graceful fallback for missing AI components
- Proper initialization of NLPProcessor and ModelManager with correct parameters
- Detailed logging for troubleshooting
- Appropriate HTTP status codes for API errors

## Security

- JWT token-based authentication
- Password hashing with bcrypt
- Role-based access control
- Protection against brute force attacks
