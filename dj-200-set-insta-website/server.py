import datetime
import http.server
import json
import os
import socket
import socketserver
import urllib.parse
from http import HTTPStatus
from typing import Dict, Any, Union, List, Tuple, Optional
import mimetypes
import logging
from urllib.parse import parse_qs, urlparse, unquote
import cgi
import io
import shutil
import tempfile
import re
from email.parser import BytesParser
from email import policy
import ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 8000
PORT = DEFAULT_PORT  # For backward compatibility
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'data.json')

# Ensure the data directory exists
os.makedirs(SCRIPT_DIR, exist_ok=True)

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=self.base_path, **kwargs)

    def is_authenticated(self) -> bool:
        """Check if the user is authenticated via session cookie."""
        if 'Cookie' not in self.headers:
            return False
        
        # In a real app, verify the session token here
        # For now, we'll just check if there's any cookie
        return 'session=' in self.headers['Cookie']

    def do_GET(self):
        # Parse the URL
        parsed_path = urlparse(self.path)
        path = unquote(parsed_path.path)
        
        # Define public routes that don't require authentication
        public_routes = {'/login', '/login.html', '/register', '/register.html', '/style.css', '/script.js'}
        
        # Check if the path is a file that exists
        file_path = os.path.normpath(os.path.join(self.base_path, path.lstrip('/')))
        is_file = os.path.isfile(file_path)
        
        # Redirect root to login page
        if path == '/':
            self.send_response(302)
            self.send_header('Location', '/login.html')
            self.end_headers()
            return
            
        # Handle login and register pages
        if path in {'/login', '/login.html'}:
            self.path = '/login.html'
        elif path in {'/register', '/register.html'}:
            self.path = '/register.html'
        # Protect index and other pages that require authentication
        elif path in {'/index', '/index.html'} or (is_file and path not in public_routes):
            if not self.is_authenticated():
                self.send_response(302)
                self.send_header('Location', '/login.html?redirect=' + path)
                self.end_headers()
                return
            self.path = path if is_file else '/index.html'
        
        # Check if the file exists and is allowed
        try:
            # Normalize the path to prevent directory traversal
            file_path = os.path.normpath(os.path.join(self.base_path, self.path.lstrip('/')))
            
            # Ensure the path is within the base directory
            if not file_path.startswith(self.base_path):
                self.send_error(403, "Forbidden")
                return
                
            # Check if it's a file and exists
            if os.path.isfile(file_path):
                # Get the MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type:
                    self.send_response(200)
                    self.send_header('Content-type', mime_type)
                    self.end_headers()
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    # Default to octet-stream if MIME type is unknown
                    self.send_response(200)
                    self.send_header('Content-type', 'application/octet-stream')
                    self.end_headers()
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'File not found')
                
        except Exception as e:
            self.send_error(500, f"Server Error: {str(e)}")

    def _parse_post_data(self, content_type: str, post_data: bytes) -> Dict[str, str]:
        """Parse POST data based on content type
        
        Args:
            content_type: The Content-Type header from the request
            post_data: The raw POST data as bytes
            
        Returns:
            dict: Parsed data with consistent value types
        """
        def ensure_str(value: Union[str, List[Any], bytes, None]) -> str:
            """Ensure the value is a string, handling multiple types"""
            if value is None:
                return ''
            if isinstance(value, bytes):
                try:
                    return value.decode('utf-8', errors='ignore').strip()
                except UnicodeDecodeError:
                    return ''
            if isinstance(value, list):
                if not value:
                    return ''
                return ensure_str(value[0])
            return str(value).strip()
            
        try:
            # Handle JSON data
            if content_type == 'application/json':
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    return {k: ensure_str(v) for k, v in data.items()}
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON data received")
                    return {}
            
            # Handle URL-encoded form data
            elif content_type == 'application/x-www-form-urlencoded':
                try:
                    from urllib.parse import parse_qs, unquote_plus
                    data = parse_qs(post_data.decode('utf-8'), keep_blank_values=True)
                    return {k: ensure_str(v[0] if isinstance(v, list) and len(v) == 1 else v) 
                           for k, v in data.items()}
                except Exception as e:
                    logger.error(f"Error parsing form data: {str(e)}")
                    return {}
            
            # Handle multipart form data
            elif content_type.startswith('multipart/form-data'):
                try:
                    # Use a simpler approach for form data without files
                    # Just parse it as URL-encoded data
                    from urllib.parse import parse_qs, unquote_plus
                    data = {}
                    for item in post_data.split(b'&'):
                        if b'=' in item:
                            key, value = item.split(b'=', 1)
                            data[key.decode('latin1')] = unquote_plus(value.decode('latin1'))
                    return data
                except Exception as e:
                    logger.error(f"Error parsing multipart form data: {str(e)}")
                    return {}
            
            logger.warning(f"Unsupported content type: {content_type}")
            return {}
            
        except Exception as e:
            logger.error(f"Unexpected error in _parse_post_data: {str(e)}", exc_info=True)
            return {}
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        content_type = self.headers.get('Content-Type', '').split(';')[0].strip()
        
        if content_length == 0:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'No data received')
            return
            
        try:
            post_data = self.rfile.read(content_length)
            
            # Parse the data based on content type
            try:
                data = self._parse_post_data(content_type, post_data)
                if not data:
                    raise ValueError('Unsupported content type')
                    
                # Get and clean form data
                username = str(data.get('username', '')).strip()
                password = str(data.get('password', '')).strip()
                email = str(data.get('email', '')).strip()
                
                if not username or not password:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'error',
                        'message': 'Gebruikersnaam en wachtwoord zijn verplicht'
                    }).encode('utf-8'))
                    return
                    
                if self.path == '/login':
                    self._handle_login(username, password)
                elif self.path == '/register':
                    self._handle_register(username, password, email)
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Not found')
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'message': 'Ongeldige gegevens ontvangen'
                }).encode('utf-8'))
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'message': f'Serverfout: {str(e)}'
            }).encode('utf-8'))
    
    def _send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        """Send a JSON response with the given status code and data
        
        Args:
            status_code: HTTP status code
            data: Dictionary to be sent as JSON
        """
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error sending JSON response: {str(e)}")
    
    def _handle_login(self, username: str, password: str) -> None:
        """Handle user login
        
        Args:
            username: The username to login
            password: The password to verify
        """
        try:
            # Input validation
            if not username or not password:
                self._send_json_response(400, {
                    'status': 'error',
                    'message': 'Vul alle velden in'
                })
                return
            
            # Check if data file exists
            if not os.path.exists(DATA_FILE):
                logger.warning(f"Data file not found at {DATA_FILE}")
                self._send_json_response(401, {
                    'status': 'error',
                    'message': 'Gebruikersnaam of wachtwoord is onjuist'
                })
                return
            
            # Load user data
            try:
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                
                # Ensure users is a dictionary
                if not isinstance(users, dict):
                    logger.error(f"Invalid data format in {DATA_FILE}")
                    users = {}
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading user data from {DATA_FILE}: {str(e)}")
                self._send_json_response(500, {
                    'status': 'error',
                    'message': 'Interne serverfout bij het inloggen'
                })
                return
            
            # Debug: Log available usernames
            logger.info(f"Available usernames: {list(users.keys())}")
            logger.info(f"Attempting login for user: {username}")
            
            # Verify credentials (case-sensitive username, case-sensitive password)
            if username in users and users[username].get('password') == password:
                # Update last login time
                users[username]['last_login'] = datetime.datetime.now().isoformat()
                try:
                    with open(DATA_FILE, 'w', encoding='utf-8') as f:
                        json.dump(users, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Error updating last login time: {str(e)}")
                
                # Log successful login
                logger.info(f"User logged in: {username}")
                
                # Create a simple session token (in production, use a proper session management system)
                session_token = f"session_{username}_{int(datetime.datetime.now().timestamp())}"
                
                # Prepare success response with session cookie
                response_data = {
                    'status': 'success', 
                    'message': 'Inloggen gelukt!',
                    'redirect': '/index.html'
                }
                
                # Send response with Set-Cookie header
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Set-Cookie', f'session={session_token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
            else:
                # Log failed login attempt
                logger.warning(f"Failed login attempt for user: {username}")
                self._send_json_response(401, {
                    'status': 'error', 
                    'message': 'Gebruikersnaam of wachtwoord is onjuist'
                })
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}", exc_info=True)
            self._send_json_response(500, {
                'status': 'error',
                'message': 'Er is een fout opgetreden bij het inloggen. Probeer het later opnieuw.'
            })
    
    def _handle_register(self, username: str, password: str, email: str) -> None:
        """Handle user registration
        
        Args:
            username: The username to register
            password: The password for the new account
            email: The email address for the new account
        """
        try:
            # Input validation
            if not username or not password or not email:
                self._send_json_response(400, {
                    'status': 'error',
                    'message': 'Vul alle velden in'
                })
                return
                
            if len(password) < 6:
                self._send_json_response(400, {
                    'status': 'error',
                    'message': 'Wachtwoord moet minimaal 6 tekens lang zijn'
                })
                return
                
            if '@' not in email or '.' not in email.split('@')[-1]:
                self._send_json_response(400, {
                    'status': 'error',
                    'message': 'Voer een geldig e-mailadres in'
                })
                return
            
            # Load existing users
            users = {}
            if os.path.exists(DATA_FILE):
                try:
                    with open(DATA_FILE, 'r', encoding='utf-8') as f:
                        users = json.load(f)
                        if not isinstance(users, dict):
                            users = {}
                except (json.JSONDecodeError, FileNotFoundError):
                    users = {}
            
            # Check if username or email already exists
            if username.lower() in {u.lower() for u in users.keys()}:
                self._send_json_response(400, {
                    'status': 'error',
                    'message': 'Gebruikersnaam is al in gebruik'
                })
                return
                
            if any(user_data.get('email', '').lower() == email.lower() 
                  for user_data in users.values()):
                self._send_json_response(400, {
                    'status': 'error',
                    'message': 'E-mailadres is al geregistreerd'
                })
                return
            
            # Add new user
            users[username] = {
                'password': password,  # In a real app, hash the password!
                'email': email,
                'created_at': datetime.datetime.now().isoformat(),
                'last_login': None
            }
            
            # Save to file with atomic write
            temp_file = f"{DATA_FILE}.tmp"
            try:
                # First write to temp file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(users, f, indent=2, ensure_ascii=False)
                
                # Then atomically replace the old file
                if os.path.exists(DATA_FILE):
                    os.replace(temp_file, DATA_FILE)
                else:
                    # If file doesn't exist, create parent directories if needed
                    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
                    os.rename(temp_file, DATA_FILE)
                
                logger.info(f"Successfully saved user data to {DATA_FILE}")
                
            except Exception as e:
                logger.error(f"Error saving user data: {str(e)}")
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                raise
            
            # Log successful registration
            logger.info(f"New user registered: {username} ({email})")
            
            # Send success response
            self._send_json_response(200, {
                'status': 'success',
                'message': 'Registratie succesvol! Je wordt nu doorgestuurd...',
                'redirect': '/login.html'
            })
            
        except Exception as e:
            logger.error(f"Registration error: {str(e)}", exc_info=True)
            self._send_json_response(500, {
                'status': 'error',
                'message': 'Er is een fout opgetreden bij de registratie. Probeer het later opnieuw.'
            })

def run_server(port=8000, max_retries=5):
    """Start the HTTP server with port handling for Windows.
    
    Args:
        port: Starting port number
        max_retries: Maximum number of ports to try
    """
    # Initialize data file if it doesn't exist
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
            logger.info(f"Created new data file at {DATA_FILE}")
        except Exception as e:
            logger.error(f"Failed to create data file: {str(e)}")
            raise
    
    # Add common MIME types
    mimetypes.add_type('text/css', '.css')
    mimetypes.add_type('application/javascript', '.js')
    
    # Try to start the server, incrementing the port if needed
    for attempt in range(max_retries):
        current_port = port + attempt
        try:
            # Create server with allow_reuse_address=True to prevent 'Address already in use' errors
            httpd = socketserver.TCPServer(("", current_port), MyHandler, bind_and_activate=False)
            httpd.allow_reuse_address = True
            httpd.server_bind()
            httpd.server_activate()
            
            print(f"\nServing at http://localhost:{current_port}")
            print("Press Ctrl+C to stop the server")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down the server...")
                httpd.shutdown()
                httpd.server_close()
                return
            except Exception as e:
                logger.error(f"Server error: {str(e)}")
                break
                
        except OSError as e:
            if "Address already in use" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Port {current_port} is in use, trying next port...")
                continue
            logger.error(f"Failed to start server on port {current_port}: {str(e)}")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            break
    else:
        logger.error(f"Failed to start server after {max_retries} attempts")

if __name__ == "__main__":
    try:
        print(f"Starting server on port {DEFAULT_PORT}...")
        run_server(port=DEFAULT_PORT)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        print(f"Error: {str(e)}")
        print("Press Enter to exit...")
        input()
