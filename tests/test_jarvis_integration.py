"""
JARVIS Integration Server - Simplified
======================================
A lightweight WebSocket and HTTP server for JARVIS AI model integration.
"""
import os
import time
import socket
import logging
import asyncio
import threading
import json
import signal
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Set, Optional, Protocol, Union, Awaitable
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Import WebSocket related modules
try:
    import websockets
    # We'll use the main websockets module directly for better compatibility
    from websockets.typing import Data
except ImportError as e:
    print("Error: websockets module not found.")
    print("Please install it with: pip install 'websockets>=10.0'")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis_server.log')
    ]
)

logger = logging.getLogger(__name__)

class SimpleAIModel:
    """A simple AI model that can be used when the main model is not available"""
    def __init__(self):
        self.responses = {
            "hallo": "Hoi! Hoe kan ik je helpen?",
            "hey": "Hey! Wat kan ik voor je doen?",
            "hi": "Hi daar! Waar kan ik je mee helpen?",
            "help": "Ik ben een eenvoudige AI assistent. Ik kan basic vragen beantwoorden en gesprekken voeren.",
            "wie ben je": "Ik ben J.A.R.V.I.S., je AI assistent. Momenteel draai ik in fallback modus.",
            "wat kun je": "In fallback modus kan ik eenvoudige gesprekken voeren en basis vragen beantwoorden.",
            "hoe gaat het": "Met mij gaat het goed! Ik draai in fallback modus maar ben nog steeds beschikbaar om te helpen.",
            "dag": "Tot ziens! Het was fijn om je te helpen.",
            "doei": "Doei! Kom gerust terug als je meer vragen hebt.",
        }
        self.default_response = "Ik begrijp je bericht. Helaas draai ik momenteel in fallback modus met beperkte functionaliteit."

    def generate_response(self, message: str) -> str:
        """Generate a response based on the input message"""
        message = message.lower().strip()
        
        # Check for exact matches
        if message in self.responses:
            return self.responses[message]
            
        # Check for partial matches
        for key in self.responses:
            if key in message:
                return self.responses[key]
                
        return self.default_response

class CustomHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler that serves static files"""
    def __init__(self, *args, **kwargs):
        self.directory = str(Path(__file__).parent)
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"HTTP: {format % args}")

class JarvisServer:
    def __init__(self, host: str = '127.0.0.1', http_port: int = 8080, ws_port: int = 8765) -> None:
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.clients: Set[Any] = set()  # Using Any for WebSocketServerProtocol compatibility
        self.model = SimpleAIModel()
        self.stats = {
            'start_time': datetime.now(),
            'connections': 0,
            'messages': 0
        }
        self._server = None
        self._http_server = None
    
    async def start(self) -> None:
        """Start the WebSocket and HTTP servers"""
        # Start WebSocket server in the background
        ws_task = asyncio.create_task(self.start_websocket_server())
        
        # Start HTTP server in the background
        http_task = asyncio.create_task(self.run_http_server())
        
        # Wait for both servers to be ready
        await asyncio.gather(ws_task, http_task, return_exceptions=True)
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    
    def run_http_server_sync(self) -> None:
        """Run the HTTP server in a synchronous way"""
        class AsyncHTTPRequestHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                try:
                    return SimpleHTTPRequestHandler.do_GET(self)
                except Exception as e:
                    logger.error(f"Error handling GET request: {e}")
                    self.send_error(500, str(e))
            
            def log_message(self, format, *args):
                logger.info(f"HTTP {self.address_string()} - {format % args}")
        
        self._http_server = HTTPServer((self.host, self.http_port), AsyncHTTPRequestHandler)
        logger.info(f"HTTP server started on http://{self.host}:{self.http_port}")
        self._http_server.serve_forever()
    
    async def run_http_server(self) -> None:
        """Run the HTTP server in a separate thread"""
        # Run the HTTP server in a separate thread
        def start_http_server():
            try:
                self.run_http_server_sync()
            except Exception as e:
                logger.error(f"HTTP server error: {e}")
        
        http_thread = threading.Thread(target=start_http_server, daemon=True)
        http_thread.start()
        
        # Give the server a moment to start
        await asyncio.sleep(0.1)
    
    async def stop(self) -> None:
        """Stop the servers"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._http_server:
            self._http_server.shutdown()
        logger.info("Servers stopped")
    
    async def handle_websocket(self, websocket, path: str = '') -> None:
        """Handle WebSocket connections
        
        Args:
            websocket: The WebSocket connection
            path: Optional path (for compatibility with older websockets versions)
        """
        self.clients.add(websocket)
        self.stats['connections'] += 1
        client_ip = 'unknown'
        
        try:
            if hasattr(websocket, 'remote_address') and websocket.remote_address:
                client_ip = websocket.remote_address[0] if isinstance(websocket.remote_address, (list, tuple)) else str(websocket.remote_address)
                
            logger.info(f"New WebSocket connection from {client_ip}. Total clients: {len(self.clients)}")
            
            async for message in websocket:
                self.stats['messages'] += 1
                logger.debug(f"Received message from {client_ip}: {message}")
                
                try:
                    data = json.loads(message)
                    response = await self._process_message(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received from {client_ip}")
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"Error processing message from {client_ip}: {e}")
                    logger.error(traceback.format_exc())
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': f'Error processing message: {str(e)}'
                    }))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed by {client_ip}")
        except Exception as e:
            logger.error(f"WebSocket error with {client_ip}: {e}")
            logger.error(traceback.format_exc())
        finally:
            try:
                self.clients.remove(websocket)
                logger.info(f"Client {client_ip} disconnected. Remaining clients: {len(self.clients)}")
            except (KeyError, ValueError):
                pass
    
    async def _process_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message"""
        try:
            message_type = data.get('type', 'chat')
            logger.debug(f"Processing message of type: {message_type}")
            
            if message_type == 'chat':
                user_message = data.get('message', '')
                logger.debug(f"Processing chat message: {user_message}")
                
                response = self.model.generate_response(user_message)
                logger.debug(f"Generated model response: {response}")
                
                return {
                    'type': 'response',
                    'message': response,
                    'timestamp': datetime.now().isoformat()
                }
            
            elif message_type == 'status':
                return {
                    'type': 'status',
                    'server': 'running',
                    'model': 'fallback_active',
                    'clients': len(self.clients),
                    'uptime': str(datetime.now() - self.stats['start_time']),
                    'stats': self.stats
                }
            
            else:
                logger.warning(f"Unknown message type received: {message_type}")
                return {
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }
        except Exception as e:
            logger.error(f"Error in _process_message: {e}\n{traceback.format_exc()}")
            return {
                'type': 'error',
                'message': 'Error processing message',
                'details': str(e)
            }
    
    def start_http_server(self):
        """Start HTTP server for serving static files"""
        try:
            httpd = HTTPServer((self.host, self.http_port), CustomHandler)
            logger.info(f"HTTP server starting on http://{self.host}:{self.http_port}")
            httpd.serve_forever()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
    
    def is_port_available(self, port: int) -> bool:
        """Check if a port is available on the given host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((self.host, port))
                # Get the port number that was actually bound
                _, bound_port = s.getsockname()
                # Close the socket immediately
                s.close()
                # On Windows, we need to wait a moment for the port to be released
                if os.name == 'nt':
                    time.sleep(0.1)
                return True
            except OSError as e:
                logger.debug(f"Port {port} is not available: {e}")
                return False
            except Exception as e:
                logger.error(f"Error checking port {port}: {e}")
                return False
    
    async def start_websocket_server(self, max_attempts: int = 10) -> bool:
        """Start WebSocket server with automatic port selection"""
        original_port = self.ws_port
        server = None
        
        for attempt in range(max_attempts):
            try:
                if not self.is_port_available(self.ws_port):
                    logger.warning(f"Port {self.ws_port} is in use, trying next port...")
                    self.ws_port += 1
                    continue
                
                # Create server with minimal options for maximum compatibility
                server = await websockets.serve(
                    self.handle_websocket,
                    self.host,
                    self.ws_port,
                    # Skip reuse_port as it's not supported on Windows
                    ping_interval=30,
                    ping_timeout=30,
                    close_timeout=1,  # Faster cleanup
                    # Disable compression for better performance
                    compression=None,
                    # Use smaller buffer sizes for better behavior on Windows
                    read_limit=2**16,
                    write_limit=2**16
                )
                
                if self.ws_port != original_port:
                    logger.warning(f"Port {original_port} was in use, using port {self.ws_port} instead")
                
                logger.info(f"WebSocket server started on ws://{self.host}:{self.ws_port}")
                logger.info(f"Server ready! Open http://{self.host}:{self.http_port} in your browser")
                
                # Keep the server running until it's closed
                await server.wait_closed()
                return True
                
            except OSError as e:
                if server is not None:
                    server.close()
                    await server.wait_closed()
                
                if "Address already in use" in str(e) or "address is already in use" in str(e).lower():
                    self.ws_port += 1
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(0.1)  # Small delay before retry
                        continue
                
                logger.error(f"Failed to start WebSocket server after {attempt + 1} attempts: {e}")
                logger.debug(traceback.format_exc())
                return False
            except Exception as e:
                if server is not None:
                    server.close()
                    await server.wait_closed()
                logger.error(f"Unexpected error starting WebSocket server: {e}")
                logger.debug(traceback.format_exc())
                return False
        
        logger.error(f"Could not find an available port in range {original_port}-{self.ws_port}")
        return False

def main():
    """Main entry point"""
    server = JarvisServer()
    
    # Start HTTP server in a separate thread
    http_thread = threading.Thread(target=server.start_http_server, daemon=True)
    http_thread.start()
    
    # Start WebSocket server in the main thread
    try:
        asyncio.get_event_loop().run_until_complete(server.start_websocket_server())
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server stopped")

if __name__ == "__main__":
    main()