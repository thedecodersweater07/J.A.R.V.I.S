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
from typing import Dict, Any, Set, Optional, Protocol, Union, Awaitable, Type
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Try to import the actual model
try:
    from models.jarvis import JarvisModel, JarvisLanguageModel, ModelLoadError
except ImportError:
    JarvisModel = None
    JarvisLanguageModel = None
    ModelLoadError = RuntimeError

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
    def __init__(self, host: str = '127.0.0.1', http_port: int = 5000, ws_port: int = 5001) -> None:
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.clients: Set[Any] = set()  # Using Any for WebSocketServerProtocol compatibility
        
        # Try to initialize the actual model, fall back to SimpleAIModel if it fails
        self.model = None  # type: Optional[Union[SimpleAIModel, Any]]
        self.model_type = 'fallback'
        try:
            if JarvisLanguageModel is not None:
                self.model = JarvisLanguageModel()  # type: ignore
                self.model_type = 'jarvis_language_model'
                logger.info("Successfully loaded JARVIS Language Model")
            else:
                raise ImportError("JarvisLanguageModel not available")
        except Exception as e:
            logger.warning(f"Could not load JARVIS model, using fallback: {e}")
            self.model = SimpleAIModel()
            self.model_type = 'fallback'
        
        self.stats = {
            'start_time': datetime.now(),
            'connections': 0,
            'messages': 0,
            'model_type': self.model_type
        }
        self._server = None
        self._http_server = None
    
    async def start(self) -> None:
        """Start the WebSocket and HTTP servers"""
        try:
            # Start WebSocket server in a separate thread
            ws_thread = threading.Thread(target=self.start_websocket_server, daemon=True)
            ws_thread.start()
            
            # Start HTTP server in the current thread
            http_thread = threading.Thread(target=self.run_http_server, daemon=True)
            http_thread.start()
            
            # Keep the main thread alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in server: {e}")
            logger.error(traceback.format_exc())
            raise
    
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
            wait_closed = getattr(self._server, 'wait_closed', None)
            if callable(wait_closed):
                try:
                    # Call the method and check if it's a coroutine
                    result = wait_closed()
                    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                        await result
                except Exception as e:
                    logger.warning(f"Error while waiting for server to close: {e}")
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
            # Get client IP address
            if hasattr(websocket, 'remote_address') and websocket.remote_address:
                client_ip = websocket.remote_address[0] if isinstance(websocket.remote_address, (list, tuple)) else str(websocket.remote_address)
            
            logger.info(f"New WebSocket connection from {client_ip}. Total clients: {len(self.clients)}")
            
            # Send initial connection message
            await websocket.send(json.dumps({
                'type': 'connection',
                'status': 'connected',
                'message': 'Successfully connected to JARVIS server',
                'model': self.model_type,
                'timestamp': datetime.now().isoformat()
            }))
            
            # Main message loop
            while True:
                try:
                    # Wait for a message with a timeout
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=300)  # 5 minute timeout
                    except asyncio.TimeoutError:
                        # Send a ping to check if client is still connected
                        try:
                            await websocket.ping()
                            continue
                        except:
                            logger.warning(f"Connection timeout for {client_ip}")
                            break
                    
                    # Process the received message
                    try:
                        data = json.loads(message)
                        response = await self._process_message(data)
                        await websocket.send(json.dumps(response))
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON format',
                            'timestamp': datetime.now().isoformat()
                        }))
                    except Exception as e:
                        logger.error(f"Error processing message from {client_ip}: {e}")
                        logger.error(traceback.format_exc())
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f'Error processing message: {str(e)}',
                            'timestamp': datetime.now().isoformat()
                        }))
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"WebSocket connection closed by {client_ip}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in WebSocket handler for {client_ip}: {e}")
                    logger.error(traceback.format_exc())
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_ip} disconnected")
        except Exception as e:
            logger.error(f"WebSocket error with {client_ip}: {e}")
            logger.error(traceback.format_exc())
        finally:
            try:
                await websocket.close()
            except:
                pass
            self.clients.discard(websocket)
            logger.info(f"Client {client_ip} disconnected. Remaining clients: {len(self.clients)}")
    
    async def _process_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message"""
        message_type = data.get('type', 'chat')
        
        try:
            if message_type == 'chat':
                user_message = data.get('message', '')
                if not user_message:
                    return {
                        'type': 'error',
                        'message': 'No message provided'
                    }
                
                # Log the message
                logger.info(f"Received message: {user_message}")
                self.stats['messages'] += 1
                
                # Get response from model
                if self.model is None:
                    return {
                        'type': 'error',
                        'message': 'No model available',
                        'timestamp': datetime.now().isoformat()
                    }
                
                try:
                    if hasattr(self.model, 'generate_response'):
                        if asyncio.iscoroutinefunction(self.model.generate_response):
                            response = await self.model.generate_response(user_message)
                        else:
                            response = self.model.generate_response(user_message)  # type: ignore
                    else:
                        response = "Error: Model does not support generate_response"
                    
                    return {
                        'type': 'response',
                        'message': response,
                        'timestamp': datetime.now().isoformat(),
                        'model': self.model_type
                    }
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    logger.error(traceback.format_exc())
                    return {
                        'type': 'error',
                        'message': f'Error generating response: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                
            elif message_type == 'status':
                return {
                    'type': 'status',
                    'status': 'ok',
                    'model': self.model_type,
                    'stats': {
                        'connections': len(self.clients),
                        'messages': self.stats['messages'],
                        'uptime': str(datetime.now() - self.stats['start_time'])
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
            else:
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
    
    def _run_websocket_server(self):
        """Run the WebSocket server in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create server
            start_server = websockets.serve(
                self.handle_websocket,
                self.host,
                self.ws_port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=1
            )
            
            # Start the server
            self._server = loop.run_until_complete(start_server)
            
            # Update the port in case it changed
            for sock in self._server.sockets:
                self.ws_port = sock.getsockname()[1]
                break
                
            logger.info(f"WebSocket server started on ws://{self.host}:{self.ws_port}")
            
            # Run the event loop
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Close all client connections
            for client in list(self.clients):
                try:
                    loop.run_until_complete(client.close())
                except:
                    pass
            
            # Close the server
            if hasattr(self, '_server') and self._server:
                self._server.close()
                loop.run_until_complete(self._server.wait_closed())
            
            # Close the event loop
            if not loop.is_closed():
                loop.close()
    
    def start_websocket_server(self) -> bool:
        """Start WebSocket server in a separate thread"""
        try:
            # Try to find an available port
            for port_offset in range(20):  # Try up to 20 ports
                current_port = self.ws_port + port_offset
                if self.is_port_available(current_port):
                    self.ws_port = current_port
                    break
            else:
                logger.error("Could not find an available port")
                return False
            
            # Start the WebSocket server in a daemon thread
            self._ws_thread = threading.Thread(
                target=self._run_websocket_server,
                daemon=True,
                name="WebSocketServerThread"
            )
            self._ws_thread.start()
            
            # Give it a moment to start
            time.sleep(0.5)
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            logger.error(traceback.format_exc())
            return False

def main():
    """Main entry point"""
    try:
        # Create and start server
        server = JarvisServer()
        
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=server.run_http_server, daemon=True)
        http_thread.start()
        
        # Start WebSocket server
        if not server.start_websocket_server():
            logger.error("Failed to start WebSocket server")
            return False
        
        logger.info(f"Server ready!")
        logger.info(f"WebSocket: ws://{server.host}:{server.ws_port}")
        logger.info(f"HTTP: http://{server.host}:{server.http_port}")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()