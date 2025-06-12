"""Integration test server for JARVIS"""

import asyncio
import json
import logging
import websockets
from websockets.legacy.server import WebSocketServerProtocol, serve as websocket_serve
import signal
import sys
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Awaitable, Optional, cast
import threading
import os

# Add the project root to the Python path
import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import JARVIS model
from models.jarvis import JarvisLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_free_port(start_port: int, max_attempts: int = 10) -> int:
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('127.0.0.1', port))
                return port
        except OSError:
            logger.warning(f"Port {port} is already in use")
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")

def check_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('127.0.0.1', port))
            return False
        except OSError:
            return True

class JarvisServer:
    def __init__(self, host: str = '127.0.0.1', http_port: Optional[int] = None, ws_port: Optional[int] = None):
        self.host = host
        # Find free ports if none specified
        self.http_port = http_port or find_free_port(18080)
        self.ws_port = ws_port or find_free_port(self.http_port + 1)
        
        self.clients: set[WebSocketServerProtocol] = set()
        self.is_running = False
        self.is_connected = False
        self.http_server: Optional[HTTPServer] = None
        self.http_thread: Optional[threading.Thread] = None
        self.ws_server = None
        
        # Initialize JARVIS model
        self.jarvis = JarvisLanguageModel()
        
        # Update the WebSocket port in the JavaScript file
        self.update_ws_port_in_js()
        
    def update_ws_port_in_js(self) -> None:
        """Update the WebSocket port in the JavaScript file"""
        try:
            js_file = Path(__file__).parent / 'script.js'
            if not js_file.exists():
                logger.warning(f"Could not find {js_file}")
                return
            
            # Read with UTF-8 encoding
            content = js_file.read_text(encoding='utf-8')
            
            # Find the WS_PORT line and update it
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if 'const WS_PORT' in line:
                    lines[i] = f"const WS_PORT = {self.ws_port};"
                    break
            
            # Write back with UTF-8 encoding
            js_file.write_text('\n'.join(lines), encoding='utf-8')
            logger.info(f"Updated WebSocket port in script.js to {self.ws_port}")
        except Exception as e:
            logger.error(f"Failed to update WebSocket port in script.js: {e}")

    def start_http_server(self) -> None:
        """Start the HTTP server in a separate thread"""
        try:
            # Change to the directory containing the test files
            os.chdir(Path(__file__).parent)
            
            handler = SimpleHTTPRequestHandler
            self.http_server = HTTPServer((self.host, self.http_port), handler)
            logger.info(f"HTTP server started at http://{self.host}:{self.http_port}")
            self.http_server.serve_forever()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
            self.is_running = False
    
    async def connect(self) -> None:
        """Connect to the server and start all services"""
        if self.is_connected:
            logger.warning("Server is already connected")
            return

        try:
            # Start HTTP server in a separate thread
            self.http_thread = threading.Thread(target=self.start_http_server)
            self.http_thread.daemon = True
            self.http_thread.start()
            
            # Start WebSocket server
            self.ws_server = await websocket_serve(
                self.handle_websocket,
                self.host,
                self.ws_port,
                ping_interval=20,
                ping_timeout=20
            )
            
            self.is_running = True
            self.is_connected = True
            
            logger.info("Successfully connected to all services")
            logger.info(f"HTTP server running at http://{self.host}:{self.http_port}")
            logger.info(f"WebSocket server running at ws://{self.host}:{self.ws_port}")
            
            # Print server info
            self.print_server_info()
            
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Disconnect from all services and cleanup"""
        if not self.is_connected:
            logger.warning("Server is not connected")
            return
            
        logger.info("Disconnecting from all services...")
        
        # Close all client connections
        if self.clients:
            logger.info(f"Closing {len(self.clients)} client connections...")
            await asyncio.gather(*[ws.close() for ws in self.clients])
        self.clients.clear()
        
        # Stop WebSocket server
        if self.ws_server:
            logger.info("Stopping WebSocket server...")
            self.ws_server.close()
            await self.ws_server.wait_closed()
            self.ws_server = None
        
        # Stop HTTP server
        if self.http_server:
            logger.info("Stopping HTTP server...")
            self.http_server.shutdown()
            self.http_server.server_close()
            self.http_server = None
        
        # Stop HTTP thread
        if self.http_thread and self.http_thread.is_alive():
            logger.info("Waiting for HTTP thread to stop...")
            self.http_thread.join(timeout=5)
            self.http_thread = None

        self.is_running = False
        self.is_connected = False
        logger.info("Successfully disconnected from all services")

    async def handle_websocket(self, websocket: WebSocketServerProtocol) -> None:
        """Handle WebSocket connections"""
        try:
            # Register client
            client_id = id(websocket)
            self.clients.add(websocket)
            logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
            
            # Send initial status
            await websocket.send(json.dumps({
                'type': 'status',
                'status': 'Connected',
                'client_id': client_id
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'message':
                        user_message = data.get('message', '')
                        
                        # Get response from JARVIS
                        response = self.jarvis.generate_response(user_message)
                        
                        # Send response back to client
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'message': response,
                            'client_id': client_id
                        }))
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Error processing message from client {client_id}: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e),
                        'client_id': client_id
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        finally:
            # Unregister client
            self.clients.remove(websocket)
            logger.info(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")

    async def start(self) -> None:
        """Start the server"""
        await self.connect()
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the server"""
        self.is_running = False
        await self.disconnect()

    def print_server_info(self) -> None:
        """Print server information in a nice box"""
        info = [
            "J.A.R.V.I.S. AI Server",
            f"• HTTP:      http://{self.host}:{self.http_port}",
            f"• WebSocket: ws://{self.host}:{self.ws_port}",
            f"• Status:    {'Connected' if self.is_connected else 'Disconnected'}"
        ]
        
        # Calculate box width
        width = max(len(line) for line in info) + 4
        
        # Print top border
        print("╔" + "═" * width + "╗")
        
        # Print title
        print("║" + info[0].center(width) + "║")
        
        # Print separator
        print("╠" + "═" * width + "╣")
        
        # Print info lines
        for line in info[1:]:
            print("║" + line.ljust(width) + "║")
        
        # Print separator
        print("╠" + "═" * width + "╣")
        
        # Print instructions
        print("║" + "Gebruik Ctrl+C om te stoppen".center(width) + "║")
        
        # Print bottom border
        print("╚" + "═" * width + "╝")

def handle_shutdown(signum: int, frame: Any) -> None:
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Create and start server
    server = JarvisServer()
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Ensure proper cleanup
        if server.is_connected:
            asyncio.run(server.disconnect())
