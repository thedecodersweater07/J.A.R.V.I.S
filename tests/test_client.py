"""
Test client for the JARVIS WebSocket server.
"""
import asyncio
import json
import sys
import websockets
import signal
import time
from typing import Dict, Any, Optional

# Configuration
DEFAULT_PORT = 5001  # Default WebSocket port
SERVER_URI = f"ws://localhost:{DEFAULT_PORT}"
CONNECTION_TIMEOUT = 10  # seconds
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 5

class JarvisClient:
    def __init__(self, server_uri: str):
        self.server_uri = server_uri
        self.websocket = None
        self.running = False
        self.connected = False
        self.last_ping = time.time()
        self.loop = asyncio.get_event_loop()
        
    async def connect(self) -> bool:
        """Connect to the WebSocket server with timeout and retry logic."""
        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            try:
                print(f"[{attempt}/{MAX_RECONNECT_ATTEMPTS}] Connecting to {self.server_uri}...")
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.server_uri,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=1
                    ),
                    timeout=CONNECTION_TIMEOUT
                )
                self.connected = True
                print("✅ Successfully connected to JARVIS server!")
                print("Type your message and press Enter. Type 'exit' to quit.")
                return True
                
            except asyncio.TimeoutError:
                print(f"⌛ Connection to {self.server_uri} timed out after {CONNECTION_TIMEOUT} seconds")
            except OSError as e:
                if "10013" in str(e):
                    print("🔒 Permission denied. Try running as administrator or use a different port.")
                else:
                    print(f"🔌 Connection error: {e}")
            except Exception as e:
                print(f"⚠️ Failed to connect: {e}")
                
            if attempt < MAX_RECONNECT_ATTEMPTS:
                print(f"🔄 Retrying in {RECONNECT_DELAY} seconds...")
                await asyncio.sleep(RECONNECT_DELAY)
        
        print("❌ Failed to connect after multiple attempts")
        return False
        
    async def ensure_connection(self) -> bool:
        """Ensure we have an active connection."""
        if not self.connected or self.websocket is None or self.websocket.closed:  # type: ignore
            return await self.connect()
        return True
    
    async def send_message(self, message: str) -> Optional[Dict]:
        """Send a message to the server and return the response."""
        if not self.websocket or not self.connected:
            print("Not connected to server")
            return None
            
        try:
            # Ensure the message is a string and not empty
            if not message or not message.strip():
                return {'type': 'error', 'message': 'Message cannot be empty'}
                
            await self.websocket.send(json.dumps({
                'type': 'chat',
                'message': message.strip(),
                'timestamp': time.time()
            }))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
            return json.loads(response)
            
        except asyncio.TimeoutError:
            print("Request timed out")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")
            self.connected = False
            await self.reconnect()
        except Exception as e:
            print(f"Error sending message: {e}")
            
        return None
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect to the server."""
        print("Attempting to reconnect...")
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
            
        while not self.websocket and self.running:
            if await self.connect():
                return True
            print(f"Reconnecting in {RECONNECT_DELAY} seconds...")
            await asyncio.sleep(RECONNECT_DELAY)
            
        return False
    
    async def handle_messages(self):
        """Handle incoming messages from the server."""
        if self.websocket is None:
            print("No WebSocket connection")
            self.connected = False
            return
            
        try:
            async for message in self.websocket:  # type: ignore
                try:
                    response = json.loads(message)
                    print(f"\n\nJARVIS ({response.get('model', 'unknown')}): {response.get('message', 'No response')}")
                    print("\nYou: ", end="", flush=True)
                except json.JSONDecodeError:
                    print(f"\n\nReceived non-JSON message: {message}")
                    print("\nYou: ", end="", flush=True)
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection to server closed.")
            self.connected = False
        except Exception as e:
            print(f"\nError in message handler: {e}")
            self.connected = False
    
    async def run(self):
        """Run the client's main loop."""
        self.running = True
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutting down...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initial connection
        if not await self.connect():
            print(f"Could not connect to {self.server_uri}")
            return
        
        # Start message handler
        message_task = asyncio.create_task(self.handle_messages())
        
        try:
            # Main input loop
            while self.running:
                try:
                    # Get user input without blocking
                    try:
                        message = await asyncio.get_event_loop().run_in_executor(
                            None, input, "\nYou: "
                        )
                    except (EOFError, KeyboardInterrupt):
                        break
                        
                    if message.lower().strip() in ('exit', 'quit'):
                        print("Disconnecting...")
                        break
                        
                    # Ensure we're still connected
                    if not await self.ensure_connection():
                        print("Lost connection to server. Please restart the client.")
                        self.running = False
                        break
                        
                    # Send message and get response
                    response = await self.send_message(message)
                    if response and 'error' in response:
                        print(f"Error: {response['message']}")
                    
                except Exception as e:
                    print(f"\nError: {e}")
                    if not await self.reconnect():
                        break
        finally:
            # Cleanup
            self.running = False
            if not message_task.done() and not message_task.cancelled():
                message_task.cancel()
                try:
                    await asyncio.wait_for(message_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                    
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception as e:
                    print(f"Error closing WebSocket: {e}")
            print("Disconnected from server.")

async def main():
    """Main function to run the client."""
    global SERVER_URI
    
    # Allow overriding the server URI via command line
    if len(sys.argv) > 1:
        SERVER_URI = sys.argv[1]
    
    client = JarvisClient(SERVER_URI)
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
