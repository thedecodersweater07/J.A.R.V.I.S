"""
J.A.R.V.I.S. WebSocket Manager
-----------------------------
Handles WebSocket connections and message routing.
"""

import json
import logging
import asyncio
from typing import Dict, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket

logger = logging.getLogger("jarvis.websocket")

class WebSocketManager:
    def __init__(self):
        self._active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
        logger.info("WebSocket manager initialized")
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self._active_connections)}")
        await self.broadcast_status()
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            # Find and remove the client
            client_id = None
            for cid, ws in self._active_connections.items():
                if ws == websocket:
                    client_id = cid
                    break
            if client_id:
                del self._active_connections[client_id]
                logger.info(f"Client {client_id} disconnected. Remaining: {len(self._active_connections)}")
    
    async def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        async with self._lock:
            for websocket in self._active_connections.values():
                try:
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self._active_connections.clear()
        logger.info("All WebSocket connections closed")
    
    async def send_personal_json(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a JSON message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)
    
    async def send_error(self, websocket: WebSocket, error_message: str):
        """Send an error message to a specific client."""
        await self.send_personal_json(
            {
                "type": "error",
                "payload": {
                    "message": error_message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            websocket
        )
    
    async def broadcast_json(self, message: Dict[str, Any]):
        """Broadcast a JSON message to all connected clients."""
        if not self._active_connections:
            return
            
        async with self._lock:
            dead_connections = []
            for client_id, websocket in self._active_connections.items():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    dead_connections.append(client_id)
            
            # Clean up dead connections
            for client_id in dead_connections:
                del self._active_connections[client_id]
    
    async def broadcast_status(self):
        """Broadcast system status to all clients."""
        await self.broadcast_json({
            "type": "status",
            "payload": {
                "connected_clients": len(self._active_connections),
                "timestamp": datetime.utcnow().isoformat()
            }
        })
    
    async def handle_message(self, websocket: WebSocket, client_id: str, data: str):
        """Handle incoming WebSocket messages."""
        try:
            message = json.loads(data)
            message_type = message.get("type")
            
            if not message_type:
                raise ValueError("Message type not specified")
            
            if message_type == "chat":
                # Handle chat messages
                await self.broadcast_json({
                    "type": "chat",
                    "payload": {
                        "sender": client_id,
                        "message": message.get("payload", {}).get("message", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
            
            elif message_type == "status_request":
                # Handle status requests
                await self.broadcast_status()
            
            elif message_type == "ping":
                # Handle ping messages
                await self.send_personal_json({"type": "pong"}, websocket)
            
            else:
                # Handle unknown message types
                await self.send_personal_json(
                    {
                        "type": "ack",
                        "payload": f"Received unknown message type: {message_type}"
                    },
                    websocket
                )
                
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON message")
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}", exc_info=True)
            raise

# Singleton instance
manager = WebSocketManager()
