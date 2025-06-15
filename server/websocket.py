import json
import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional, Dict, Any

logger = logging.getLogger("websocket")

class JarvisModelStub:
    """Stub for JARVIS model when dependencies are not available"""
    def process_message(self, message: str) -> str:
        return "JARVIS: I'm currently running in limited mode. Some features may not be available."

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.jarvis_model = self._initialize_model()
    
    def _initialize_model(self):
        """Safely initialize the JARVIS model"""
        try:
            from models.jarvis import create_jarvis_model
            logger.info("Initializing JARVIS model...")
            return create_jarvis_model(model_type="language")
        except ImportError as e:
            logger.warning(f"Could not import JARVIS model: {e}")
            return JarvisModelStub()
        except Exception as e:
            logger.error(f"Error initializing JARVIS model: {e}")
            return JarvisModelStub()

    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client connected: {client_id}")
        await self.send_message("J.A.R.V.I.S online. All systems operational.", client_id)

    def disconnect(self, client_id: str):
        """Handle client disconnection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client disconnected: {client_id}")

    async def send_message(self, message: str, client_id: str):
        """Send message to a specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json({"type": "system", "content": message})
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def process_message(self, message: str, client_id: str):
        try:
            # Process the message using the JARVIS model
            response = self.jarvis_model.process_message(message)
            await self.send_message(response, client_id)
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            await self.send_message(error_msg, client_id)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connection"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "user" and "content" in message:
                await manager.process_message(message["content"], client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(client_id)
