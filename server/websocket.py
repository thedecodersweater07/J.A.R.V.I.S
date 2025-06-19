import json
import sys
import asyncio
import logging
import importlib
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional, Dict, Any, Union, TYPE_CHECKING

logger = logging.getLogger("websocket")

class JarvisModelStub:
    """Stub for JARVIS model when dependencies are not available"""
    def process_input(self, text: str, user_id: str = "default", **kwargs) -> dict:
        return {
            "response": "JARVIS: I'm currently running in limited mode. Some features may not be available.",
            "confidence": 1.0,
            "metadata": {}
        }

# Type hint for the model to avoid circular imports
if TYPE_CHECKING:
    from models.jarvis import JarvisModel

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._jarvis_model: Union[JarvisModel, JarvisModelStub, None] = None
    
    @property
    def jarvis_model(self) -> Union['JarvisModel', JarvisModelStub]:
        """Lazy load the JARVIS model when first accessed"""
        if self._jarvis_model is None:
            self._jarvis_model = self._initialize_model()
        return self._jarvis_model
    
    def _initialize_model(self) -> Union['JarvisModel', JarvisModelStub]:
        """Safely initialize the JARVIS model"""
        try:
            # Clear any existing imports to avoid module caching issues
            if 'models.jarvis' in sys.modules:
                importlib.invalidate_caches()
                sys.modules.pop('models.jarvis', None)
            
            # Import the module and function dynamically
            jarvis_module = importlib.import_module('models.jarvis')
            if hasattr(jarvis_module, 'create_jarvis_model'):
                logger.info("Initializing JARVIS model...")
                return jarvis_module.create_jarvis_model(model_type="language")
            else:
                logger.warning("create_jarvis_model function not found in models.jarvis")
                return JarvisModelStub()
                
        except ImportError as e:
            logger.warning(f"Could not import JARVIS model: {e}")
            return JarvisModelStub()
        except Exception as e:
            logger.error(f"Error initializing JARVIS model: {e}", exc_info=True)
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
            result = self.jarvis_model.process_input(text=message, user_id=client_id)
            response = result.get("response", "I'm sorry, I couldn't process that request.")
            await self.send_message(str(response), client_id)
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg, exc_info=True)
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
