import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.message_handlers = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nieuwe WebSocket verbinding. Totaal: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Verbinding verbroken. Resterende verbindingen: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Fout bij verzenden bericht: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                await handle_message(websocket, data)
            except WebSocketDisconnect:
                logger.info("Client heeft de verbinding verbroken")
                break
            except Exception as e:
                logger.error(f"Fout bij verwerken bericht: {e}")
                await websocket.send_json({"error": str(e)})
    except Exception as e:
        logger.error(f"WebSocket fout: {e}")
    finally:
        manager.disconnect(websocket)

async def handle_message(websocket: WebSocket, message: str):
    try:
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "chat":
            # Importeer hier om circulaire imports te voorkomen
            from models.jarvis import handle_message as handle_chat_message
            response = await handle_chat_message(data.get("content", ""))
            await websocket.send_json({
                "type": "chat_response",
                "content": response
            })
            
    except json.JSONDecodeError:
        await websocket.send_json({"error": "Ongeldig JSON formaat"})
    except Exception as e:
        logger.error(f"Fout in handle_message: {e}")
        await websocket.send_json({"error": f"Interne serverfout: {str(e)}"})

# Serve static files
app.mount("/static", StaticFiles(directory="tests"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("tests/index.html")

def start_websocket_server(host="127.0.0.1", port=8080):
    """Start de WebSocket server"""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    logger.info(f"WebSocket server gestart op ws://{host}:{port}")
    return server

if __name__ == "__main__":
    server = start_websocket_server()
    asyncio.run(server.serve())
