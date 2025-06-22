from fastapi import WebSocket, APIRouter, WebSocketDisconnect
import logging

websocket_router = APIRouter()
logger = logging.getLogger("jarvis.websocket")

@websocket_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
