from fastapi import WebSocket, APIRouter, WebSocketDisconnect
import logging
import json
import asyncio
from models.jarvis import JarvisModel

websocket_router = APIRouter()
logger = logging.getLogger("jarvis.websocket")

# Gebruik één globale JarvisModel instantie voor alle verbindingen
jarvis_model = JarvisModel()

@websocket_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"[WebSocket] Ontvangen van {client_id}: {data}")
            # Probeer JSON te parsen, anders gebruik als plain text
            try:
                msg = json.loads(data)
                user_message = msg.get("content") or msg.get("message") or data
            except Exception:
                user_message = data
            logger.info(f"[WebSocket] Verwerken via JarvisModel: {user_message}")
            # Verwerk AI-antwoord in threadpool zodat event loop niet blokkeert
            loop = asyncio.get_running_loop()
            try:
                response = await loop.run_in_executor(
                    None,  # default threadpool
                    lambda: jarvis_model.process_input(user_message, user_id=client_id)
                )
                # Maak response plat als 'response' een dict is
                if isinstance(response, dict) and isinstance(response.get('response'), dict):
                    # Haal de string uit de geneste dict
                    inner = response['response']
                    if 'response' in inner and isinstance(inner['response'], str):
                        response['response'] = inner['response']
                    else:
                        response['response'] = str(inner)
                logger.info(f"[WebSocket] AI-response: {response}")
                await websocket.send_text(json.dumps(response))
                logger.info(f"[WebSocket] Response verstuurd naar {client_id}")
            except Exception as e:
                logger.error(f"[WebSocket] Fout bij AI-verwerking: {e}", exc_info=True)
                await websocket.send_text(json.dumps({"success": False, "error": str(e)}))
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
