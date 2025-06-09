import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter
from typing import Dict
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Import AI components
from core.ai.adapters import ModelManagerAdapter
from llm.core.llm_core import LLMCore
from nlp.processor import NLPProcessor
from core.ai.coordinator import AICoordinator


logger = logging.getLogger("websocket-handler")

router = APIRouter()

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Initialize AI components
ai_coordinator: AICoordinator = None
model_manager: ModelManagerAdapter = None
nlp_processor: NLPProcessor = None
llm_core: LLMCore = None

async def initialize_ai_components():
    """Initialize AI components"""
    global ai_coordinator, model_manager, nlp_processor, llm_core
    try:
        # Initialize LLM Core
        llm_core = LLMCore()
        await llm_core.initialize()
        
        # Initialize Model Manager
        model_manager = ModelManagerAdapter(base_path=Path("data/models"))
        await model_manager.initialize()
        
        # Initialize NLP Processor
        nlp_processor = NLPProcessor(model_name="nl_core_news_sm")
        await nlp_processor.initialize()
        
        # Initialize AI Coordinator
        ai_coordinator = AICoordinator(
            llm_core=llm_core,
            model_manager=model_manager,
            nlp_processor=nlp_processor
        )
        await ai_coordinator.initialize()
        
        logger.info("AI components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing AI components: {str(e)}")
        return False

async def get_ai_response(message: str) -> str:
    """Generate AI response using the full AI system"""
    try:
        # Process message through NLP
        processed_text = await nlp_processor.process_text(message)
        logger.info(f"NLP processed text: {processed_text}")
        
        # Get AI response using the coordinator
        response = await ai_coordinator.process_query(processed_text)
        logger.info(f"AI response: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return f"AI: Error processing your message: {str(e)}"

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Generate a unique client ID
    client_id = f"client_{id(websocket)}"
    active_connections[client_id] = websocket
    
    try:
        # Initialize AI components if not already initialized
        if not ai_coordinator:
            success = await initialize_ai_components()
            if not success:
                response = {
                    'type': 'error',
                    'content': "AI components failed to initialize"
                }
                await websocket.send_text(json.dumps(response))
                raise Exception("AI initialization failed")
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Log incoming message with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{timestamp}] Received message from {client_id}: {message}")
            
            # Process incoming message
            response = await process_message(message)
            
            # Log response with timestamp
            logger.info(f"[{timestamp}] Sending response to {client_id}: {response}")
            
            # Send response back
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] Client {client_id} disconnected")
        del active_connections[client_id]
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"[{timestamp}] Error in websocket handler: {str(e)}")
        response = {
            'type': 'error',
            'content': f"Server error: {str(e)}"
        }
        await websocket.send_text(json.dumps(response))
        raise

async def process_message(message: dict) -> dict:
    """Process incoming messages from the backup screen"""
    try:
        if message.get('type') == 'user':
            content = message['content']
            
            # Get AI response
            response_text = await get_ai_response(content)
            
            # Format response with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_response = f"[{timestamp}] AI: {response_text}"
            
            return {
                'type': 'ai',
                'content': formatted_response
            }
        
        # Handle other message types
        elif message.get('type') == 'status':
            return {
                'type': 'status',
                'content': {
                    'ai_status': 'online',
                    'connections': len(active_connections),
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        return {
            'type': 'error',
            'content': 'Unknown message type'
        }
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {
            'type': 'error',
            'content': f"Error processing your message: {str(e)}"
        }

# Add a health check endpoint
@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "connections": len(active_connections),
        "ai_components": {
            "llm_core": bool(llm_core),
            "model_manager": bool(model_manager),
            "nlp_processor": bool(nlp_processor),
            "ai_coordinator": bool(ai_coordinator)
        },
        "timestamp": datetime.now().isoformat()
    }
