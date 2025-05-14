"""
AI API Routes
Handles AI processing requests and responses
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

# Import dependencies for auth
from .auth import TokenData
from .dependencies import get_current_active_user

# Setup logging
logger = logging.getLogger("jarvis-server.ai")

# Models
class AIRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    request_type: str = "text"  # text, nlp, ml, full

class AIResponse(BaseModel):
    response: Union[str, Dict[str, Any]]
    request_id: str
    processing_time: float
    timestamp: str

# Create router
router = APIRouter(
    prefix="/ai",
    tags=["ai"],
    responses={404: {"description": "Not found"}},
)

# Routes
@router.post("/query", response_model=AIResponse)
async def process_ai_query(
    request: AIRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Process AI query"""
    # Generate request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    logger.info(f"Processing AI request {request_id} from user {current_user['username']}")
    
    try:
        # Get components from app state
        from ..app import get_ai_components
        llm_core, model_manager, nlp_processor = get_ai_components()
        
        # Process based on request type
        if request.request_type == "nlp" and nlp_processor:
            # NLP processing
            result = nlp_processor.process(request.query, request.context)
        elif request.request_type == "ml" and model_manager:
            # ML processing
            result = {"error": "Direct ML processing not implemented"}
        elif request.request_type == "full" and llm_core and nlp_processor:
            # Full pipeline processing
            # First process with NLP
            nlp_result = nlp_processor.process(request.query)
            
            # Then process with LLM
            llm_context = {
                "nlp": nlp_result,
                "user": current_user["username"],
                "timestamp": datetime.now().isoformat()
            }
            if request.context:
                llm_context.update(request.context)
                
            llm_result = llm_core.generate(request.query, llm_context)
            
            # Combine results
            result = {
                "nlp": nlp_result,
                "llm": llm_result,
                "combined_response": llm_result.get("text", "No response generated")
            }
        elif llm_core:
            # Default to LLM processing
            result = llm_core.generate(request.query, request.context)
        else:
            # Fallback if no components available
            result = {
                "text": "I'm sorry, but the AI processing components are not available at the moment.",
                "error": "AI components not initialized"
            }
    except Exception as e:
        logger.error(f"Error processing AI request: {e}", exc_info=True)
        result = {
            "error": str(e),
            "text": "An error occurred while processing your request."
        }
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return AIResponse(
        response=result,
        request_id=request_id,
        processing_time=processing_time,
        timestamp=datetime.now().isoformat()
    )

@router.get("/status")
async def get_ai_status(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Get AI components status"""
    # Get components from app state
    from ..app import get_ai_components
    llm_core, model_manager, nlp_processor = get_ai_components()
    
    status = {
        "llm_available": llm_core is not None,
        "ml_available": model_manager is not None,
        "nlp_available": nlp_processor is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    return status
