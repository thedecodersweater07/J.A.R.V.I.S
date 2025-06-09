from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class AIRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    request_type: str = "text"
    options: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the weather like?",
                "context": {"location": "New York"},
                "request_type": "text",
                "options": {"max_length": 100}
            }
        }

class AIResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    processing_time: Optional[float] = None
