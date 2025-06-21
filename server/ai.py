from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.post('/ask')
async def ask_ai(question: Dict[str, Any]):
    # Dummy AI response for demo
    user_input = question.get('question', '')
    if not user_input:
        raise HTTPException(status_code=400, detail="No question provided")
    return {"answer": f"AI antwoord op: {user_input}"}
