from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict, Any

router = APIRouter()

@router.post('/login')
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Dummy login logic for demo
    if form_data.username == 'admin' and form_data.password == 'admin':
        return {"access_token": "demo-token", "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

@router.get('/me')
async def get_me():
    return {"username": "admin", "role": "admin"}
