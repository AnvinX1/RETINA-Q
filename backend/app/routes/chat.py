from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from app.services.llm_agent import generate_chat_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    context_data: dict
    chat_history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str

@router.post("")
async def submit_chat(request: ChatRequest):
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        reply = generate_chat_response(
            context_data=request.context_data,
            user_message=request.message,
            chat_history=request.chat_history
        )
        return ChatResponse(reply=reply)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
