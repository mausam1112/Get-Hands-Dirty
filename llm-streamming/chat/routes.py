from chat.services import ChatService
from fastapi import APIRouter

router = APIRouter()
chat_service = ChatService()


@router.post("/generate")
async def generate(user_query: str):
    return await chat_service.generate(user_query)
