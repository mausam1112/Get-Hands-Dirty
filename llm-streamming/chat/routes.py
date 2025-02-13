from chat.configs import chat_settings
from chat.services import ChatService
from fastapi import APIRouter

router = APIRouter()
chat_service = ChatService(chat_settings.LLM_TYPE, chat_settings.LLM_NAME)


@router.post("/generate")
async def generate(user_query: str):
    return await chat_service.generate(user_query)
