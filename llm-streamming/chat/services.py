import asyncio

from fastapi.responses import StreamingResponse
from chat import utils
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk
from fastapi import HTTPException


class ChatService:
    def __init__(
        self,
        llm_type: str = "ollama",
        llm_name: str = "llama3.2",
    ) -> None:
        self.llm_type = llm_type
        self.llm_name = llm_name

        self.llm = utils.load_llm(self.llm_type, self.llm_name)

    async def generate(self, user_query: str):
        """Method which generates the answer."""

        if not user_query:
            raise HTTPException(400, {"msg": "Please provide your query."})

        messages = [
            SystemMessage(
                content="You are expert in the field of AI. Response with user query"
            ),
            HumanMessage(content=user_query),
        ]

        async def content_generator():
            async for event in self.llm.astream_events(input=messages, version="v2"):
                try:
                    if event.get("event", "") == "on_chat_model_stream" and isinstance(
                        event.get("data"), dict
                    ):
                        chunk = event.get("data").get("chunk")
                        if isinstance(chunk, AIMessageChunk) and isinstance(
                            chunk.content, str
                        ):
                            yield chunk.content.encode("utf-8")
                    await asyncio.sleep(0.1)
                except Exception:
                    raise HTTPException(500, "Internal Server Error")

        return StreamingResponse(content_generator(), media_type="text/markdown")
