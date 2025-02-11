from chat import utils
from langchain_core.messages import SystemMessage, HumanMessage
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

        async for event in self.llm.astream(input=messages):
            print(event.content, end=" ")
