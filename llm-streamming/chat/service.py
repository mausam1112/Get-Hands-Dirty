from chat import utils


class ChatService:
    def __init__(
        self,
        llm_type: str = "ollama",
        llm_name: str = "llama3.2",
    ) -> None:
        self.llm_type = llm_type
        self.llm_name = llm_name

        self.llm = utils.load_llm(self.llm_type, self.llm_name)
