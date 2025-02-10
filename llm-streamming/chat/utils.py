from langchain_ollama.chat_models import ChatOllama


def load_llm(llm_type: str, llm_name: str):
    match llm_type:
        case "ollama" | _:
            return ChatOllama(model=llm_name)
