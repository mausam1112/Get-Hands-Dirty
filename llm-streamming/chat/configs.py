from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ChatSetting(BaseSettings):
    """Setting for chat feature"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    LLM_TYPE: str = Field(default=..., validation_alias="LLM_TYPE")
    LLM_NAME: str = Field(default=..., validation_alias="LLM_NAME")


chat_settings = ChatSetting()
