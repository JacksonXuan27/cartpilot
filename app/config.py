from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000
    database_url: str = Field(default="sqlite:///./data/cartpilot.db", min_length=1)
    database_timeout_seconds: float = Field(default=5.0, gt=0)
    database_pool_size: int = Field(default=5, ge=1)
    rerank_provider: Literal["lexical", "siliconflow"] = "lexical"
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", min_length=1)
    rerank_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        min_length=1,
    )
    rerank_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("RERANK_API_KEY", "rerank_api_key"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
