from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_provider: Literal["stub", "openai-compatible"] = "stub"
    chat_model: str = "local-stub"
    chat_base_url: str | None = None
    chat_api_key: str | None = Field(default=None, repr=False)
    token_budget: int = 2000
    database_url: str = Field(default="sqlite:///./data/cartpilot.db", min_length=1)
    database_timeout_seconds: float = Field(default=5.0, gt=0)
    database_pool_size: int = Field(default=5, ge=1)
    embedding_provider: Literal["hash", "openai-compatible"] = "hash"
    embedding_model: str = "BAAI/bge-m3"
    embedding_base_url: str | None = None
    embed_api_key: str | None = Field(default=None, repr=False)
    embedding_dimension: int = Field(default=64, ge=2)
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "cartpilot_chunks"
    rerank_provider: Literal["lexical", "siliconflow"] = "lexical"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str | None = Field(default=None, repr=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
