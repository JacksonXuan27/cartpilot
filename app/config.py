from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

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


class Settings(RerankerSettings):
    chat_provider: Literal["stub", "openai-compatible"] = "stub"
    chat_model: str = "local-stub"
    chat_base_url: str | None = None
    chat_api_key: str | None = Field(default=None, repr=False)
    token_budget: int = 2000
    database_url: str = Field(default="sqlite:///./data/cartpilot.db", min_length=1)
    database_timeout_seconds: float = Field(default=5.0, gt=0)
    database_pool_size: int = Field(default=5, ge=1)
    milvus_uri: str | None = None
    milvus_collection: str | None = Field(default=None, min_length=1)
    embedding_provider: Literal["hash", "openai-compatible"] = "hash"
    embedding_model: str = Field(default="BAAI/bge-m3", min_length=1)
    embedding_base_url: str = Field(
        default="https://api.siliconflow.cn/v1", min_length=1
    )
    embedding_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "EMBEDDING_API_KEY", "EMBED_API_KEY", "embedding_api_key"
        ),
    )
    embedding_dimension: int = Field(default=64, ge=2)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_reranker_settings() -> RerankerSettings:
    return RerankerSettings()
