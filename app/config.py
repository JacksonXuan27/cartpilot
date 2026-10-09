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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
