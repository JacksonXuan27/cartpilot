from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

