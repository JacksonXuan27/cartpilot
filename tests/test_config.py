import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_requires_provider_credentials(monkeypatch):
    for name in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_reads_environment_values(monkeypatch):
    values = {
        "CHAT_MODEL": "demo-chat",
        "CHAT_BASE_URL": "https://provider.example/v1",
        "CHAT_API_KEY": "test-key",
        "TOKEN_BUDGET": "750",
        "DATABASE_URL": "sqlite:///./data/test.db",
        "DATABASE_TIMEOUT_SECONDS": "3.5",
        "DATABASE_POOL_SIZE": "3",
        "RERANK_PROVIDER": "siliconflow",
        "RERANK_MODEL": "BAAI/bge-reranker-v2-m3",
        "RERANK_BASE_URL": "https://api.siliconflow.cn/v1",
        "RERANK_API_KEY": "rerank-test-key",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.chat_model == "demo-chat"
    assert settings.chat_base_url == "https://provider.example/v1"
    assert settings.chat_api_key == "test-key"
    assert settings.token_budget == 750
    assert settings.database_url == "sqlite:///./data/test.db"
    assert settings.database_timeout_seconds == 3.5
    assert settings.database_pool_size == 3
    assert settings.rerank_provider == "siliconflow"
    assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert settings.rerank_base_url == "https://api.siliconflow.cn/v1"
    assert settings.rerank_api_key == "rerank-test-key"


def test_settings_rejects_unknown_rerank_provider(monkeypatch):
    values = {
        "CHAT_MODEL": "demo-chat",
        "CHAT_BASE_URL": "https://provider.example/v1",
        "CHAT_API_KEY": "test-key",
        "RERANK_PROVIDER": "unknown",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
