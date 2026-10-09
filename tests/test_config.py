import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_defaults_to_stub_provider_without_credentials(monkeypatch):
    for name in ("CHAT_PROVIDER", "CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.chat_provider == "stub"


def test_settings_reads_environment_values(monkeypatch):
    values = {
        "CHAT_MODEL": "demo-chat",
        "CHAT_PROVIDER": "openai-compatible",
        "CHAT_BASE_URL": "https://provider.example/v1",
        "CHAT_API_KEY": "test-key",
        "TOKEN_BUDGET": "750",
        "DATABASE_URL": "sqlite:///./data/test.db",
        "DATABASE_TIMEOUT_SECONDS": "3.5",
        "DATABASE_POOL_SIZE": "3",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.chat_model == "demo-chat"
    assert settings.chat_provider == "openai-compatible"
    assert settings.chat_base_url == "https://provider.example/v1"
    assert settings.chat_api_key == "test-key"
    assert settings.token_budget == 750
    assert settings.database_url == "sqlite:///./data/test.db"
    assert settings.database_timeout_seconds == 3.5
    assert settings.database_pool_size == 3


def test_settings_rejects_unknown_chat_provider(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "unknown")
    monkeypatch.setenv("CHAT_MODEL", "demo-chat")
    monkeypatch.setenv("CHAT_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("CHAT_API_KEY", "test-key")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_loads_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHAT_PROVIDER=openai-compatible\n"
        "CHAT_MODEL=demo-chat\n"
        "CHAT_BASE_URL=https://provider.example/v1\n"
        "CHAT_API_KEY=test-key\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.chat_provider == "openai-compatible"
    assert settings.chat_model == "demo-chat"
    assert settings.chat_api_key == "test-key"
