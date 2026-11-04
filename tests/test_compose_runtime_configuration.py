import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is unavailable")
def test_compose_passes_chat_embedding_milvus_and_rerank_settings(tmp_path):
    values = {
        "CHAT_PROVIDER": "openai-compatible",
        "CHAT_MODEL": "chat-test",
        "CHAT_BASE_URL": "https://chat.example/v1",
        "CHAT_API_KEY": "chat-test-key",
        "EMBEDDING_PROVIDER": "openai-compatible",
        "EMBEDDING_MODEL": "embedding-test",
        "EMBEDDING_BASE_URL": "https://embedding.example/v1",
        "EMBED_API_KEY": "embedding-test-key",
        "EMBEDDING_DIMENSION": "3",
        "MILVUS_URI": "http://milvus:19530",
        "MILVUS_COLLECTION": "test_chunks",
        "RERANK_PROVIDER": "siliconflow",
        "RERANK_MODEL": "rerank-test",
        "RERANK_BASE_URL": "https://rerank.example/v1",
        "RERANK_API_KEY": "rerank-test-key",
    }
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{name}={value}" for name, value in values.items()),
                        encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(env_file), "-f", str(root / "compose.yaml"),
         "config", "--format", "json"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    environment = json.loads(result.stdout)["services"]["app"]["environment"]

    for name, value in values.items():
        assert environment.get(name) == value, name
