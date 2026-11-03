import os

from app.config import Settings
from app.vector_store import InMemoryVectorStore, MilvusVectorStore
from app.vector_store_factory import (
    _load_milvus_client_class,
    create_vector_store,
)


def test_vector_store_factory_uses_memory_without_milvus_uri():
    store = create_vector_store(Settings(_env_file=None), dimension=64)

    assert isinstance(store, InMemoryVectorStore)


def test_vector_store_factory_builds_milvus_client_from_configured_uri():
    settings = Settings(
        _env_file=None,
        milvus_uri="http://milvus:19530",
        milvus_collection="test_chunks",
    )
    constructed: list[dict[str, str]] = []

    class FakeMilvusClient:
        def __init__(self, *, uri: str) -> None:
            constructed.append({"uri": uri})

    store = create_vector_store(
        settings,
        dimension=1024,
        client_factory=FakeMilvusClient,
    )

    assert isinstance(store, MilvusVectorStore)
    assert store.collection_name == "test_chunks"
    assert store.dimension == 1024
    assert constructed == [{"uri": "http://milvus:19530"}]


def test_importing_pymilvus_does_not_load_dotenv_into_process_environment(monkeypatch):
    names = ("CHAT_PROVIDER", "CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY")
    for name in names:
        monkeypatch.delenv(name, raising=False)

    client_class = _load_milvus_client_class()
    assert callable(client_class)

    for name in names:
        assert name not in os.environ
