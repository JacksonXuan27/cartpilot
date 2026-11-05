import pytest

from app.config import Settings
from app.embeddings import HashEmbeddingProvider
from app.external_embeddings import OpenAICompatibleEmbeddingProvider
from app.reranking import LexicalReranker
from app.runtime_integrations import (
    create_embedding_provider,
    create_reranker,
    create_vector_store,
)
from app.siliconflow_reranker import SiliconFlowReranker
from app.vector_store import MilvusVectorStore


def test_runtime_integrations_keep_local_deterministic_defaults():
    settings = Settings(_env_file=None)

    assert isinstance(create_embedding_provider(settings), HashEmbeddingProvider)
    assert isinstance(create_reranker(settings), LexicalReranker)


def test_runtime_integrations_build_configured_external_providers():
    settings = Settings(
        embedding_provider="openai-compatible",
        embedding_model="BAAI/bge-m3",
        embedding_base_url="https://embedding.example/v1",
        embed_api_key="embedding-key",
        embedding_dimension=1024,
        rerank_provider="siliconflow",
        rerank_model="BAAI/bge-reranker-v2-m3",
        rerank_base_url="https://rerank.example/v1",
        rerank_api_key="rerank-key",
        _env_file=None,
    )

    embedding = create_embedding_provider(settings)
    reranker = create_reranker(settings)

    assert isinstance(embedding, OpenAICompatibleEmbeddingProvider)
    assert embedding.dimension == 1024
    assert isinstance(reranker, SiliconFlowReranker)


def test_external_integrations_require_credentials():
    with pytest.raises(ValueError, match="EMBED_API_KEY"):
        create_embedding_provider(
            Settings(embedding_provider="openai-compatible", _env_file=None)
        )
    with pytest.raises(ValueError, match="RERANK_API_KEY"):
        create_reranker(Settings(rerank_provider="siliconflow", _env_file=None))


def test_vector_store_uses_configured_milvus_address_and_collection(monkeypatch):
    received = {}

    class ClientAdapter:
        def __init__(self, uri):
            received["uri"] = uri

    monkeypatch.setattr("app.runtime_integrations.PyMilvusClientAdapter", ClientAdapter)
    settings = Settings(
        milvus_uri="http://milvus.internal:19530",
        milvus_collection="acceptance_vectors",
        embedding_dimension=1024,
        _env_file=None,
    )

    store = create_vector_store(settings)

    assert isinstance(store, MilvusVectorStore)
    assert received["uri"] == "http://milvus.internal:19530"
    assert store.collection_name == "acceptance_vectors"
    assert store.dimension == 1024
