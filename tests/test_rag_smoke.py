import asyncio
from uuid import UUID

from fastapi.testclient import TestClient

from app.embeddings import HashEmbeddingProvider
from app.knowledge_base import KnowledgeBaseService
from app.main import app
from app.providers import StubModelProvider
from app.vector_store import InMemoryVectorStore, VectorRecord


def test_rag_http_smoke_from_indexed_chunk_to_answer():
    embedding_provider = HashEmbeddingProvider(dimension=16)
    vector_store = InMemoryVectorStore(dimension=16)
    model_provider = StubModelProvider(reply="退款通常在三个工作日内到账。")
    app.state.knowledge_base_service = KnowledgeBaseService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        model_provider=model_provider,
    )

    vector = asyncio.run(embedding_provider.embed("退款多久到账"))
    asyncio.run(
        vector_store.upsert(
            [
                VectorRecord(
                    record_id="smoke-refund-1",
                    document_id=UUID("00000000-0000-0000-0000-000000000301"),
                    chunk_index=0,
                    content="原路退款通常在三个工作日内到账。",
                    vector=vector,
                    metadata={"source": "refund-policy.md"},
                )
            ]
        )
    )

    response = TestClient(app).post(
        "/knowledge/query",
        json={"query": "退款多久到账", "top_k": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "退款通常在三个工作日内到账。"
    assert body["sources"][0]["metadata"]["source"] == "refund-policy.md"
