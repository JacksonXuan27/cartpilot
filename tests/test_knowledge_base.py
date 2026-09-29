from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.embeddings import HashEmbeddingProvider
from app.confidence import ThresholdConfidencePolicy
from app.knowledge_base import (
    KnowledgeBaseService,
    KnowledgeQueryRequest,
)
from app.main import app
from app.providers import StubModelProvider
from app.vector_store import InMemoryVectorStore, VectorRecord


DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000201")


def make_service(reply: str = "退款通常在三个工作日内到账。") -> KnowledgeBaseService:
    embedding_provider = HashEmbeddingProvider(dimension=16)
    vector_store = InMemoryVectorStore(dimension=16)
    model_provider = StubModelProvider(reply=reply)
    return KnowledgeBaseService(embedding_provider, vector_store, model_provider)


@pytest.mark.asyncio
async def test_knowledge_service_retrieves_context_and_generates_answer():
    service = make_service()
    vector = await service.embedding_provider.embed("退款多久到账")
    await service.vector_store.upsert(
        [
            VectorRecord(
                record_id="refund-1",
                document_id=DOCUMENT_ID,
                chunk_index=0,
                content="原路退款通常在三个工作日内到账。",
                vector=vector,
                metadata={"source": "refund-policy.md"},
            )
        ]
    )

    response = await service.query(
        KnowledgeQueryRequest(query="退款多久到账", top_k=3)
    )

    assert response.answer == "退款通常在三个工作日内到账。"
    assert response.sources[0].document_id == DOCUMENT_ID
    assert response.sources[0].metadata == {"source": "refund-policy.md"}


@pytest.mark.asyncio
async def test_knowledge_service_returns_empty_sources_without_documents():
    service = make_service(reply="暂未找到相关知识。")

    response = await service.query(KnowledgeQueryRequest(query="未知问题"))

    assert response.answer == "暂未找到相关知识。"
    assert response.sources == []


@pytest.mark.asyncio
async def test_knowledge_service_refuses_low_confidence_context():
    service = make_service(reply="不应调用模型生成答案。")
    service.confidence_policy = ThresholdConfidencePolicy(min_score=0.9)
    vector = await service.embedding_provider.embed("退款多久到账")
    await service.vector_store.upsert(
        [
            VectorRecord(
                record_id="low-confidence",
                document_id=DOCUMENT_ID,
                chunk_index=0,
                content="物流状态会按节点更新。",
                vector=tuple(-value for value in vector),
                metadata={"source": "shipping.md"},
            )
        ]
    )

    response = await service.query(KnowledgeQueryRequest(query="退款多久到账"))

    assert response.answer == "暂时无法根据知识库确认答案，请补充更多信息。"
    assert response.sources == []
    assert service.model_provider.calls == []


def test_knowledge_query_endpoint_uses_application_service():
    app.state.knowledge_base_service = make_service("请参考退款政策。")

    response = TestClient(app).post(
        "/knowledge/query",
        json={"query": "退款政策", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "请参考退款政策。"
    assert body["request_id"]
    assert body["sources"] == []


def test_knowledge_query_validates_query_and_top_k():
    client = TestClient(app)

    assert client.post("/knowledge/query", json={"query": ""}).status_code == 422
    assert client.post("/knowledge/query", json={"query": "refund", "top_k": 0}).status_code == 422
