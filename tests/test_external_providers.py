import httpx
import pytest

from app.external_embeddings import OpenAICompatibleEmbeddingProvider
from app.retrieval import RetrievalResult
from app.siliconflow_reranker import SiliconFlowReranker


@pytest.mark.asyncio
async def test_embedding_provider_sends_configured_model_and_checks_dimension():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.0, 0.6, 0.8]}]},
        )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://embedding.example/v1",
        model="demo-embedding",
        api_key="test-embedding-key",
        dimension=3,
        transport=httpx.MockTransport(respond),
    )

    vector = await provider.embed("refund status")

    assert vector == (0.0, 0.6, 0.8)
    assert requests[0].url.path == "/v1/embeddings"
    assert requests[0].headers["Authorization"] == "Bearer test-embedding-key"
    assert requests[0].read() == (
        b'{"model":"demo-embedding","input":"refund status"}'
    )


@pytest.mark.asyncio
async def test_embedding_provider_rejects_provider_dimension_mismatch():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://embedding.example/v1",
        model="demo-embedding",
        api_key="test-key",
        dimension=2,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            )
        ),
    )

    with pytest.raises(ValueError, match="expected 2"):
        await provider.embed("refund")


@pytest.mark.asyncio
async def test_siliconflow_reranker_maps_remote_scores_to_candidates():
    document_id = "00000000-0000-0000-0000-000000000101"
    candidate = RetrievalResult(
        record_id="refund-1",
        document_id=__import__("uuid").UUID(document_id),
        chunk_index=0,
        content="Refunds are returned within three business days.",
        score=0.4,
        method="vector",
        metadata={"source": "refund.md"},
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.93}]},
        )

    reranker = SiliconFlowReranker(
        base_url="https://rerank.example/v1",
        model="demo-reranker",
        api_key="test-rerank-key",
        transport=httpx.MockTransport(respond),
    )

    results = await reranker.rerank("refund timing", [candidate], top_k=1)

    assert results[0].score == pytest.approx(0.93)
    assert results[0].record_id == candidate.record_id
    assert results[0].metadata == candidate.metadata
    assert requests[0].url.path == "/v1/rerank"
    assert requests[0].headers["Authorization"] == "Bearer test-rerank-key"
