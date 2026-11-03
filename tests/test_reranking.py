import json
from uuid import UUID

import httpx
import pytest

from app.config import Settings
from app.reranking import (
    LexicalReranker,
    Reranker,
    RerankerError,
    SiliconFlowReranker,
    create_reranker,
)
from app.retrieval import RetrievalResult


DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000601")


def make_result(record_id: str, content: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        record_id=record_id,
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content=content,
        score=score,
        method="hybrid",
        metadata={"source": "test"},
    )


@pytest.mark.asyncio
async def test_lexical_reranker_implements_protocol_and_prioritizes_query_overlap():
    reranker = LexicalReranker()
    candidates = [
        make_result("generic", "这是物流的一般说明", 0.99),
        make_result("relevant", "退款通常在三个工作日内到账", 0.2),
    ]

    assert isinstance(reranker, Reranker)
    reranked = await reranker.rerank("退款多久到账", candidates)

    assert [candidate.record_id for candidate in reranked] == [
        "relevant",
        "generic",
    ]
    assert reranked[0].score > reranked[1].score


@pytest.mark.asyncio
async def test_reranker_adds_phrase_bonus_and_preserves_candidate_fields():
    reranker = LexicalReranker(original_score_weight=0, phrase_match_bonus=0.3)
    candidate = make_result("refund", "退款多久到账", 0.5)

    reranked = await reranker.rerank("退款多久到账", [candidate])

    assert reranked[0].score == pytest.approx(1.3)
    assert reranked[0].document_id == candidate.document_id
    assert reranked[0].metadata == candidate.metadata
    assert reranked[0].method == "hybrid"


@pytest.mark.asyncio
async def test_reranker_respects_top_k_and_deterministic_ties():
    reranker = LexicalReranker(original_score_weight=0, phrase_match_bonus=0)
    candidates = [
        make_result("b", "退款政策", 1),
        make_result("a", "退款政策", 1),
        make_result("c", "退款政策", 1),
    ]

    reranked = await reranker.rerank("退款", candidates, top_k=2)

    assert [candidate.record_id for candidate in reranked] == ["a", "b"]


@pytest.mark.asyncio
async def test_reranker_returns_empty_for_empty_candidates():
    assert await LexicalReranker().rerank("退款", []) == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LexicalReranker(original_score_weight=-1),
        lambda: LexicalReranker(phrase_match_bonus=float("nan")),
    ],
)
def test_reranker_rejects_invalid_configuration(factory):
    with pytest.raises(RerankerError):
        factory()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", None])
async def test_reranker_rejects_invalid_query(query):
    with pytest.raises(RerankerError):
        await LexicalReranker().rerank(query, [])


@pytest.mark.asyncio
async def test_reranker_rejects_invalid_top_k():
    with pytest.raises(RerankerError, match="top_k"):
        await LexicalReranker().rerank("退款", [], top_k=0)


@pytest.mark.asyncio
async def test_siliconflow_reranker_maps_response_indices_to_original_candidates():
    requests = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.97},
                    {"index": 1, "relevance_score": 0.82},
                ]
            },
        )

    candidates = [
        make_result("generic", "物流的一般说明", 0.8),
        make_result("refund", "退款政策说明", 0.3),
        make_result("return", "退货退款详细流程", 0.2),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
        reranker = SiliconFlowReranker(
            model="BAAI/bge-reranker-v2-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            client=client,
        )

        ranked = await reranker.rerank("如何申请退货退款", candidates, top_k=2)

    assert isinstance(reranker, Reranker)
    assert [candidate.record_id for candidate in ranked] == ["return", "refund"]
    assert [candidate.score for candidate in ranked] == [0.97, 0.82]
    assert all(candidate.method == "hybrid" for candidate in ranked)
    assert requests[0].url == "https://api.siliconflow.cn/v1/rerank"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "如何申请退货退款",
        "documents": [candidate.content for candidate in candidates],
        "top_n": 2,
        "return_documents": False,
    }


@pytest.mark.asyncio
async def test_siliconflow_reranker_sanitizes_upstream_errors():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="private upstream response")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
        reranker = SiliconFlowReranker(
            model="BAAI/bge-reranker-v2-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            client=client,
        )
        with pytest.raises(RerankerError, match="request failed") as error:
            await reranker.rerank("refund", [make_result("one", "refund policy", 1)])

    assert "private upstream" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.7}],
        [{"index": 4, "relevance_score": 0.8}],
    ],
)
async def test_siliconflow_reranker_rejects_invalid_response_indices_and_scores(results):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": results})
        )
    ) as client:
        reranker = SiliconFlowReranker(
            model="BAAI/bge-reranker-v2-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            client=client,
        )
        with pytest.raises(RerankerError, match="response"):
            await reranker.rerank(
                "refund",
                [make_result("one", "refund policy", 1)],
            )


@pytest.mark.asyncio
async def test_siliconflow_reranker_rejects_non_finite_response_scores():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b'{"results":[{"index":0,"relevance_score":NaN}]}',
            )
        )
    ) as client:
        reranker = SiliconFlowReranker(
            model="BAAI/bge-reranker-v2-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            client=client,
        )
        with pytest.raises(RerankerError, match="response"):
            await reranker.rerank(
                "refund",
                [make_result("one", "refund policy", 1)],
            )


def test_reranker_factory_uses_lexical_default():
    settings = Settings(
        _env_file=None,
        chat_model="chat",
        chat_base_url="https://chat.example/v1",
        chat_api_key="chat-key",
    )

    assert isinstance(create_reranker(settings), LexicalReranker)


def test_reranker_factory_requires_key_for_siliconflow():
    settings = Settings(
        _env_file=None,
        chat_model="chat",
        chat_base_url="https://chat.example/v1",
        chat_api_key="chat-key",
        rerank_provider="siliconflow",
    )

    with pytest.raises(RerankerError, match="RERANK_API_KEY"):
        create_reranker(settings)
