from uuid import UUID

import pytest

from app.reranking import LexicalReranker, Reranker, RerankerError
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
