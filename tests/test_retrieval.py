from uuid import UUID

import pytest

from app.keyword_retrieval import KeywordMatch
from app.retrieval import (
    RetrievalResult,
    RetrievalResultError,
    from_keyword_match,
    from_vector_match,
    normalize_matches,
    reciprocal_rank_fusion,
)
from app.vector_store import VectorMatch


DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000501")


def test_keyword_match_maps_to_the_common_retrieval_shape():
    match = KeywordMatch(
        record_id="keyword-1",
        document_id=DOCUMENT_ID,
        content="退款政策",
        score=2.4,
        metadata={"source": "faq.md"},
    )

    result = from_keyword_match(match)

    assert result == RetrievalResult(
        record_id="keyword-1",
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content="退款政策",
        score=2.4,
        method="keyword",
        metadata={"source": "faq.md"},
    )


def test_vector_match_preserves_chunk_index_and_method():
    match = VectorMatch(
        record_id="vector-1",
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="退款通常三个工作日到账",
        score=0.91,
        metadata={"source": "refund.md"},
    )

    result = from_vector_match(match)

    assert result.record_id == "vector-1"
    assert result.chunk_index == 3
    assert result.method == "vector"


def test_normalize_matches_keeps_each_retriever_result_without_fusing():
    keyword_match = KeywordMatch(
        record_id="keyword-1",
        document_id=DOCUMENT_ID,
        content="退款政策",
        score=2.4,
        metadata={"source": "faq.md"},
    )
    vector_match = VectorMatch(
        record_id="vector-1",
        document_id=DOCUMENT_ID,
        chunk_index=1,
        content="退款通常三个工作日到账",
        score=0.91,
        metadata={"source": "refund.md"},
    )

    results = normalize_matches([keyword_match], [vector_match])

    assert [(result.record_id, result.method) for result in results] == [
        ("keyword-1", "keyword"),
        ("vector-1", "vector"),
    ]


def test_retrieval_result_copies_metadata():
    metadata = {"source": "faq.md"}
    result = RetrievalResult(
        record_id="record-1",
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content="content",
        score=0.5,
        method="keyword",
        metadata=metadata,
    )

    metadata["source"] = "changed"

    assert result.metadata == {"source": "faq.md"}


def test_rrf_fuses_ranks_deduplicates_and_marks_hybrid_results():
    keyword_results = [
        RetrievalResult(
            record_id="shared",
            document_id=DOCUMENT_ID,
            chunk_index=0,
            content="退款政策",
            score=4.0,
            method="keyword",
            metadata={"keyword_source": "faq.md"},
        ),
        RetrievalResult(
            record_id="keyword-only",
            document_id=DOCUMENT_ID,
            chunk_index=1,
            content="退款说明",
            score=2.0,
            method="keyword",
        ),
    ]
    vector_results = [
        RetrievalResult(
            record_id="vector-only",
            document_id=DOCUMENT_ID,
            chunk_index=2,
            content="到账时间",
            score=0.9,
            method="vector",
        ),
        RetrievalResult(
            record_id="shared",
            document_id=DOCUMENT_ID,
            chunk_index=0,
            content="退款政策",
            score=0.8,
            method="vector",
            metadata={"vector_source": "refund.md"},
        ),
    ]

    fused = reciprocal_rank_fusion(
        keyword_results,
        vector_results,
        rank_constant=1,
        top_k=3,
    )

    assert [result.record_id for result in fused] == [
        "shared",
        "vector-only",
        "keyword-only",
    ]
    assert fused[0].method == "hybrid"
    assert fused[0].score == pytest.approx(1 / 2 + 1 / 3)
    assert fused[0].metadata == {
        "keyword_source": "faq.md",
        "vector_source": "refund.md",
    }


def test_rrf_respects_source_weights_and_deterministic_ties():
    keyword_results = [
        RetrievalResult(
            record_id="keyword",
            document_id=DOCUMENT_ID,
            chunk_index=0,
            content="keyword",
            score=1,
            method="keyword",
        )
    ]
    vector_results = [
        RetrievalResult(
            record_id="vector",
            document_id=DOCUMENT_ID,
            chunk_index=0,
            content="vector",
            score=1,
            method="vector",
        )
    ]

    fused = reciprocal_rank_fusion(
        keyword_results,
        vector_results,
        rank_constant=1,
        keyword_weight=1,
        vector_weight=2,
    )

    assert [result.record_id for result in fused] == ["vector", "keyword"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank_constant": 0},
        {"keyword_weight": -1},
        {"vector_weight": 0, "keyword_weight": 0},
        {"top_k": 0},
    ],
)
def test_rrf_rejects_invalid_configuration(kwargs):
    with pytest.raises(RetrievalResultError):
        reciprocal_rank_fusion([], [], **kwargs)


def test_rrf_rejects_mismatched_result_methods():
    result = RetrievalResult(
        record_id="wrong",
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content="content",
        score=1,
        method="vector",
    )

    with pytest.raises(RetrievalResultError, match="incompatible"):
        reciprocal_rank_fusion([result], [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"record_id": "", "content": "content"},
        {"record_id": "record-1", "chunk_index": -1, "content": "content"},
        {"record_id": "record-1", "content": " ", "score": 0.5},
        {"record_id": "record-1", "content": "content", "score": float("nan")},
    ],
)
def test_retrieval_result_rejects_invalid_values(kwargs):
    defaults = {
        "record_id": "record-1",
        "document_id": DOCUMENT_ID,
        "chunk_index": 0,
        "content": "content",
        "score": 0.5,
        "method": "vector",
        "metadata": {},
    }
    defaults.update(kwargs)

    with pytest.raises(RetrievalResultError):
        RetrievalResult(**defaults)
