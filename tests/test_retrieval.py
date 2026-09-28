from uuid import UUID

import pytest

from app.keyword_retrieval import KeywordMatch
from app.retrieval import (
    RetrievalResult,
    RetrievalResultError,
    from_keyword_match,
    from_vector_match,
    normalize_matches,
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
