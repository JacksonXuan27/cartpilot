from uuid import UUID

import pytest

from app.keyword_retrieval import (
    BM25Index,
    KeywordDocument,
    KeywordRetrievalError,
)


def make_document(record_id: str, content: str, source: str = "faq.md"):
    return KeywordDocument(
        record_id=record_id,
        document_id=UUID("00000000-0000-0000-0000-000000000401"),
        content=content,
        metadata={"source": source},
    )


def test_bm25_ranks_documents_by_query_term_relevance():
    index = BM25Index()
    index.upsert(
        [
            make_document("refund", "退款 退款 通常需要三个工作日到账"),
            make_document("shipping", "物流通常需要两个工作日更新"),
            make_document("mixed", "退款政策和物流说明"),
        ]
    )

    matches = index.search("退款多久到账", top_k=3)

    assert [match.record_id for match in matches] == ["refund", "mixed"]
    assert matches[0].score > matches[1].score > 0


def test_bm25_normalizes_case_and_supports_chinese_terms():
    index = BM25Index()
    index.upsert([make_document("faq", "Refund policy applies to paid orders")])

    matches = index.search("退款政策")

    assert matches == []
    assert index.search("REFUND POLICY")[0].record_id == "faq"


def test_bm25_upsert_replaces_statistics_and_remove_deletes_document():
    index = BM25Index()
    index.upsert([make_document("doc", "物流查询")])
    index.upsert([make_document("doc", "退款政策")])

    assert index.search("物流") == []
    assert index.search("退款")[0].record_id == "doc"
    assert index.remove("doc") is True
    assert index.size == 0
    assert index.remove("missing") is False


def test_bm25_returns_deterministic_tie_order_and_copies_metadata():
    index = BM25Index()
    index.upsert(
        [
            make_document("b", "refund policy"),
            make_document("a", "refund policy"),
        ]
    )

    matches = index.search("refund")
    matches[0].metadata["source"] = "changed"

    assert [match.record_id for match in matches] == ["a", "b"]
    assert index.search("refund")[0].metadata["source"] == "faq.md"


@pytest.mark.parametrize(
    "operation",
    [
        lambda index: index.search(""),
        lambda index: index.search("refund", top_k=0),
        lambda index: index.upsert([make_document("empty", "")]),
        lambda index: BM25Index(k1=-1),
        lambda index: BM25Index(b=2),
    ],
)
def test_bm25_rejects_invalid_configuration_and_inputs(operation):
    with pytest.raises(KeywordRetrievalError):
        operation(BM25Index())
