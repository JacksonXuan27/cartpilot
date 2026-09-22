from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.data_models import (
    KnowledgeDocument,
    OrderRecord,
    RetrievalHit,
    RetrievalRecord,
)


NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def test_order_record_validates_persisted_order_fields():
    order = OrderRecord(
        order_id="ORD-6001",
        status="paid",
        product_name="Noise-cancelling headphones",
        quantity=1,
        total_amount=899.0,
        currency="CNY",
        placed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )

    assert order.order_id == "ORD-6001"
    assert order.model_dump(mode="json")["placed_at"].endswith("Z")


def test_knowledge_document_keeps_source_version_and_metadata():
    document_id = uuid4()
    document = KnowledgeDocument(
        document_id=document_id,
        title="退款时效",
        content="原路退款通常需要三至五个工作日到账。",
        source="after-sales/refund-policy.md",
        metadata={"category": "refund", "locale": "zh-CN"},
        created_at=NOW,
        updated_at=NOW,
    )

    assert document.document_id == document_id
    assert document.version == 1
    assert document.metadata["category"] == "refund"


def test_retrieval_record_contains_ranked_hits():
    document_id = uuid4()
    record = RetrievalRecord(
        record_id=uuid4(),
        query="退款多久到账",
        top_k=3,
        hits=[RetrievalHit(document_id=document_id, score=0.93, rank=1)],
        latency_ms=12.5,
        created_at=NOW,
    )

    assert record.hits[0].document_id == document_id
    assert record.hits[0].rank == 1


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            OrderRecord,
            {
                "order_id": "bad id",
                "status": "paid",
                "product_name": "Keyboard",
                "quantity": 1,
                "total_amount": 1,
                "currency": "CNY",
                "placed_at": NOW,
                "created_at": NOW,
                "updated_at": NOW,
            },
        ),
        (
            KnowledgeDocument,
            {
                "document_id": uuid4(),
                "title": "Policy",
                "content": "Content",
                "source": "policy.md",
                "version": 0,
                "created_at": NOW,
                "updated_at": NOW,
            },
        ),
        (
            RetrievalRecord,
            {
                "record_id": uuid4(),
                "query": "refund",
                "top_k": 0,
                "latency_ms": 1,
                "created_at": NOW,
            },
        ),
    ],
)
def test_data_models_reject_invalid_persistence_values(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_data_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        KnowledgeDocument(
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            title="Policy",
            content="Content",
            source="policy.md",
            created_at=NOW,
            updated_at=NOW,
            unexpected="value",
        )
