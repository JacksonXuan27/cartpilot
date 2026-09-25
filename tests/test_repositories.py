from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.data_models import (
    KnowledgeDocument,
    OrderRecord,
    RetrievalHit,
    RetrievalRecord,
)
from app.database import DatabaseManager, DatabaseNotInitializedError
from app.repositories import (
    KnowledgeDocumentRepository,
    OrderRepository,
    RecordNotFoundError,
    RetrievalRecordRepository,
    initialize_schema,
)


NOW = datetime(2026, 9, 25, 10, 30, tzinfo=timezone.utc)


@pytest.fixture
def database():
    manager = DatabaseManager("sqlite:///:memory:")
    manager.open()
    initialize_schema(manager)
    yield manager
    manager.close()


@pytest.mark.asyncio
async def test_order_repository_saves_reads_and_updates_orders(database):
    repository = OrderRepository(database)
    order = OrderRecord(
        order_id="ORD-7001",
        status="paid",
        product_name="Desk lamp",
        quantity=2,
        total_amount=240,
        currency="CNY",
        placed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )

    await repository.save(order)
    updated = order.model_copy(update={"status": "shipped"})
    await repository.save(updated)

    assert await repository.get("ORD-7001") == updated
    assert await repository.list_recent() == [updated]


@pytest.mark.asyncio
async def test_knowledge_repository_searches_by_text_and_source(database):
    repository = KnowledgeDocumentRepository(database)
    document = KnowledgeDocument(
        document_id=uuid4(),
        title="退款到账时间",
        content="退款一般在三个工作日内到账。",
        source="refund-policy",
        created_at=NOW,
        updated_at=NOW,
    )
    await repository.save(document)

    assert await repository.get(document.document_id) == document
    assert await repository.search(query="三个工作日", source="refund-policy") == [
        document
    ]
    assert await repository.search(query="not found") == []


@pytest.mark.asyncio
async def test_retrieval_repository_round_trips_ranked_hits(database):
    repository = RetrievalRecordRepository(database)
    record = RetrievalRecord(
        record_id=uuid4(),
        query="退款到账时间",
        top_k=3,
        hits=[RetrievalHit(document_id=uuid4(), score=0.87, rank=1)],
        latency_ms=8.4,
        created_at=NOW,
    )
    await repository.save(record)

    assert await repository.get(record.record_id) == record
    assert await repository.list_recent() == [record]


@pytest.mark.asyncio
async def test_repositories_report_missing_records(database):
    with pytest.raises(RecordNotFoundError, match="ORD-404"):
        await OrderRepository(database).get("ORD-404")
    with pytest.raises(RecordNotFoundError, match="knowledge document"):
        await KnowledgeDocumentRepository(database).get(uuid4())
    with pytest.raises(RecordNotFoundError, match="retrieval record"):
        await RetrievalRecordRepository(database).get(uuid4())


@pytest.mark.asyncio
async def test_repository_requires_an_open_database():
    repository = OrderRepository(DatabaseManager("sqlite:///:memory:"))

    with pytest.raises(DatabaseNotInitializedError):
        await repository.get("ORD-7001")


@pytest.mark.asyncio
async def test_repository_rejects_out_of_range_limits(database):
    with pytest.raises(ValueError, match="limit"):
        await OrderRepository(database).list_recent(limit=0)
    with pytest.raises(ValueError, match="limit"):
        await KnowledgeDocumentRepository(database).search(limit=1001)
