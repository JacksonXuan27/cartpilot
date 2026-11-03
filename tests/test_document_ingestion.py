from datetime import timezone

import pytest

from app.database import DatabaseManager
from app.document_ingestion import DocumentIngestionError, DocumentIngestionService
from app.embeddings import HashEmbeddingProvider
from app.repositories import KnowledgeDocumentRepository, initialize_schema
from app.vector_store import InMemoryVectorStore


class RecordingEmbeddingProvider:
    def __init__(self, dimension: int = 16) -> None:
        self.provider = HashEmbeddingProvider(dimension=dimension)
        self.dimension = dimension
        self.batch_sizes: list[int] = []

    async def embed(self, text: str):
        return await self.provider.embed(text)

    async def embed_batch(self, texts):
        self.batch_sizes.append(len(texts))
        return await self.provider.embed_batch(texts)


@pytest.fixture
def ingestion_service():
    database = DatabaseManager("sqlite:///:memory:")
    database.open()
    initialize_schema(database)
    embedding_provider = RecordingEmbeddingProvider()
    vector_store = InMemoryVectorStore(dimension=embedding_provider.dimension)
    service = DocumentIngestionService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        document_repository=KnowledgeDocumentRepository(database),
    )
    yield service, embedding_provider, vector_store, database
    database.close()


@pytest.mark.asyncio
async def test_ingestion_persists_document_and_indexes_stable_chunk_ids(
    ingestion_service,
):
    service, _, vector_store, database = ingestion_service

    result = await service.ingest(
        title="Refund policy",
        source="policies/refunds",
        content="Refunds take three business days.",
        batch_size=2,
    )
    stored = await KnowledgeDocumentRepository(database).get(result.document_id)
    matches = await vector_store.search(
        await service.embedding_provider.embed("Refunds take three business days."),
        top_k=5,
    )

    assert result.version == 1
    assert result.chunk_count == result.vector_count == 1
    assert stored.title == "Refund policy"
    assert stored.source == "policies/refunds"
    assert stored.content == "Refunds take three business days."
    assert stored.created_at.tzinfo == timezone.utc
    assert matches[0].record_id == f"{result.document_id}:0"


@pytest.mark.asyncio
async def test_reimport_updates_existing_document_and_removes_stale_chunks(
    ingestion_service,
):
    service, _, vector_store, database = ingestion_service
    await service.ingest(
        title="Returns",
        source="policies/returns",
        content="First policy section. " * 8 + "Trailing section unique phrase.",
        chunk_size=30,
        overlap_chars=0,
    )

    result = await service.ingest(
        title="Returns",
        source="policies/returns",
        content="Updated policy only.",
        chunk_size=30,
        overlap_chars=0,
    )
    stored = await KnowledgeDocumentRepository(database).get(result.document_id)
    matches = await vector_store.search(
        await service.embedding_provider.embed("Updated policy only."),
        top_k=10,
    )

    assert result.version == 2
    assert result.chunk_count == 1
    assert stored.content == "Updated policy only."
    assert stored.created_at < stored.updated_at
    assert {match.chunk_index for match in matches} == {0}
    assert {match.document_id for match in matches} == {result.document_id}


@pytest.mark.asyncio
async def test_ingestion_batches_embeddings_and_rejects_invalid_batch_size(
    ingestion_service,
):
    service, embedding_provider, _, _ = ingestion_service

    result = await service.ingest(
        title="Shipping",
        source="policies/shipping",
        content="First paragraph describes delivery.\n\n"
        "Second paragraph describes tracking.\n\n"
        "Third paragraph describes delays.",
        chunk_size=32,
        overlap_chars=0,
        batch_size=2,
    )

    assert result.chunk_count >= 3
    assert max(embedding_provider.batch_sizes) <= 2
    with pytest.raises(DocumentIngestionError, match="batch_size"):
        await service.ingest(
            title="Invalid",
            source="policies/invalid",
            content="Some content.",
            batch_size=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "title,source,content",
    [
        ("", "policy", "content"),
        ("title", "", "content"),
        ("title", "policy", "   "),
    ],
)
async def test_ingestion_rejects_empty_document_fields(
    ingestion_service, title, source, content
):
    service, _, _, _ = ingestion_service

    with pytest.raises(DocumentIngestionError):
        await service.ingest(title=title, source=source, content=content)
