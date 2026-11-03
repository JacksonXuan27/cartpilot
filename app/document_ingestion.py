from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

from app.data_models import KnowledgeDocument
from app.document_processing import (
    DocumentChunker,
    DocumentCleaner,
    DocumentProcessingError,
)
from app.embeddings import EmbeddingProvider
from app.repositories import KnowledgeDocumentRepository, RecordNotFoundError
from app.vector_store import VectorRecord, VectorStore


class DocumentIngestionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentIngestionResult:
    document_id: UUID
    version: int
    chunk_count: int
    vector_count: int


@dataclass(slots=True)
class DocumentIngestionService:
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    document_repository: KnowledgeDocumentRepository
    chunker: DocumentChunker = field(default_factory=DocumentChunker)

    async def ingest(
        self,
        *,
        title: str,
        source: str,
        content: str,
        metadata: Mapping[str, str] | None = None,
        batch_size: int = 32,
        chunk_size: int | None = None,
        overlap_chars: int | None = None,
    ) -> DocumentIngestionResult:
        normalized_title = title.strip() if isinstance(title, str) else ""
        normalized_source = source.strip() if isinstance(source, str) else ""
        if not normalized_title:
            raise DocumentIngestionError("title cannot be empty")
        if not normalized_source:
            raise DocumentIngestionError("source cannot be empty")
        if batch_size < 1:
            raise DocumentIngestionError("batch_size must be positive")
        if self.embedding_provider.dimension != self.vector_store.dimension:
            raise DocumentIngestionError(
                "embedding and vector store dimensions must match"
            )

        try:
            cleaned_content = DocumentCleaner().clean(content)
        except DocumentProcessingError as exc:
            raise DocumentIngestionError(str(exc)) from exc
        chunker = self.chunker
        if chunk_size is not None or overlap_chars is not None:
            chunker = DocumentChunker(
                max_chars=(
                    self.chunker.max_chars if chunk_size is None else chunk_size
                ),
                overlap_chars=(
                    self.chunker.overlap_chars
                    if overlap_chars is None
                    else overlap_chars
                ),
            )
        chunks = chunker.split(cleaned_content)
        document_id = uuid5(NAMESPACE_URL, normalized_source)
        now = datetime.now(timezone.utc)
        try:
            existing = await self.document_repository.get(document_id)
        except RecordNotFoundError:
            existing = None

        vector_metadata = dict(metadata or {})
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in vector_metadata.items()
        ):
            raise DocumentIngestionError("metadata keys and values must be strings")
        vector_metadata["source"] = normalized_source
        vector_metadata["title"] = normalized_title

        vectors = []
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            batch_vectors = await self.embedding_provider.embed_batch(
                [chunk.content for chunk in batch]
            )
            if len(batch_vectors) != len(batch):
                raise DocumentIngestionError(
                    "embedding provider returned an unexpected vector count"
                )
            vectors.extend(batch_vectors)

        records = [
            VectorRecord(
                record_id=f"{document_id}:{chunk.index}",
                document_id=document_id,
                chunk_index=chunk.index,
                content=chunk.content,
                vector=vector,
                metadata=vector_metadata,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        written = await self.vector_store.upsert(records)
        if written != len(records):
            raise DocumentIngestionError(
                "vector store returned an unexpected write count"
            )
        await self.vector_store.delete_document_chunks(
            document_id,
            from_chunk_index=len(chunks),
        )

        document = KnowledgeDocument(
            document_id=document_id,
            title=normalized_title,
            content=cleaned_content,
            source=normalized_source,
            version=1 if existing is None else existing.version + 1,
            metadata=vector_metadata,
            created_at=now if existing is None else existing.created_at,
            updated_at=now,
        )
        await self.document_repository.save(document)
        return DocumentIngestionResult(
            document_id=document_id,
            version=document.version,
            chunk_count=len(chunks),
            vector_count=written,
        )
