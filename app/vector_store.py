import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from app.embeddings import EmbeddingVector


class VectorStoreError(RuntimeError):
    pass


class VectorDimensionError(VectorStoreError, ValueError):
    pass


class VectorStoreResponseError(VectorStoreError):
    pass


@dataclass(frozen=True, slots=True)
class VectorRecord:
    record_id: str
    document_id: UUID
    chunk_index: int
    content: str
    vector: EmbeddingVector
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorMatch:
    record_id: str
    document_id: UUID
    chunk_index: int
    content: str
    score: float
    metadata: Mapping[str, str]


@runtime_checkable
class VectorStore(Protocol):
    @property
    def dimension(self) -> int:
        """Return the collection vector dimension."""

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or update vector records and return the number written."""

    async def search(
        self, query_vector: EmbeddingVector, top_k: int = 5
    ) -> list[VectorMatch]:
        """Return the nearest records ordered by descending similarity."""


@dataclass(slots=True)
class InMemoryVectorStore:
    dimension: int
    _records: dict[str, VectorRecord] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise VectorDimensionError("dimension must be at least 2")

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            self._validate_vector(record.vector)
        for record in records:
            self._records[record.record_id] = record
        return len(records)

    async def search(
        self, query_vector: EmbeddingVector, top_k: int = 5
    ) -> list[VectorMatch]:
        self._validate_vector(query_vector)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        ranked = sorted(
            (
                (self._cosine_similarity(query_vector, record.vector), record)
                for record in self._records.values()
            ),
            key=lambda item: (-item[0], item[1].record_id),
        )
        return [
            VectorMatch(
                record_id=record.record_id,
                document_id=record.document_id,
                chunk_index=record.chunk_index,
                content=record.content,
                score=score,
                metadata=dict(record.metadata),
            )
            for score, record in ranked[:top_k]
        ]

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dimension:
            raise VectorDimensionError(
                f"expected vector dimension {self.dimension}, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise VectorDimensionError("vector values must be finite")

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float], right: Sequence[float]
    ) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            left_norm * right_norm
        )


class MilvusClientLike(Protocol):
    def has_collection(self, collection_name: str) -> bool: ...

    def create_collection(self, **kwargs: Any) -> Any: ...

    def insert(self, **kwargs: Any) -> Any: ...

    def search(self, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class MilvusVectorStore:
    client: MilvusClientLike
    collection_name: str
    dimension: int
    metric_type: str = "COSINE"
    _ready: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise VectorStoreError("collection_name cannot be empty")
        if self.dimension < 2:
            raise VectorDimensionError("dimension must be at least 2")

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            self._validate_vector(record.vector)
        if not records:
            return 0
        await self._ensure_collection()
        rows = [self._to_row(record) for record in records]
        response = await asyncio.to_thread(
            self.client.insert,
            collection_name=self.collection_name,
            data=rows,
        )
        self._check_response(response, "insert")
        return len(records)

    async def search(
        self, query_vector: EmbeddingVector, top_k: int = 5
    ) -> list[VectorMatch]:
        self._validate_vector(query_vector)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        await self._ensure_collection()
        response = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[list(query_vector)],
            limit=top_k,
            output_fields=[
                "document_id",
                "chunk_index",
                "content",
                "metadata",
            ],
        )
        return self._parse_search_response(response)

    async def _ensure_collection(self) -> None:
        if self._ready:
            return
        exists = await asyncio.to_thread(
            self.client.has_collection, self.collection_name
        )
        if not exists:
            response = await asyncio.to_thread(
                self.client.create_collection,
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type=self.metric_type,
                auto_id=False,
                enable_dynamic_field=True,
            )
            self._check_response(response, "create_collection")
        self._ready = True

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dimension:
            raise VectorDimensionError(
                f"expected vector dimension {self.dimension}, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise VectorDimensionError("vector values must be finite")

    @staticmethod
    def _to_row(record: VectorRecord) -> dict[str, Any]:
        return {
            "id": record.record_id,
            "document_id": str(record.document_id),
            "chunk_index": record.chunk_index,
            "content": record.content,
            "vector": list(record.vector),
            "metadata": json.dumps(dict(record.metadata), ensure_ascii=False),
        }

    @staticmethod
    def _check_response(response: Any, operation: str) -> None:
        if isinstance(response, Mapping) and response.get("code", 0) not in {0, None}:
            raise VectorStoreResponseError(
                f"Milvus {operation} failed: {response.get('message', response)}"
            )

    @staticmethod
    def _parse_search_response(response: Any) -> list[VectorMatch]:
        if not isinstance(response, list) or not response:
            return []
        matches: list[VectorMatch] = []
        for item in response[0]:
            entity = item.get("entity", item)
            raw_metadata = entity.get("metadata", "{}")
            metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            matches.append(
                VectorMatch(
                    record_id=str(item.get("id", entity.get("id"))),
                    document_id=UUID(str(entity["document_id"])),
                    chunk_index=int(entity["chunk_index"]),
                    content=str(entity["content"]),
                    score=float(item.get("distance", item.get("score", 0.0))),
                    metadata=dict(metadata),
                )
            )
        return matches
