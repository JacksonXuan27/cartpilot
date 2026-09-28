import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from app.keyword_retrieval import KeywordMatch
from app.vector_store import VectorMatch


class RetrievalResultError(ValueError):
    pass


RetrievalMethod = Literal["keyword", "vector"]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    record_id: str
    document_id: UUID
    chunk_index: int
    content: str
    score: float
    method: RetrievalMethod
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise RetrievalResultError("record_id cannot be empty")
        if self.chunk_index < 0:
            raise RetrievalResultError("chunk_index must be non-negative")
        if not self.content.strip():
            raise RetrievalResultError("content cannot be empty")
        if not math.isfinite(self.score):
            raise RetrievalResultError("score must be finite")
        object.__setattr__(self, "metadata", dict(self.metadata))


def from_keyword_match(match: KeywordMatch) -> RetrievalResult:
    return RetrievalResult(
        record_id=match.record_id,
        document_id=match.document_id,
        chunk_index=0,
        content=match.content,
        score=match.score,
        method="keyword",
        metadata=match.metadata,
    )


def from_vector_match(match: VectorMatch) -> RetrievalResult:
    return RetrievalResult(
        record_id=match.record_id,
        document_id=match.document_id,
        chunk_index=match.chunk_index,
        content=match.content,
        score=match.score,
        method="vector",
        metadata=match.metadata,
    )


def normalize_matches(
    keyword_matches: Iterable[KeywordMatch] = (),
    vector_matches: Iterable[VectorMatch] = (),
) -> list[RetrievalResult]:
    results = [from_keyword_match(match) for match in keyword_matches]
    results.extend(from_vector_match(match) for match in vector_matches)
    return results
