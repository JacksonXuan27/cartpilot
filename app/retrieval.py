import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from app.keyword_retrieval import KeywordMatch
from app.vector_store import VectorMatch


class RetrievalResultError(ValueError):
    pass


RetrievalMethod = Literal["keyword", "vector", "hybrid"]


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


def reciprocal_rank_fusion(
    keyword_results: Sequence[RetrievalResult] = (),
    vector_results: Sequence[RetrievalResult] = (),
    *,
    rank_constant: int = 60,
    keyword_weight: float = 1.0,
    vector_weight: float = 1.0,
    top_k: int = 10,
) -> list[RetrievalResult]:
    if rank_constant < 1:
        raise RetrievalResultError("rank_constant must be positive")
    if keyword_weight < 0 or not math.isfinite(keyword_weight):
        raise RetrievalResultError("keyword_weight must be finite and non-negative")
    if vector_weight < 0 or not math.isfinite(vector_weight):
        raise RetrievalResultError("vector_weight must be finite and non-negative")
    if keyword_weight == 0 and vector_weight == 0:
        raise RetrievalResultError("at least one fusion weight must be positive")
    if top_k < 1:
        raise RetrievalResultError("top_k must be positive")

    contributions: dict[str, float] = {}
    representatives: dict[str, RetrievalResult] = {}
    methods: dict[str, set[RetrievalMethod]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for weight, method, results in (
        (keyword_weight, "keyword", keyword_results),
        (vector_weight, "vector", vector_results),
    ):
        for rank, result in enumerate(results, start=1):
            if result.method != method:
                raise RetrievalResultError(
                    f"{method} result has incompatible method: {result.method}"
                )
            contributions[result.record_id] = contributions.get(result.record_id, 0.0) + (
                weight / (rank_constant + rank)
            )
            representatives.setdefault(result.record_id, result)
            methods.setdefault(result.record_id, set()).add(method)
            metadata.setdefault(result.record_id, {}).update(result.metadata)

    fused: list[RetrievalResult] = []
    for record_id, score in contributions.items():
        representative = representatives[record_id]
        contributing_methods = methods[record_id]
        fused.append(
            RetrievalResult(
                record_id=record_id,
                document_id=representative.document_id,
                chunk_index=representative.chunk_index,
                content=representative.content,
                score=score,
                method=(
                    "hybrid"
                    if len(contributing_methods) > 1
                    else next(iter(contributing_methods))
                ),
                metadata=metadata[record_id],
            )
        )
    fused.sort(key=lambda result: (-result.score, result.record_id))
    return fused[:top_k]
