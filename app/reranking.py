import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.keyword_retrieval import _tokenize
from app.retrieval import RetrievalResult, RetrievalResultError


class RerankerError(ValueError):
    pass


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Return candidates ordered by query relevance."""


@dataclass(slots=True)
class LexicalReranker:
    original_score_weight: float = 0.1
    phrase_match_bonus: float = 0.2

    def __post_init__(self) -> None:
        if self.original_score_weight < 0 or not math.isfinite(
            self.original_score_weight
        ):
            raise RerankerError(
                "original_score_weight must be finite and non-negative"
            )
        if self.phrase_match_bonus < 0 or not math.isfinite(self.phrase_match_bonus):
            raise RerankerError(
                "phrase_match_bonus must be finite and non-negative"
            )

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        normalized_query = _normalize_query(query)
        if top_k < 1:
            raise RerankerError("top_k must be positive")
        if not candidates:
            return []

        query_terms = set(_tokenize(normalized_query))
        if not query_terms:
            raise RerankerError("query must contain searchable terms")

        maximum_original_score = max(
            max(candidate.score, 0.0) for candidate in candidates
        )
        scored: list[tuple[float, RetrievalResult]] = []
        for candidate in candidates:
            normalized_content = _normalize_query(candidate.content)
            content_terms = set(_tokenize(normalized_content))
            overlap_score = len(query_terms & content_terms) / len(query_terms)
            phrase_bonus = (
                self.phrase_match_bonus
                if normalized_query in normalized_content
                else 0.0
            )
            original_score = (
                max(candidate.score, 0.0) / maximum_original_score
                if maximum_original_score
                else 0.0
            )
            rerank_score = (
                overlap_score
                + phrase_bonus
                + self.original_score_weight * original_score
            )
            scored.append((rerank_score, candidate))

        scored.sort(key=lambda item: (-item[0], item[1].record_id))
        return [
            _with_score(candidate, score) for score, candidate in scored[:top_k]
        ]


def _with_score(candidate: RetrievalResult, score: float) -> RetrievalResult:
    try:
        return RetrievalResult(
            record_id=candidate.record_id,
            document_id=candidate.document_id,
            chunk_index=candidate.chunk_index,
            content=candidate.content,
            score=score,
            method=candidate.method,
            metadata=candidate.metadata,
        )
    except RetrievalResultError as exc:
        raise RerankerError(str(exc)) from exc


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise RerankerError("query must be a string")
    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise RerankerError("query cannot be empty")
    return normalized
