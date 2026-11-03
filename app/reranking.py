import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from app.config import RerankerSettings
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


@dataclass(slots=True)
class SiliconFlowReranker:
    model: str
    api_key: str = field(repr=False)
    base_url: str
    client: httpx.AsyncClient | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise RerankerError("model cannot be empty")
        if not self.api_key.strip():
            raise RerankerError("api_key cannot be empty")
        if not self.base_url.strip():
            raise RerankerError("base_url cannot be empty")
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise RerankerError("timeout_seconds must be positive and finite")

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

        top_n = min(top_k, len(candidates))
        body = {
            "model": self.model,
            "query": normalized_query,
            "documents": [candidate.content for candidate in candidates],
            "top_n": top_n,
            "return_documents": False,
        }
        if self.client is None:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await self._request(client, body)
        else:
            response = await self._request(self.client, body)
        return self._parse_response(response, candidates, top_n)

    async def _request(
        self,
        client: httpx.AsyncClient,
        body: dict[str, object],
    ) -> httpx.Response:
        try:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            raise RerankerError("SiliconFlow rerank request failed") from None

    @staticmethod
    def _parse_response(
        response: httpx.Response,
        candidates: Sequence[RetrievalResult],
        top_n: int,
    ) -> list[RetrievalResult]:
        try:
            results = response.json()["results"]
            if not isinstance(results, list) or not 1 <= len(results) <= top_n:
                raise ValueError("invalid result count")

            reranked: list[RetrievalResult] = []
            seen_indices: set[int] = set()
            for item in results:
                index = item["index"]
                score = item["relevance_score"]
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < len(candidates)
                    or index in seen_indices
                    or isinstance(score, bool)
                ):
                    raise ValueError("invalid rerank result")
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError("rerank score must be finite")
                seen_indices.add(index)
                reranked.append(_with_score(candidates[index], score))
            return reranked
        except RerankerError:
            raise
        except (IndexError, KeyError, TypeError, ValueError):
            raise RerankerError("SiliconFlow returned an invalid rerank response") from None


def create_reranker(settings: RerankerSettings) -> Reranker:
    if settings.rerank_provider == "lexical":
        return LexicalReranker()
    if settings.rerank_api_key is None or not settings.rerank_api_key.strip():
        raise RerankerError("RERANK_API_KEY is required for SiliconFlow reranking")
    return SiliconFlowReranker(
        model=settings.rerank_model,
        api_key=settings.rerank_api_key,
        base_url=settings.rerank_base_url,
    )


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
