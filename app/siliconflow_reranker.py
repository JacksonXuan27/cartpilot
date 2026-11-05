from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.reranking import RerankerError, _with_score
from app.retrieval import RetrievalResult


@dataclass(slots=True)
class SiliconFlowReranker:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 30.0
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip() or not self.api_key.strip():
            raise RerankerError("rerank base URL, model, and API key are required")

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise RerankerError("query cannot be empty")
        if top_k < 1:
            raise RerankerError("top_k must be positive")
        if not candidates:
            return []
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/rerank",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": [candidate.content for candidate in candidates],
                        "top_n": min(top_k, len(candidates)),
                        "return_documents": False,
                    },
                )
                response.raise_for_status()
                payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RerankerError("rerank provider request failed") from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise RerankerError("rerank provider returned an invalid response")
        ranked: list[tuple[float, RetrievalResult]] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise RerankerError("rerank provider returned an invalid result")
            index = item.get("index")
            score = item.get("relevance_score")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(candidates)
                or index in seen
                or not isinstance(score, (int, float))
            ):
                raise RerankerError("rerank provider returned an invalid ranking")
            seen.add(index)
            ranked.append((float(score), candidates[index]))
        ranked.sort(key=lambda item: (-item[0], item[1].record_id))
        return [_with_score(candidate, score) for score, candidate in ranked[:top_k]]
