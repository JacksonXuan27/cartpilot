from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.embeddings import EmbeddingError, EmbeddingVector


@dataclass(slots=True)
class OpenAICompatibleEmbeddingProvider:
    base_url: str
    model: str
    api_key: str
    dimension: int
    timeout_seconds: float = 30.0
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip() or not self.api_key.strip():
            raise EmbeddingError("embedding base URL, model, and API key are required")
        if self.dimension < 2:
            raise EmbeddingError("embedding dimension must be at least 2")

    async def embed(self, text: str) -> EmbeddingVector:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("text cannot be empty")
        payload = await self._request({"model": self.model, "input": text})
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != 1:
            raise EmbeddingError("embedding response must contain one vector")
        vector = data[0].get("embedding") if isinstance(data[0], dict) else None
        if not isinstance(vector, list) or len(vector) != self.dimension:
            actual = len(vector) if isinstance(vector, list) else "invalid"
            raise EmbeddingError(
                f"expected {self.dimension} embedding dimensions, got {actual}"
            )
        if any(not isinstance(value, (int, float)) for value in vector):
            raise EmbeddingError("embedding response contains non-numeric values")
        return tuple(float(value) for value in vector)

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("texts cannot contain empty values")
        payload = await self._request({"model": self.model, "input": list(texts)})
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError("embedding response count does not match input")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(vector, list) or len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError("embedding response has an unexpected dimension")
        if any(
            not isinstance(value, (int, float))
            for vector in vectors
            for value in vector
        ):
            raise EmbeddingError("embedding response contains non-numeric values")
        return [tuple(float(value) for value in vector) for vector in vectors]

    async def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError("embedding provider request failed") from exc
        if not isinstance(payload, dict):
            raise EmbeddingError("embedding provider returned an invalid response")
        return payload
