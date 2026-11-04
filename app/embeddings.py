import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from app.config import Settings


class EmbeddingError(ValueError):
    pass


EmbeddingVector = tuple[float, ...]


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension for this provider."""

    async def embed(self, text: str) -> EmbeddingVector:
        """Create one embedding vector for a non-empty text."""

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Create embeddings while preserving input order."""


@dataclass(slots=True)
class HashEmbeddingProvider:
    dimension: int = 64

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise EmbeddingError("dimension must be at least 2")

    async def embed(self, text: str) -> EmbeddingVector:
        normalized = _normalize_text(text)
        values = [0.0] * self.dimension
        tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
        if not tokens:
            raise EmbeddingError("text must contain at least one token")

        for token in tokens:
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=16
            ).digest()
            for offset in range(0, len(digest), 4):
                bucket = int.from_bytes(digest[offset : offset + 4], "big")
                index = bucket % self.dimension
                sign = 1.0 if bucket & 1 else -1.0
                values[index] += sign

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise EmbeddingError("embedding vector has zero magnitude")
        return tuple(value / norm for value in values)

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [await self.embed(text) for text in texts]


@dataclass(slots=True)
class OpenAICompatibleEmbeddingProvider:
    model: str
    api_key: str = field(repr=False)
    base_url: str
    dimension: int
    client: httpx.AsyncClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise EmbeddingError("model cannot be empty")
        if not self.api_key.strip():
            raise EmbeddingError("api_key cannot be empty")
        if not self.base_url.strip():
            raise EmbeddingError("base_url cannot be empty")
        if self.dimension < 2:
            raise EmbeddingError("dimension must be at least 2")

    async def embed(self, text: str) -> EmbeddingVector:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        for text in texts:
            _normalize_text(text)

        request_body = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self.client is None:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await self._post_embeddings(client, request_body)
        else:
            response = await self._post_embeddings(self.client, request_body)
        return self._parse_response(response, len(texts))

    async def _post_embeddings(
        self,
        client: httpx.AsyncClient,
        request_body: dict[str, object],
    ) -> httpx.Response:
        try:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_body,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            raise EmbeddingError("embedding request failed") from None

    def _parse_response(
        self, response: httpx.Response, expected_count: int
    ) -> list[EmbeddingVector]:
        try:
            data = response.json()["data"]
            if not isinstance(data, list) or len(data) != expected_count:
                raise ValueError("unexpected embedding count")

            ordered: list[EmbeddingVector | None] = [None] * expected_count
            for item in data:
                index = item["index"]
                raw_vector = item["embedding"]
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < expected_count
                    or ordered[index] is not None
                    or not isinstance(raw_vector, list)
                ):
                    raise ValueError("invalid embedding response")
                if len(raw_vector) != self.dimension:
                    raise EmbeddingError(
                        f"expected embedding dimension {self.dimension}, "
                        f"got {len(raw_vector)}"
                    )
                vector = tuple(float(value) for value in raw_vector)
                if not all(math.isfinite(value) for value in vector):
                    raise ValueError("embedding values must be finite")
                ordered[index] = vector
            if any(vector is None for vector in ordered):
                raise ValueError("missing embedding response index")
            return [vector for vector in ordered if vector is not None]
        except EmbeddingError:
            raise
        except (IndexError, KeyError, TypeError, ValueError):
            raise EmbeddingError("invalid embedding response") from None


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider(dimension=settings.embedding_dimension)
    if settings.embedding_api_key is None or not settings.embedding_api_key.strip():
        raise EmbeddingError(
            "EMBEDDING_API_KEY is required for the configured embedding provider"
        )
    return OpenAICompatibleEmbeddingProvider(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        dimension=settings.embedding_dimension,
    )


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise EmbeddingError("text must be a string")
    normalized = " ".join(text.split()).casefold()
    if not normalized:
        raise EmbeddingError("text cannot be empty")
    return normalized
