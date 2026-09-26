import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise EmbeddingError("text must be a string")
    normalized = " ".join(text.split()).casefold()
    if not normalized:
        raise EmbeddingError("text cannot be empty")
    return normalized
