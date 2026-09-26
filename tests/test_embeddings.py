import math

import pytest

from app.embeddings import EmbeddingError, EmbeddingProvider, HashEmbeddingProvider


@pytest.mark.asyncio
async def test_hash_embedding_provider_implements_embedding_protocol():
    provider = HashEmbeddingProvider(dimension=16)

    assert isinstance(provider, EmbeddingProvider)
    vector = await provider.embed("Refund usually takes three days")

    assert len(vector) == 16
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)


@pytest.mark.asyncio
async def test_hash_embeddings_are_deterministic_and_case_insensitive():
    provider = HashEmbeddingProvider(dimension=16)

    first = await provider.embed("Order status")
    second = await provider.embed("  order   STATUS  ")

    assert first == second


@pytest.mark.asyncio
async def test_batch_embeddings_preserve_order_and_dimension():
    provider = HashEmbeddingProvider(dimension=8)

    vectors = await provider.embed_batch(["refund", "logistics", "order"])

    assert len(vectors) == 3
    assert all(len(vector) == 8 for vector in vectors)
    assert vectors[0] != vectors[1]


@pytest.mark.asyncio
async def test_embedding_provider_rejects_empty_or_invalid_text():
    provider = HashEmbeddingProvider()

    with pytest.raises(EmbeddingError, match="empty"):
        await provider.embed("   ")
    with pytest.raises(EmbeddingError, match="string"):
        await provider.embed(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("dimension", [0, 1, -4])
def test_embedding_provider_rejects_invalid_dimensions(dimension):
    with pytest.raises(EmbeddingError, match="dimension"):
        HashEmbeddingProvider(dimension=dimension)
