import json
import math

import httpx
import pytest

from app.config import Settings
from app.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    create_embedding_provider,
)


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


@pytest.mark.asyncio
async def test_openai_compatible_provider_sends_batch_and_restores_response_order():
    requests = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ],
                "model": "BAAI/bge-m3",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            model="BAAI/bge-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            dimension=3,
            client=client,
        )

        vectors = await provider.embed_batch(["退款进度", "物流状态"])

    assert vectors == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert requests[0].url == "https://api.siliconflow.cn/v1/embeddings"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {
        "model": "BAAI/bge-m3",
        "input": ["退款进度", "物流状态"],
        "encoding_format": "float",
    }


@pytest.mark.asyncio
async def test_openai_compatible_provider_rejects_wrong_dimensions():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            model="BAAI/bge-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            dimension=3,
            client=client,
        )

        with pytest.raises(EmbeddingError, match="dimension"):
            await provider.embed("退款进度")


@pytest.mark.asyncio
async def test_openai_compatible_provider_does_not_expose_provider_error_body():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="secret-bearing upstream response")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            model="BAAI/bge-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            dimension=3,
            client=client,
        )

        with pytest.raises(EmbeddingError, match="request failed") as error:
            await provider.embed("退款进度")

    assert "secret-bearing" not in str(error.value)


@pytest.mark.asyncio
async def test_openai_compatible_provider_rejects_missing_response_indices():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                    {"index": 0, "embedding": [0.0, 1.0, 0.0]},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            model="BAAI/bge-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            dimension=3,
            client=client,
        )

        with pytest.raises(EmbeddingError, match="response"):
            await provider.embed_batch(["退款进度", "物流状态"])


def test_embedding_provider_factory_requires_credentials_for_remote_provider():
    settings = Settings(
        _env_file=None,
        embedding_provider="openai-compatible",
        embedding_dimension=1024,
    )

    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        create_embedding_provider(settings)


def test_embedding_provider_factory_keeps_hash_provider_as_local_default():
    provider = create_embedding_provider(
        Settings(_env_file=None, embedding_dimension=16)
    )

    assert isinstance(provider, HashEmbeddingProvider)
    assert provider.dimension == 16
