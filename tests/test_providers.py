import pytest

from app.contracts import ChatMessage, TokenUsage
from app.providers import ChatModelProvider, ModelProviderError, StubModelProvider


@pytest.mark.asyncio
async def test_stub_provider_implements_the_model_provider_protocol():
    provider = StubModelProvider(reply="Your order is on the way.")
    assert isinstance(provider, ChatModelProvider)

    result = await provider.complete(
        [ChatMessage(role="user", content="Where is my order?")]
    )

    assert result.message == ChatMessage(
        role="assistant", content="Your order is on the way."
    )
    assert result.finish_reason == "stop"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_stub_provider_records_an_isolated_conversation():
    provider = StubModelProvider(
        usage=TokenUsage(prompt_tokens=5, completion_tokens=4, total_tokens=9)
    )
    message = ChatMessage(role="user", content="Can I return this item?")

    await provider.complete([message])
    message.content = "changed after the request"

    assert provider.calls[0][0].content == "Can I return this item?"


@pytest.mark.asyncio
async def test_stub_provider_rejects_an_empty_conversation():
    provider = StubModelProvider()

    with pytest.raises(ModelProviderError, match="at least one message"):
        await provider.complete([])
