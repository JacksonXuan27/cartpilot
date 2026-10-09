import pytest

from app.config import Settings
from app.contracts import ChatMessage, TokenUsage
from app.providers import (
    ModelProviderError,
    OpenAICompatibleChatProvider,
    StubModelProvider,
    create_chat_model_provider,
)


class FakeAIMessage:
    def __init__(self, content, usage_metadata=None, response_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata or {}
        self.response_metadata = response_metadata or {}
        self.tool_calls = []


class FakeChatModel:
    def __init__(self, response=None, chunks=None):
        self.response = response or FakeAIMessage("Hello from the model")
        self.chunks = chunks or []
        self.received_messages = None

    async def ainvoke(self, messages):
        self.received_messages = messages
        return self.response

    async def astream(self, messages):
        self.received_messages = messages
        for chunk in self.chunks:
            yield chunk


def test_factory_uses_stub_provider_by_default():
    provider = create_chat_model_provider(Settings(_env_file=None))

    assert isinstance(provider, StubModelProvider)


def test_factory_requires_credentials_for_real_provider():
    settings = Settings(
        chat_provider="openai-compatible",
        chat_model="demo-chat",
        chat_base_url="https://provider.example/v1",
        chat_api_key="",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="CHAT_API_KEY"):
        create_chat_model_provider(settings)


def test_factory_passes_credentials_to_openai_client():
    settings = Settings(
        chat_provider="openai-compatible",
        chat_model="demo-chat",
        chat_base_url="https://provider.example/v1",
        chat_api_key="test-key",
        _env_file=None,
    )
    received = {}

    def model_factory(**kwargs):
        received.update(kwargs)
        return FakeChatModel()

    provider = create_chat_model_provider(settings, model_factory=model_factory)

    assert isinstance(provider, OpenAICompatibleChatProvider)
    assert received == {
        "model": "demo-chat",
        "base_url": "https://provider.example/v1",
        "api_key": "test-key",
    }


@pytest.mark.asyncio
async def test_openai_provider_maps_completion_and_token_usage():
    model = FakeChatModel(
        FakeAIMessage(
            "The answer",
            usage_metadata={
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            response_metadata={"finish_reason": "stop"},
        )
    )
    provider = OpenAICompatibleChatProvider(model)

    result = await provider.complete(
        [
            ChatMessage(role="system", content="Be concise"),
            ChatMessage(role="user", content="Hi"),
        ]
    )

    assert result.message == ChatMessage(role="assistant", content="The answer")
    assert result.usage == TokenUsage(
        prompt_tokens=7, completion_tokens=3, total_tokens=10
    )
    assert [message.type for message in model.received_messages] == ["system", "human"]


@pytest.mark.asyncio
async def test_openai_provider_streams_text_and_finish_reason():
    model = FakeChatModel(
        chunks=[
            FakeAIMessage("Hel"),
            FakeAIMessage("lo", response_metadata={"finish_reason": "stop"}),
        ]
    )
    provider = OpenAICompatibleChatProvider(model)

    chunks = [
        chunk
        async for chunk in provider.stream(
            [ChatMessage(role="user", content="Hi")]
        )
    ]

    assert [chunk.delta for chunk in chunks] == ["Hel", "lo"]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_provider_rejects_tools_until_tool_roundtrip_is_supported():
    provider = OpenAICompatibleChatProvider(FakeChatModel())

    with pytest.raises(ModelProviderError, match="tool calling is not supported"):
        await provider.complete(
            [ChatMessage(role="user", content="Check my order")],
            tools=[{"name": "order_lookup"}],
        )
