import json
import os
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.contracts import ChatMessage
from app.embeddings import EmbeddingError, OpenAICompatibleEmbeddingProvider
from app.providers import ModelProviderError, OpenAICompatibleChatProvider
from app.tool_orchestrator import ToolOrchestrator
from app.tool_registry import ToolRegistry
from app.vector_store_factory import create_vector_store, _load_milvus_client_class
from tests.test_runtime_configuration import MilvusBoundaryClient


class EchoArguments(BaseModel):
    value: str


class EchoTool:
    name = "test.echo"
    description = "Return the supplied text."
    input_model = EchoArguments

    async def execute(self, arguments):
        return {"echo": arguments["value"]}


class ToolBoundaryModel:
    def __init__(self):
        self.messages = []
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.messages.append(messages)
        tool_calls = [] if len(self.messages) > 1 else [
            {"name": "test.echo", "args": {"value": "hello"}, "id": "call-1"}
        ]
        return SimpleNamespace(
            content="Tool says hello" if not tool_calls else "",
            tool_calls=tool_calls,
            usage_metadata={},
            response_metadata={"finish_reason": "tool_calls" if tool_calls else "stop"},
        )


@pytest.mark.asyncio
async def test_configured_provider_preserves_tool_call_ids_across_rounds():
    model = ToolBoundaryModel()
    provider = OpenAICompatibleChatProvider(model)
    result = await ToolOrchestrator(provider, ToolRegistry([EchoTool()])).complete(
        [ChatMessage(role="user", content="Echo hello")]
    )

    assert result.message.content == "Tool says hello"
    assert model.tools[0]["name"] == "test.echo"
    second_round = model.messages[1]
    assert second_round[-2].tool_calls[0]["id"] == "call-1"
    assert second_round[-1].tool_call_id == "call-1"
    assert second_round[-1].content == '{"echo": "hello"}'
    assert "tool_calls" not in ChatMessage.model_json_schema()["properties"]


@pytest.mark.asyncio
async def test_configured_model_sanitizes_completion_and_stream_failures():
    class FailingModel:
        async def ainvoke(self, messages):
            raise RuntimeError("private-provider-response")

        async def astream(self, messages):
            raise RuntimeError("private-provider-response")
            yield

    provider = OpenAICompatibleChatProvider(FailingModel())
    messages = [ChatMessage(role="user", content="hello")]
    with pytest.raises(ModelProviderError, match="request failed") as completion:
        await provider.complete(messages)
    with pytest.raises(ModelProviderError, match="stream failed") as stream:
        async for chunk in provider.stream(messages):
            pass

    assert "private-provider-response" not in str(completion.value)
    assert "private-provider-response" not in str(stream.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
    {"data": [{"index": True, "embedding": [1.0, 0.0, 0.0]}]},
    {"data": []},
    {"data": [{"index": 0, "embedding": [float("nan"), 0.0, 0.0]}]},
])
async def test_configured_embeddings_reject_invalid_vectors(response):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=json.dumps(response))
    )) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            model="embedding-test", api_key="embedding-test-key",
            base_url="https://embedding.example/v1", dimension=3, client=client,
        )
        with pytest.raises(EmbeddingError):
            await provider.embed("物流")


@pytest.mark.asyncio
async def test_configured_embeddings_sanitize_http_error_bodies():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(401, text="private-provider-response")
    )) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            model="embedding-test", api_key="embedding-test-key",
            base_url="https://embedding.example/v1", dimension=3, client=client,
        )
        with pytest.raises(EmbeddingError, match="request failed") as error:
            await provider.embed("物流")

    assert "private-provider-response" not in str(error.value)


def test_vector_collection_uses_embedding_dimension_unless_explicitly_named():
    store = create_vector_store(
        Settings(_env_file=None, milvus_uri="http://milvus:19530"), dimension=1024,
        client_factory=lambda **parameters: MilvusBoundaryClient(),
    )
    assert store.collection_name == "cartpilot_chunks_1024"
    store = create_vector_store(
        Settings(_env_file=None, milvus_uri="http://milvus:19530",
                 milvus_collection="existing_chunks"), dimension=1024,
        client_factory=lambda **parameters: MilvusBoundaryClient(),
    )
    assert store.collection_name == "existing_chunks"


def test_milvus_import_does_not_load_dotenv_into_process_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CARTPILOT_IMPORT_SENTINEL", raising=False)
    (tmp_path / ".env").write_text("CARTPILOT_IMPORT_SENTINEL=must-not-load\n")

    assert callable(_load_milvus_client_class())
    assert "CARTPILOT_IMPORT_SENTINEL" not in os.environ
