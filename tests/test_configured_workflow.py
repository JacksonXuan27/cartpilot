import json
from collections.abc import Mapping

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.contracts import ChatMessage
import app.main as main
import app.providers as providers
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowStatus


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class EchoTool:
    name = "test.echo"
    description = "Return the supplied text."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, object]:
        return {"echo": arguments["value"]}


def configured_settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        chat_provider="openai-compatible",
        chat_model="test-chat",
        chat_base_url="https://model.example/v1",
        chat_api_key="test-secret",
        **overrides,
    )


def test_application_uses_configured_model_and_injected_workflow_tools():
    application = main.create_app(
        configured_settings(), tool_registry=ToolRegistry([EchoTool()])
    )

    runtime = application.state.workflow_runtime
    assert isinstance(runtime.model_provider, providers.OpenAICompatibleChatProvider)
    assert [tool.name for tool in runtime.tool_registry.definitions()] == ["test.echo"]


@pytest.mark.asyncio
async def test_workflow_executes_real_provider_tool_call_and_returns_tool_result():
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-echo-1",
                        "type": "function",
                        "function": {
                            "name": "test.echo",
                            "arguments": '{"value":"hello"}',
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "The tool returned hello."}
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"completion-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": "test-chat",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        from langchain_openai import ChatOpenAI

        provider = providers.create_chat_model_provider(
            configured_settings(),
            model_factory=lambda **options: ChatOpenAI(
                **options, http_async_client=client, max_retries=0
            ),
        )
        application = main.create_app(
            configured_settings(),
            tool_registry=ToolRegistry([EchoTool()]),
            model_provider=provider,
        )
        state = await application.state.workflow_runtime.run(
            [ChatMessage(role="user", content="Echo hello")]
        )

    assert state.status is WorkflowStatus.COMPLETED
    assert state.data["answer"] == "The tool returned hello."
    assert state.data["react_iterations"] == 2
    assert state.data["tool_history"][0]["result"] == {"echo": "hello"}
    assert requests[0]["tools"][0]["function"]["name"] == "test.echo"
    assert requests[1]["messages"][-2]["tool_calls"][0]["id"] == "call-echo-1"
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "content": '{"echo": "hello"}',
        "tool_call_id": "call-echo-1",
    }
