from collections.abc import AsyncIterator, Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import ChatMessage
from app.providers import OpenAICompatibleChatProvider
from app.workflow_runtime import default_workflow_runtime
from app.react_loop import ReactLoopNode
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class EchoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class EchoTool:
    name = "test.echo"
    description = "Return the supplied value."
    input_model = EchoArguments

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, object]:
        return {"echo": arguments["value"]}


class FakeAIMessage:
    def __init__(self, content, tool_calls=None, usage_metadata=None, response_metadata=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage_metadata or {}
        self.response_metadata = response_metadata or {}


class FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = None
        self.received_messages = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.received_messages.append(messages)
        return self.responses.pop(0)

    async def astream(self, messages) -> AsyncIterator[FakeAIMessage]:
        yield FakeAIMessage("stream is not used")


@pytest.mark.asyncio
async def test_openai_compatible_provider_supports_react_tool_roundtrip():
    model = FakeChatModel(
        [
            FakeAIMessage(
                "",
                tool_calls=[
                    {
                        "name": "test.echo",
                        "args": {"value": "hello"},
                        "id": "call-echo-1",
                        "type": "tool_call",
                    }
                ],
                response_metadata={"finish_reason": "tool_calls"},
            ),
            FakeAIMessage("Tool says hello", response_metadata={"finish_reason": "stop"}),
        ]
    )
    provider = OpenAICompatibleChatProvider(model)
    state = WorkflowState(
        data={"messages": [ChatMessage(role="user", content="Echo hello") ]}
    )

    result = await ReactLoopNode(
        provider,
        ToolRegistry([EchoTool()]),
    ).execute(state, WorkflowRuntimeContext(request_id="real-tool-call"))

    assert result.status is WorkflowStatus.COMPLETED
    assert result.data["answer"] == "Tool says hello"
    assert result.data["tool_history"][0]["result"] == {"echo": "hello"}
    assert model.bound_tools == [
        {
            "name": "test.echo",
            "description": "Return the supplied value.",
            "parameters": EchoArguments.model_json_schema(),
        }
    ]
    second_round = model.received_messages[1]
    assert second_round[-2].tool_calls == [
        {
            "name": "test.echo",
            "args": {"value": "hello"},
            "id": "call-echo-1",
            "type": "tool_call",
        }
    ]
    assert second_round[-1].tool_call_id == "call-echo-1"
    assert second_round[-1].content == '{"echo": "hello"}'


def test_default_workflow_runtime_uses_the_configured_provider():
    provider = object()
    runtime = default_workflow_runtime(provider)

    assert runtime.model_provider is provider
    assert runtime.nodes[-1].model_provider is provider

def test_internal_model_tool_calls_are_not_part_of_public_chat_schema():
    assert "tool_calls" not in ChatMessage.model_json_schema()["properties"]
