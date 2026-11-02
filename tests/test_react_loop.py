from collections.abc import AsyncIterator, Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import ChatMessage, TokenUsage
from app.providers import ModelResult, ModelProviderError, StubModelProvider, ToolCall
from app.react_loop import ReactLoopError, ReactLoopNode
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowNode, WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class EchoTool:
    name = "test.echo"
    description = "Echo a value for ReAct tests."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, object]:
        return {"echo": arguments["value"]}


class FailingProvider:
    async def complete(self, messages, tools=()) -> ModelResult:
        raise ModelProviderError("model unavailable")

    def stream(self, messages) -> AsyncIterator:
        raise NotImplementedError


def make_state() -> WorkflowState:
    return WorkflowState(
        data={"messages": [ChatMessage(role="user", content="Echo hello")]}
    )


def make_tool_call(call_id: str, value: str = "hello") -> ToolCall:
    return ToolCall(
        name="test.echo",
        arguments={"value": value},
        call_id=call_id,
    )


@pytest.mark.asyncio
async def test_react_loop_completes_without_tool_calls():
    provider = StubModelProvider(reply="直接回答")
    node = ReactLoopNode(provider, ToolRegistry())
    state = make_state()

    result = await node.execute(
        state,
        WorkflowRuntimeContext(request_id="request-1", run_id="run-1"),
    )

    assert isinstance(node, WorkflowNode)
    assert result is state
    assert result.status is WorkflowStatus.COMPLETED
    assert result.current_node == "react-loop"
    assert result.data["answer"] == "直接回答"
    assert result.data["react_iterations"] == 1
    assert result.data["last_model_finish_reason"] == "stop"
    assert result.data["workflow_run_id"] == "run-1"
    assert result.data["messages"][-1] == ChatMessage(
        role="assistant", content="直接回答"
    )


@pytest.mark.asyncio
async def test_react_loop_records_cumulative_model_usage():
    provider = StubModelProvider(
        reply="工具查询完成",
        usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        tool_call_rounds=[(make_tool_call("call-token-1"),)],
    )
    state = make_state()
    context = WorkflowRuntimeContext(request_id="request-token-2", token_budget=12)

    result = await ReactLoopNode(
        provider,
        ToolRegistry([EchoTool()]),
    ).execute(state, context)

    assert result.status is WorkflowStatus.COMPLETED
    assert result.data["token_usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 4,
        "total_tokens": 12,
    }
    assert result.data["token_budget"] == 12


@pytest.mark.asyncio
async def test_react_loop_stops_when_token_budget_is_exceeded():
    provider = StubModelProvider(
        reply="不应返回最终答案",
        usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    state = make_state()
    context = WorkflowRuntimeContext(request_id="request-token-3", token_budget=5)

    with pytest.raises(ReactLoopError, match="token budget exceeded"):
        await ReactLoopNode(provider, ToolRegistry()).execute(state, context)

    assert state.status is WorkflowStatus.FAILED
    assert state.data["error"]["code"] == "token_budget_exceeded"
    assert state.data["token_usage"]["total_tokens"] == 6
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_react_loop_executes_tools_then_uses_tool_message_for_final_answer():
    provider = StubModelProvider(
        reply="工具查询完成",
        tool_call_rounds=[(make_tool_call("call-1"),)],
    )
    node = ReactLoopNode(provider, ToolRegistry([EchoTool()]))
    state = make_state()

    result = await node.execute(
        state,
        WorkflowRuntimeContext(request_id="request-2"),
    )

    assert result.status is WorkflowStatus.COMPLETED
    assert result.data["answer"] == "工具查询完成"
    assert result.data["react_iterations"] == 2
    assert result.data["tool_history"] == [
        {
            "call_id": "call-1",
            "name": "test.echo",
            "arguments": {"value": "hello"},
            "result": {"echo": "hello"},
        }
    ]
    assert [message.role for message in result.data["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert len(provider.calls) == 2
    assert provider.calls[1][-1].role == "tool"


@pytest.mark.asyncio
async def test_react_loop_executes_multiple_tool_calls_in_one_model_round():
    provider = StubModelProvider(
        reply="批量查询完成",
        tool_call_rounds=[
            (make_tool_call("call-1", "one"), make_tool_call("call-2", "two"))
        ],
    )
    state = make_state()

    result = await ReactLoopNode(
        provider,
        ToolRegistry([EchoTool()]),
    ).execute(state, WorkflowRuntimeContext(request_id="request-3"))

    assert result.data["react_iterations"] == 2
    assert [item["call_id"] for item in result.data["tool_history"]] == [
        "call-1",
        "call-2",
    ]
    assert [message.tool_call_id for message in result.data["messages"] if message.role == "tool"] == [
        "call-1",
        "call-2",
    ]


@pytest.mark.asyncio
async def test_react_loop_fails_when_iteration_limit_is_reached():
    provider = StubModelProvider(
        tool_call_rounds=[
            (make_tool_call("call-1"),),
            (make_tool_call("call-2"),),
            (make_tool_call("call-3"),),
        ]
    )
    state = make_state()

    with pytest.raises(ReactLoopError, match="iteration limit"):
        await ReactLoopNode(
            provider,
            ToolRegistry([EchoTool()]),
            max_iterations=2,
        ).execute(state, WorkflowRuntimeContext(request_id="request-4"))

    assert state.status is WorkflowStatus.FAILED
    assert state.data["error"]["code"] == "react_iteration_limit"
    assert state.data["react_iterations"] == 2
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_react_loop_marks_state_failed_when_model_call_fails():
    state = make_state()

    with pytest.raises(ReactLoopError, match="model unavailable") as error:
        await ReactLoopNode(FailingProvider(), ToolRegistry()).execute(
            state,
            WorkflowRuntimeContext(request_id="request-5"),
        )

    assert state.status is WorkflowStatus.FAILED
    assert state.data["error"]["code"] == "react_loop_failed"
    assert isinstance(error.value.__cause__, ModelProviderError)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{}, {"messages": []}, {"messages": "invalid"}])
async def test_react_loop_rejects_invalid_message_state(data):
    state = WorkflowState(data=data)

    with pytest.raises(ReactLoopError):
        await ReactLoopNode(StubModelProvider(), ToolRegistry()).execute(
            state,
            WorkflowRuntimeContext(request_id="request-6"),
        )

    assert state.status is WorkflowStatus.FAILED


def test_react_loop_rejects_invalid_configuration():
    with pytest.raises(ReactLoopError, match="max_iterations"):
        ReactLoopNode(StubModelProvider(), ToolRegistry(), max_iterations=0)
