from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import ChatMessage
from app.intent_routing import IntentRouterNode
from app.providers import ModelResult, StubModelProvider, ToolCall
from app.react_loop import ReactLoopNode
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowStatus
from app.workflow_runtime import WorkflowRuntime, WorkflowRuntimeError


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class EchoTool:
    name = "test.echo"
    description = "Echo a value for workflow runtime tests."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, object]:
        return {"echo": arguments["value"]}


def make_tool_call(call_id: str) -> ToolCall:
    return ToolCall(
        name="test.echo",
        arguments={"value": "hello"},
        call_id=call_id,
    )


@pytest.mark.anyio
async def test_runtime_runs_router_and_react_nodes():
    runtime = WorkflowRuntime(StubModelProvider(reply="回复内容"), ToolRegistry())
    state = await runtime.run([ChatMessage(role="user", content="你好")], session_id="session-1")

    assert state.status is WorkflowStatus.COMPLETED
    assert state.data["intent"] == "smalltalk"
    assert state.data["route"] == "fallback_script"
    assert state.data["answer"] == "回复内容"
    assert state.data["session_id"] == "session-1"
    assert state.data["react_iterations"] == 1
    assert state.data["confirmation_required"] is False


@pytest.mark.anyio
async def test_runtime_completes_normal_tool_call_path():
    provider = StubModelProvider(
        reply="工具查询完成",
        tool_call_rounds=[(make_tool_call("call-1"),)],
    )
    runtime = WorkflowRuntime(provider, ToolRegistry([EchoTool()]))

    state = await runtime.run([ChatMessage(role="user", content="查订单")])

    assert state.status is WorkflowStatus.COMPLETED
    assert state.current_node == "react-loop"
    assert state.data["intent"] == "order"
    assert state.data["route"] == "business"
    assert state.data["react_iterations"] == 2
    assert state.data["tool_history"][0]["result"] == {"echo": "hello"}


@pytest.mark.anyio
async def test_runtime_converts_provider_failure_to_failed_state():
    class BrokenProvider:
        async def complete(self, messages, tools=()) -> ModelResult:
            raise RuntimeError("provider is unavailable")

    runtime = WorkflowRuntime(BrokenProvider(), ToolRegistry())

    state = await runtime.run([ChatMessage(role="user", content="查订单")])

    assert state.status is WorkflowStatus.FAILED
    assert state.current_node == "react-loop"
    assert state.data["error"]["code"] == "react_loop_failed"
    assert state.data["error"]["message"] == "react loop failed: provider is unavailable"


@pytest.mark.anyio
async def test_runtime_converts_react_iteration_limit_to_failed_state():
    provider = StubModelProvider(
        tool_call_rounds=[(make_tool_call("call-1"),), (make_tool_call("call-2"),)]
    )
    registry = ToolRegistry([EchoTool()])
    runtime = WorkflowRuntime(
        provider,
        registry,
        nodes=(IntentRouterNode(), ReactLoopNode(provider, registry, max_iterations=2)),
    )

    state = await runtime.run([ChatMessage(role="user", content="查订单")])

    assert state.status is WorkflowStatus.FAILED
    assert state.data["error"]["code"] == "react_iteration_limit"
    assert state.data["error"]["iteration"] == 2
    assert state.data["react_iterations"] == 2
    assert len(provider.calls) == 2


def test_runtime_rejects_empty_or_duplicate_node_configuration():
    with pytest.raises(WorkflowRuntimeError, match="at least one node"):
        WorkflowRuntime(StubModelProvider(), ToolRegistry(), nodes=())

    class Node:
        name = "same"

    with pytest.raises(WorkflowRuntimeError, match="duplicate"):
        WorkflowRuntime(StubModelProvider(), ToolRegistry(), nodes=(Node(), Node()))


@pytest.mark.anyio
async def test_runtime_rejects_empty_messages():
    runtime = WorkflowRuntime(StubModelProvider(), ToolRegistry())
    with pytest.raises(WorkflowRuntimeError, match="at least one message"):
        await runtime.run([])
