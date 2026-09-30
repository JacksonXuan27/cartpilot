from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import ChatMessage
from app.providers import ToolCall
from app.tool_registry import ToolRegistry
from app.tool_calling import ToolCallNode, ToolCallNodeError
from app.workflow import WorkflowNode, WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class EchoTool:
    name = "test.echo"
    description = "Echo a value for workflow tests."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, object]:
        return {"value": arguments["value"], "handled": True}


class FailingTool:
    name = "test.fail"
    description = "Fail for workflow tests."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> None:
        raise RuntimeError("downstream unavailable")


@pytest.mark.asyncio
async def test_tool_call_node_executes_call_writes_result_and_appends_tool_message():
    node = ToolCallNode(ToolRegistry([EchoTool()]))
    state = WorkflowState(
        data={
            "tool_call": ToolCall(
                name="test.echo",
                arguments={"value": "hello"},
                call_id="call-echo-1",
            ),
            "messages": [ChatMessage(role="user", content="Echo hello")],
        }
    )
    context = WorkflowRuntimeContext(request_id="request-1", run_id="run-1")

    result = await node.execute(state, context)

    assert isinstance(node, WorkflowNode)
    assert result is state
    assert result.status is WorkflowStatus.RUNNING
    assert result.current_node == "tool-call"
    assert result.data["tool_name"] == "test.echo"
    assert result.data["tool_call_id"] == "call-echo-1"
    assert result.data["tool_arguments"] == {"value": "hello"}
    assert result.data["tool_result"] == {"value": "hello", "handled": True}
    assert result.data["workflow_run_id"] == "run-1"
    assert result.data["messages"][-1] == ChatMessage(
        role="tool",
        tool_call_id="call-echo-1",
        content='{"handled": true, "value": "hello"}',
    )


@pytest.mark.asyncio
async def test_tool_call_node_accepts_mapping_and_serializes_model_result():
    node = ToolCallNode(ToolRegistry([EchoTool()]))
    state = WorkflowState(
        data={
            "tool_call": {
                "name": "test.echo",
                "arguments": {"value": "mapping"},
                "call_id": "call-echo-2",
            }
        }
    )

    result = await node.execute(
        state,
        WorkflowRuntimeContext(request_id="request-2"),
    )

    assert result.data["tool_result_json"] == '{"handled": true, "value": "mapping"}'


@pytest.mark.asyncio
async def test_tool_call_node_marks_state_failed_for_invalid_arguments():
    node = ToolCallNode(ToolRegistry([EchoTool()]))
    state = WorkflowState(
        data={
            "tool_call": ToolCall(
                name="test.echo",
                arguments={"value": ""},
                call_id="call-invalid-1",
            )
        }
    )

    with pytest.raises(ToolCallNodeError, match="test.echo") as error:
        await node.execute(state, WorkflowRuntimeContext(request_id="request-3"))

    assert error.value.tool_name == "test.echo"
    assert state.status is WorkflowStatus.FAILED
    assert state.current_node == "tool-call"
    assert state.data["error"]["code"] == "tool_call_failed"
    assert state.data["error"]["tool_name"] == "test.echo"


@pytest.mark.asyncio
async def test_tool_call_node_wraps_unknown_and_runtime_tool_errors():
    unknown_state = WorkflowState(
        data={
            "tool_call": ToolCall(
                name="missing.tool", arguments={}, call_id="call-missing-1"
            )
        }
    )
    failing_state = WorkflowState(
        data={
            "tool_call": ToolCall(
                name="test.fail", arguments={"value": "x"}, call_id="call-fail-1"
            )
        }
    )
    node = ToolCallNode(ToolRegistry([FailingTool()]))
    context = WorkflowRuntimeContext(request_id="request-4")

    with pytest.raises(ToolCallNodeError, match="missing.tool"):
        await node.execute(unknown_state, context)
    with pytest.raises(ToolCallNodeError, match="test.fail") as error:
        await node.execute(failing_state, context)

    assert unknown_state.status is WorkflowStatus.FAILED
    assert failing_state.status is WorkflowStatus.FAILED
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"tool_call": None},
        {"tool_call": {"name": "test.echo", "arguments": {}}},
        {"tool_call": {"name": "test.echo", "arguments": [], "call_id": "call-1"}},
    ],
)
async def test_tool_call_node_rejects_missing_or_malformed_calls(data):
    state = WorkflowState(data=data)

    with pytest.raises(ToolCallNodeError):
        await ToolCallNode(ToolRegistry([EchoTool()])).execute(
            state,
            WorkflowRuntimeContext(request_id="request-5"),
        )

    assert state.status is WorkflowStatus.FAILED
