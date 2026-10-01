import pytest

from app.contracts import ChatMessage
from app.providers import StubModelProvider
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowStatus
from app.workflow_runtime import WorkflowRuntime, WorkflowRuntimeError


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
