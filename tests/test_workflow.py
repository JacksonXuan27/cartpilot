from datetime import timezone

import pytest

from app.workflow import (
    WorkflowError,
    WorkflowNode,
    WorkflowRuntimeContext,
    WorkflowState,
    WorkflowStatus,
)


def test_workflow_state_starts_pending_with_isolated_data():
    first = WorkflowState()
    second = WorkflowState()

    first.data["intent"] = "refund"

    assert first.workflow_id
    assert first.status is WorkflowStatus.PENDING
    assert first.current_node is None
    assert first.data == {"intent": "refund"}
    assert second.data == {}
    assert first.workflow_id != second.workflow_id


def test_workflow_state_accepts_explicit_execution_progress():
    state = WorkflowState(
        workflow_id="workflow-123",
        status=WorkflowStatus.RUNNING,
        current_node="intent-router",
        data={"query": "where is my order"},
    )

    assert state.workflow_id == "workflow-123"
    assert state.status is WorkflowStatus.RUNNING
    assert state.current_node == "intent-router"
    assert state.data["query"] == "where is my order"


def test_workflow_runtime_context_tracks_request_session_and_creation_time():
    context = WorkflowRuntimeContext(
        request_id="request-123",
        session_id="session-456",
    )

    assert context.run_id
    assert context.request_id == "request-123"
    assert context.session_id == "session-456"
    assert context.created_at.tzinfo == timezone.utc
    assert context.metadata == {}


def test_workflow_runtime_context_copies_metadata():
    metadata = {"source": "chat"}

    context = WorkflowRuntimeContext(request_id="request-123", metadata=metadata)
    metadata["source"] = "changed"

    assert context.metadata == {"source": "chat"}


@pytest.mark.asyncio
async def test_workflow_node_protocol_accepts_async_state_transformer():
    class AddIntentNode:
        name = "add-intent"

        async def execute(self, state, context):
            assert context.request_id == "request-123"
            state.data["intent"] = "refund"
            return state

    node = AddIntentNode()
    state = WorkflowState()
    context = WorkflowRuntimeContext(request_id="request-123")

    assert isinstance(node, WorkflowNode)
    result = await node.execute(state, context)

    assert result is state
    assert result.data["intent"] == "refund"


def test_workflow_node_protocol_rejects_objects_without_execute_method():
    class InvalidNode:
        name = "invalid"

    assert not isinstance(InvalidNode(), WorkflowNode)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WorkflowState(workflow_id=" "),
        lambda: WorkflowState(current_node=" "),
        lambda: WorkflowRuntimeContext(request_id=" "),
        lambda: WorkflowRuntimeContext(request_id="request-1", session_id=" "),
    ],
)
def test_workflow_models_reject_blank_identifiers(factory):
    with pytest.raises(WorkflowError):
        factory()
