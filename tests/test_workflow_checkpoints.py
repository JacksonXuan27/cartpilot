from pathlib import Path

import pytest

from app.contracts import ChatMessage
from app.database import DatabaseManager
from app.providers import StubModelProvider
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowState, WorkflowStatus
from app.workflow_checkpoints import (
    SQLiteWorkflowCheckpointStore,
    WorkflowCheckpoint,
    WorkflowCheckpointAlreadyResumedError,
    WorkflowCheckpointNotFoundError,
)
from app.workflow_runtime import WorkflowRuntime


def make_checkpoint() -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        confirmation_id="confirmation-1",
        run_id="run-1",
        state=WorkflowState(
            workflow_id="workflow-1",
            status=WorkflowStatus.AWAITING_CONFIRMATION,
            current_node="refund-confirmation-interrupt",
            data={
                "messages": [
                    ChatMessage(role="user", content="申请退款"),
                    ChatMessage(
                        role="tool",
                        content='{"order_id":"ORD-1001"}',
                        tool_call_id="call-1",
                    ),
                ],
                "confirmation_id": "confirmation-1",
                "confirmation_required": True,
            },
        ),
    )


def test_sqlite_checkpoint_store_persists_and_restores_workflow(tmp_path: Path):
    database_path = tmp_path / "workflow.db"
    first_database = DatabaseManager(f"sqlite:///{database_path}")
    first_store = SQLiteWorkflowCheckpointStore(first_database)
    first_store.save_pending(make_checkpoint())
    first_store.close()

    restored_store = SQLiteWorkflowCheckpointStore(
        DatabaseManager(f"sqlite:///{database_path}")
    )
    restored = restored_store.claim("confirmation-1")

    assert restored.status == "resuming"
    assert restored.run_id == "run-1"
    assert restored.state.workflow_id == "workflow-1"
    assert restored.state.status is WorkflowStatus.AWAITING_CONFIRMATION
    assert restored.state.data["messages"][0] == ChatMessage(
        role="user", content="申请退款"
    )
    assert restored.state.data["messages"][1].tool_call_id == "call-1"
    restored_store.close()


def test_sqlite_checkpoint_store_rejects_duplicate_claim(tmp_path: Path):
    store = SQLiteWorkflowCheckpointStore(DatabaseManager(f"sqlite:///{tmp_path / 'workflow.db'}"))
    store.save_pending(make_checkpoint())
    store.claim("confirmation-1")

    with pytest.raises(WorkflowCheckpointAlreadyResumedError):
        store.claim("confirmation-1")
    store.close()


def test_sqlite_checkpoint_store_reports_missing_confirmation(tmp_path: Path):
    store = SQLiteWorkflowCheckpointStore(DatabaseManager(f"sqlite:///{tmp_path / 'workflow.db'}"))

    with pytest.raises(WorkflowCheckpointNotFoundError):
        store.claim("missing-confirmation")
    store.close()


@pytest.mark.asyncio
async def test_workflow_resumes_persisted_checkpoint_with_new_runtime(tmp_path: Path):
    database_path = tmp_path / "workflow.db"
    initial_provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-1001",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    initial_store = SQLiteWorkflowCheckpointStore(
        DatabaseManager(f"sqlite:///{database_path}")
    )
    initial_runtime = WorkflowRuntime(
        initial_provider,
        ToolRegistry(),
        checkpoint_store=initial_store,
    )
    paused = await initial_runtime.run(
        [ChatMessage(role="user", content="ORD-1001 到货破损，我要退款")]
    )
    initial_store.close()

    resumed_provider = StubModelProvider(reply="退款申请已登记，等待审核。")
    resumed_store = SQLiteWorkflowCheckpointStore(
        DatabaseManager(f"sqlite:///{database_path}")
    )
    resumed_runtime = WorkflowRuntime(
        resumed_provider,
        ToolRegistry(),
        checkpoint_store=resumed_store,
    )

    resumed = await resumed_runtime.resume(
        str(paused.data["confirmation_id"]),
        confirmed=True,
    )

    assert resumed.status is WorkflowStatus.COMPLETED
    assert resumed.workflow_id == paused.workflow_id
    assert resumed.data["confirmation_status"] == "confirmed"
    assert resumed.data["answer"] == "退款申请已登记，等待审核。"
    assert resumed_provider.calls[0][0].role == "user"
    resumed_store.close()


@pytest.mark.asyncio
async def test_workflow_rejects_confirmation_without_running_model_again():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-1002",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    runtime = WorkflowRuntime(provider, ToolRegistry())
    paused = await runtime.run(
        [ChatMessage(role="user", content="ORD-1002 到货破损，我要退款")]
    )

    rejected = await runtime.resume(
        str(paused.data["confirmation_id"]),
        confirmed=False,
    )

    assert rejected.status is WorkflowStatus.COMPLETED
    assert rejected.data["confirmation_status"] == "rejected"
    assert rejected.data["answer"] == "已取消此售后申请。"
    assert len(provider.calls) == 1
