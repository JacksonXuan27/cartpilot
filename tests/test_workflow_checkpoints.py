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


class FailOnResumeProvider(StubModelProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resume_failures = 0

    async def complete(self, messages, tools=()):
        if self.calls:
            self.resume_failures += 1
            raise RuntimeError("model unavailable during resume")
        return await super().complete(messages, tools)


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
                "context_layers": {
                    "system": [],
                    "summary": ChatMessage(
                        role="assistant",
                        content="较早对话提取式摘要：订单 ORD-1001 需要退款。",
                    ),
                    "long_term": [],
                    "short_term": [],
                    "omitted_message_count": 2,
                },
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
    assert restored.state.data["context_layers"]["summary"] == ChatMessage(
        role="assistant",
        content="较早对话提取式摘要：订单 ORD-1001 需要退款。",
    )
    restored_store.close()


def test_sqlite_checkpoint_store_preserves_assistant_tool_calls(tmp_path: Path):
    tool_message = ChatMessage.assistant_tool_call(
        "",
        [
            {"id": "call-1", "name": "test.echo", "arguments": {"value": "hello"}}
        ],
    )
    checkpoint = make_checkpoint()
    checkpoint.state.data["messages"].append(tool_message)
    store = SQLiteWorkflowCheckpointStore(
        DatabaseManager(f"sqlite:///{tmp_path / 'tool-calls.db'}")
    )

    store.save_pending(checkpoint)
    restored = store.claim("confirmation-1")

    assert restored.state.data["messages"][-1] == tool_message
    store.close()


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


@pytest.mark.asyncio
async def test_workflow_marks_failed_resume_as_terminal_and_rejects_retry():
    provider = FailOnResumeProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-1003",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    runtime = WorkflowRuntime(provider, ToolRegistry())
    paused = await runtime.run(
        [ChatMessage(role="user", content="ORD-1003 到货破损，我要退款")]
    )

    failed = await runtime.resume(
        str(paused.data["confirmation_id"]),
        confirmed=True,
    )

    assert failed.status is WorkflowStatus.FAILED
    assert failed.data["confirmation_status"] == "confirmed"
    assert failed.data["confirmation_required"] is False
    assert failed.data["error"]["code"] == "react_loop_failed"
    assert failed.data["error"]["message"] == (
        "react loop failed: model unavailable during resume"
    )
    assert provider.resume_failures == 1

    with pytest.raises(WorkflowCheckpointAlreadyResumedError):
        await runtime.resume(
            str(paused.data["confirmation_id"]),
            confirmed=True,
        )

    assert provider.resume_failures == 1


@pytest.mark.asyncio
async def test_sqlite_store_keeps_failed_resume_terminal_after_reopen(tmp_path: Path):
    database_path = tmp_path / "workflow.db"
    provider = FailOnResumeProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-1004",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    first_store = SQLiteWorkflowCheckpointStore(
        DatabaseManager(f"sqlite:///{database_path}")
    )
    runtime = WorkflowRuntime(provider, ToolRegistry(), checkpoint_store=first_store)
    paused = await runtime.run(
        [ChatMessage(role="user", content="ORD-1004 到货破损，我要退款")]
    )

    failed = await runtime.resume(
        str(paused.data["confirmation_id"]),
        confirmed=True,
    )
    assert failed.status is WorkflowStatus.FAILED
    first_store.close()

    reopened_store = SQLiteWorkflowCheckpointStore(
        DatabaseManager(f"sqlite:///{database_path}")
    )
    reopened_runtime = WorkflowRuntime(
        StubModelProvider(reply="不应再次调用模型"),
        ToolRegistry(),
        checkpoint_store=reopened_store,
    )

    with pytest.raises(WorkflowCheckpointAlreadyResumedError):
        await reopened_runtime.resume(
            str(paused.data["confirmation_id"]),
            confirmed=True,
        )

    reopened_store.close()
