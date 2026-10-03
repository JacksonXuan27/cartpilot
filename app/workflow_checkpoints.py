import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Protocol

from app.contracts import ChatMessage
from app.database import DatabaseManager
from app.workflow import WorkflowState, WorkflowStatus


class WorkflowCheckpointError(RuntimeError):
    pass


class WorkflowCheckpointNotFoundError(WorkflowCheckpointError, LookupError):
    pass


class WorkflowCheckpointAlreadyResumedError(WorkflowCheckpointError):
    pass


class WorkflowCheckpointSerializationError(WorkflowCheckpointError, ValueError):
    pass


@dataclass(slots=True)
class WorkflowCheckpoint:
    confirmation_id: str
    run_id: str
    state: WorkflowState
    status: str = "pending"


class WorkflowCheckpointStore(Protocol):
    def save_pending(self, checkpoint: WorkflowCheckpoint) -> None:
        """Persist a workflow paused for confirmation."""

    def claim(self, confirmation_id: str) -> WorkflowCheckpoint:
        """Atomically claim a pending confirmation checkpoint once."""

    def finish(self, checkpoint: WorkflowCheckpoint, status: str) -> None:
        """Persist the resumed workflow and its terminal checkpoint status."""


class InMemoryWorkflowCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, WorkflowCheckpoint] = {}
        self._lock = RLock()

    def save_pending(self, checkpoint: WorkflowCheckpoint) -> None:
        with self._lock:
            if checkpoint.confirmation_id in self._checkpoints:
                raise WorkflowCheckpointError("confirmation ID already exists")
            self._checkpoints[checkpoint.confirmation_id] = _copy_checkpoint(checkpoint)

    def claim(self, confirmation_id: str) -> WorkflowCheckpoint:
        with self._lock:
            checkpoint = self._checkpoints.get(confirmation_id)
            if checkpoint is None:
                raise WorkflowCheckpointNotFoundError(confirmation_id)
            if checkpoint.status != "pending":
                raise WorkflowCheckpointAlreadyResumedError(confirmation_id)
            checkpoint.status = "resuming"
            return _copy_checkpoint(checkpoint)

    def finish(self, checkpoint: WorkflowCheckpoint, status: str) -> None:
        with self._lock:
            stored = self._checkpoints.get(checkpoint.confirmation_id)
            if stored is None:
                raise WorkflowCheckpointNotFoundError(checkpoint.confirmation_id)
            if stored.status != "resuming":
                raise WorkflowCheckpointAlreadyResumedError(checkpoint.confirmation_id)
            checkpoint.status = status
            self._checkpoints[checkpoint.confirmation_id] = _copy_checkpoint(checkpoint)


class SQLiteWorkflowCheckpointStore:
    def __init__(self, database: DatabaseManager) -> None:
        self._database = database
        self._lock = RLock()
        self._database.open()
        with self._database.transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    confirmation_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        self._database.close()

    def save_pending(self, checkpoint: WorkflowCheckpoint) -> None:
        payload = _serialize_state(checkpoint.state)
        with self._lock, self._database.transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO workflow_checkpoints
                        (confirmation_id, workflow_id, run_id, status, payload, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        checkpoint.confirmation_id,
                        checkpoint.state.workflow_id,
                        checkpoint.run_id,
                        payload,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            except Exception as exc:
                raise WorkflowCheckpointError("could not persist workflow checkpoint") from exc

    def claim(self, confirmation_id: str) -> WorkflowCheckpoint:
        with self._lock, self._database.transaction() as connection:
            row = connection.execute(
                "SELECT workflow_id, run_id, status, payload FROM workflow_checkpoints "
                "WHERE confirmation_id = ?",
                (confirmation_id,),
            ).fetchone()
            if row is None:
                raise WorkflowCheckpointNotFoundError(confirmation_id)
            if row["status"] != "pending":
                raise WorkflowCheckpointAlreadyResumedError(confirmation_id)
            result = connection.execute(
                "UPDATE workflow_checkpoints SET status = 'resuming', updated_at = ? "
                "WHERE confirmation_id = ? AND status = 'pending'",
                (datetime.now(timezone.utc).isoformat(), confirmation_id),
            )
            if result.rowcount != 1:
                raise WorkflowCheckpointAlreadyResumedError(confirmation_id)
            state = _deserialize_state(row["payload"])
            return WorkflowCheckpoint(
                confirmation_id=confirmation_id,
                run_id=row["run_id"],
                state=state,
                status="resuming",
            )

    def finish(self, checkpoint: WorkflowCheckpoint, status: str) -> None:
        payload = _serialize_state(checkpoint.state)
        with self._lock, self._database.transaction() as connection:
            result = connection.execute(
                """
                UPDATE workflow_checkpoints
                SET run_id = ?, status = ?, payload = ?, updated_at = ?
                WHERE confirmation_id = ? AND status = 'resuming'
                """,
                (
                    checkpoint.run_id,
                    status,
                    payload,
                    datetime.now(timezone.utc).isoformat(),
                    checkpoint.confirmation_id,
                ),
            )
            if result.rowcount != 1:
                raise WorkflowCheckpointAlreadyResumedError(
                    checkpoint.confirmation_id
                )
            checkpoint.status = status


def _copy_checkpoint(checkpoint: WorkflowCheckpoint) -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        confirmation_id=checkpoint.confirmation_id,
        run_id=checkpoint.run_id,
        state=deepcopy(checkpoint.state),
        status=checkpoint.status,
    )


def _serialize_state(state: WorkflowState) -> str:
    try:
        payload = {
            "workflow_id": state.workflow_id,
            "status": state.status.value,
            "current_node": state.current_node,
            "data": _encode_value(state.data),
        }
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WorkflowCheckpointSerializationError(
            "workflow state cannot be serialized"
        ) from exc


def _deserialize_state(payload: str) -> WorkflowState:
    try:
        decoded = json.loads(payload)
        data = _decode_value(decoded["data"])
        return WorkflowState(
            workflow_id=decoded["workflow_id"],
            status=WorkflowStatus(decoded["status"]),
            current_node=decoded["current_node"],
            data=data,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkflowCheckpointSerializationError(
            "stored workflow checkpoint is invalid"
        ) from exc


def _encode_value(value: object) -> object:
    if isinstance(value, ChatMessage):
        return {
            "__type__": "ChatMessage",
            "role": value.role,
            "content": value.content,
            "tool_call_id": value.tool_call_id,
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("workflow checkpoint mapping keys must be strings")
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported workflow checkpoint value: {type(value).__name__}")


def _decode_value(value: object) -> object:
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        if value.get("__type__") == "ChatMessage":
            return ChatMessage(
                role=value["role"],
                content=value["content"],
                tool_call_id=value.get("tool_call_id"),
            )
        return {key: _decode_value(item) for key, item in value.items()}
    return value
