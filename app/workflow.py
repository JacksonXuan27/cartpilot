from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable
from uuid import uuid4

from app.contracts import ChatMessage, TokenUsage
from app.context_layers import ContextLayerManager


class WorkflowError(ValueError):
    pass


class WorkflowTokenBudgetExceededError(RuntimeError):
    code = "token_budget_exceeded"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class WorkflowState:
    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_node: str | None = None
    data: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.workflow_id, str) or not self.workflow_id.strip():
            raise WorkflowError("workflow_id cannot be empty")
        if not isinstance(self.status, WorkflowStatus):
            raise WorkflowError("status must be a WorkflowStatus")
        if self.current_node is not None and (
            not isinstance(self.current_node, str) or not self.current_node.strip()
        ):
            raise WorkflowError("current_node cannot be empty")
        if not isinstance(self.data, Mapping):
            raise WorkflowError("data must be a mapping")
        copied_data = dict(self.data)
        if any(not isinstance(key, str) or not key.strip() for key in copied_data):
            raise WorkflowError("data keys must be non-empty strings")
        self.data = copied_data


@dataclass(slots=True)
class WorkflowRuntimeContext:
    request_id: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, str] = field(default_factory=dict)
    token_budget: int | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    context_layer_manager: ContextLayerManager = field(
        default_factory=ContextLayerManager
    )
    trace_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise WorkflowError("request_id cannot be empty")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise WorkflowError("run_id cannot be empty")
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise WorkflowError("trace_id cannot be empty")
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id.strip()
        ):
            raise WorkflowError("session_id cannot be empty")
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise WorkflowError("created_at must be timezone-aware")
        if not isinstance(self.metadata, Mapping):
            raise WorkflowError("metadata must be a mapping")
        copied_metadata = dict(self.metadata)
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            for key, value in copied_metadata.items()
        ):
            raise WorkflowError("metadata must contain non-empty string keys and string values")
        self.metadata = copied_metadata
        if self.token_budget is not None and self.token_budget < 1:
            raise WorkflowError("token_budget must be positive")
        if not isinstance(self.token_usage, TokenUsage):
            raise WorkflowError("token_usage must be a TokenUsage")

    def record_usage(self, usage: TokenUsage | None) -> TokenUsage:
        if usage is None:
            return self.token_usage

        self.token_usage = TokenUsage(
            prompt_tokens=self.token_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self.token_usage.completion_tokens + usage.completion_tokens,
            total_tokens=self.token_usage.total_tokens + usage.total_tokens,
        )
        if (
            self.token_budget is not None
            and self.token_usage.total_tokens > self.token_budget
        ):
            raise WorkflowTokenBudgetExceededError(
                f"token budget exceeded: {self.token_usage.total_tokens} > "
                f"{self.token_budget}"
            )
        return self.token_usage

    def apply_usage(self, state: "WorkflowState") -> None:
        state.data["token_usage"] = self.token_usage.model_dump(mode="json")
        if self.token_budget is not None:
            state.data["token_budget"] = self.token_budget

    def prepare_context(
        self,
        state: "WorkflowState",
        messages: Sequence[ChatMessage],
    ) -> list[ChatMessage]:
        layers = self.context_layer_manager.build(messages)
        state.data["context_layers"] = layers.as_state()
        return [message.model_copy(deep=True) for message in layers.prompt_messages]


@runtime_checkable
class WorkflowNode(Protocol):
    name: str

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        """Apply one workflow step and return the updated state."""
