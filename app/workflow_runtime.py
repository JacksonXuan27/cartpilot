from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.after_sales import AfterSalesExtractor
from app.after_sales_intent import AfterSalesIntentNode
from app.after_sales_routing import AfterSalesRoutingNode
from app.contracts import ChatMessage
from app.context_reference_resolution import ContextReferenceResolutionNode
from app.intent_routing import IntentRouterNode
from app.providers import ChatModelProvider, StubModelProvider
from app.react_loop import ReactLoopNode
from app.refund_interruption import RefundInterruptionNode
from app.tool_registry import ToolRegistry
from app.workflow_checkpoints import (
    InMemoryWorkflowCheckpointStore,
    SQLiteWorkflowCheckpointStore,
    WorkflowCheckpoint,
    WorkflowCheckpointStore,
)
from app.database import DatabaseManager
from app.workflow import WorkflowNode, WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class WorkflowRuntimeError(ValueError):
    pass


class WorkflowRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)


class WorkflowResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_id: str = Field(min_length=1)
    confirmed: bool


class WorkflowRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: WorkflowStatus
    answer: str | None = None
    intent: str | None = None
    route: str | None = None
    iterations: int = Field(default=0, ge=0)
    error: dict[str, object] | None = None
    confirmation_required: bool = False
    confirmation_id: str | None = None
    confirmation_details: dict[str, object] | None = None


@dataclass(slots=True)
class WorkflowRuntime:
    model_provider: ChatModelProvider
    tool_registry: ToolRegistry
    nodes: tuple[WorkflowNode, ...] | None = None
    checkpoint_store: WorkflowCheckpointStore | None = None

    def __post_init__(self) -> None:
        if self.nodes is None:
            self.nodes = (
                IntentRouterNode(),
                ContextReferenceResolutionNode(),
                AfterSalesIntentNode(AfterSalesExtractor(self.model_provider)),
                AfterSalesRoutingNode(),
                RefundInterruptionNode(),
                ReactLoopNode(self.model_provider, self.tool_registry),
            )
        if self.checkpoint_store is None:
            self.checkpoint_store = InMemoryWorkflowCheckpointStore()
        if not self.nodes:
            raise WorkflowRuntimeError("workflow requires at least one node")
        names: list[str] = []
        for node in self.nodes:
            name = getattr(node, "name", None)
            if not isinstance(name, str) or not name.strip():
                raise WorkflowRuntimeError("workflow nodes require non-empty names")
            if name in names:
                raise WorkflowRuntimeError(f"duplicate workflow node: {name}")
            names.append(name)

    async def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> WorkflowState:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise WorkflowRuntimeError("workflow messages must be a sequence")
        if not messages:
            raise WorkflowRuntimeError("workflow requires at least one message")
        if any(not isinstance(message, ChatMessage) for message in messages):
            raise WorkflowRuntimeError("workflow messages contain an invalid message")

        context = WorkflowRuntimeContext(
            request_id=request_id or str(uuid4()),
            session_id=session_id,
        )
        state = WorkflowState(data={
            "messages": [message.model_copy(deep=True) for message in messages],
            "session_id": session_id,
            "request_id": context.request_id,
        })
        for node in self.nodes or ():
            try:
                state = await node.execute(state, context)
            except Exception as exc:
                state.status = WorkflowStatus.FAILED
                state.current_node = getattr(node, "name", None)
                state.data["error"] = {
                    "code": getattr(exc, "code", "workflow_failed"),
                    "message": str(exc),
                    **(
                        {"iteration": exc.iteration}
                        if isinstance(getattr(exc, "iteration", None), int)
                        else {}
                    ),
                }
                break
            if state.status is WorkflowStatus.AWAITING_CONFIRMATION:
                confirmation_id = state.data.get("confirmation_id")
                if not isinstance(confirmation_id, str):
                    state.status = WorkflowStatus.FAILED
                    state.data["error"] = {
                        "code": "checkpoint_save_failed",
                        "message": "confirmation checkpoint has no ID",
                    }
                    break
                try:
                    self.checkpoint_store.save_pending(
                        WorkflowCheckpoint(
                            confirmation_id=confirmation_id,
                            run_id=context.run_id,
                            state=state,
                        )
                    )
                except Exception as exc:
                    state.status = WorkflowStatus.FAILED
                    state.data["error"] = {
                        "code": "checkpoint_save_failed",
                        "message": str(exc),
                    }
                    state.data["confirmation_required"] = False
                break
        state.data["workflow_run_id"] = context.run_id
        return state

    async def resume(
        self,
        confirmation_id: str,
        *,
        confirmed: bool,
    ) -> WorkflowState:
        checkpoint = self.checkpoint_store.claim(confirmation_id)
        state = checkpoint.state
        context = WorkflowRuntimeContext(
            request_id=str(state.data.get("request_id") or uuid4()),
            session_id=(
                state.data.get("session_id")
                if isinstance(state.data.get("session_id"), str)
                else None
            ),
        )
        checkpoint.run_id = context.run_id
        state.data["workflow_run_id"] = context.run_id
        state.data["confirmation_required"] = False

        if not confirmed:
            state.status = WorkflowStatus.COMPLETED
            state.data["confirmation_status"] = "rejected"
            state.data["answer"] = "已取消此售后申请。"
            self.checkpoint_store.finish(checkpoint, "rejected")
            return state

        state.status = WorkflowStatus.RUNNING
        state.data["confirmation_status"] = "confirmed"
        state.data.pop("answer", None)
        state.current_node = "refund-confirmation-interrupt"
        nodes = self.nodes or ()
        resume_index = next(
            (
                index
                for index, node in enumerate(nodes)
                if node.name == "refund-confirmation-interrupt"
            ),
            None,
        )
        if resume_index is None:
            state.status = WorkflowStatus.FAILED
            state.data["error"] = {
                "code": "workflow_resume_failed",
                "message": "confirmation interrupt node is not configured",
            }
        else:
            for node in nodes[resume_index + 1 :]:
                try:
                    state = await node.execute(state, context)
                except Exception as exc:
                    state.status = WorkflowStatus.FAILED
                    state.current_node = node.name
                    state.data["error"] = {
                        "code": getattr(exc, "code", "workflow_failed"),
                        "message": str(exc),
                        **(
                            {"iteration": exc.iteration}
                            if isinstance(getattr(exc, "iteration", None), int)
                            else {}
                        ),
                    }
                    break
                if state.status is WorkflowStatus.AWAITING_CONFIRMATION:
                    break

        checkpoint.state = state
        try:
            self.checkpoint_store.finish(checkpoint, state.status.value)
        except Exception as exc:
            state.status = WorkflowStatus.FAILED
            state.data["error"] = {
                "code": "checkpoint_update_failed",
                "message": str(exc),
            }
        return state


def workflow_response(state: WorkflowState) -> WorkflowRunResponse:
    error = state.data.get("error")
    return WorkflowRunResponse(
        workflow_id=state.workflow_id,
        run_id=str(state.data.get("workflow_run_id", "unknown")),
        status=state.status,
        answer=state.data.get("answer") if isinstance(state.data.get("answer"), str) else None,
        intent=state.data.get("intent") if isinstance(state.data.get("intent"), str) else None,
        route=state.data.get("route") if isinstance(state.data.get("route"), str) else None,
        iterations=int(state.data.get("react_iterations", 0)),
        error=error if isinstance(error, dict) else None,
        confirmation_required=state.data.get("confirmation_required") is True,
        confirmation_id=(
            state.data.get("confirmation_id")
            if isinstance(state.data.get("confirmation_id"), str)
            else None
        ),
        confirmation_details=(
            state.data.get("confirmation_details")
            if isinstance(state.data.get("confirmation_details"), dict)
            else None
        ),
    )


def default_workflow_runtime() -> WorkflowRuntime:
    checkpoint_store = SQLiteWorkflowCheckpointStore(
        DatabaseManager("sqlite:///./data/cartpilot.db")
    )
    return WorkflowRuntime(
        StubModelProvider(),
        ToolRegistry(),
        checkpoint_store=checkpoint_store,
    )
