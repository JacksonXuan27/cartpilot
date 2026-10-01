from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.after_sales import AfterSalesExtractor
from app.after_sales_intent import AfterSalesIntentNode
from app.contracts import ChatMessage
from app.context_reference_resolution import ContextReferenceResolutionNode
from app.intent_routing import IntentRouterNode
from app.providers import ChatModelProvider, StubModelProvider
from app.react_loop import ReactLoopNode
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowNode, WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class WorkflowRuntimeError(ValueError):
    pass


class WorkflowRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)


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


@dataclass(slots=True)
class WorkflowRuntime:
    model_provider: ChatModelProvider
    tool_registry: ToolRegistry
    nodes: tuple[WorkflowNode, ...] | None = None

    def __post_init__(self) -> None:
        if self.nodes is None:
            self.nodes = (
                IntentRouterNode(),
                ContextReferenceResolutionNode(),
                AfterSalesIntentNode(AfterSalesExtractor(self.model_provider)),
                ReactLoopNode(self.model_provider, self.tool_registry),
            )
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

        context = WorkflowRuntimeContext(request_id=request_id or str(uuid4()), session_id=session_id)
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
                    **({"iteration": exc.iteration} if isinstance(getattr(exc, "iteration", None), int) else {}),
                }
                break
        state.data["workflow_run_id"] = context.run_id
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
    )


def default_workflow_runtime() -> WorkflowRuntime:
    return WorkflowRuntime(StubModelProvider(), ToolRegistry())
