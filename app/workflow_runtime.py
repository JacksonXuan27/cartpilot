from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.after_sales import AfterSalesExtractor
from app.after_sales_intent import AfterSalesIntentNode
from app.after_sales_routing import AfterSalesRoutingNode
from app.contracts import ChatMessage, TokenUsage
from app.context_layers import ContextLayerManager
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
from app.observability import (
    InMemoryModelMetricsRecorder,
    InMemoryTraceRecorder,
    ModelMetricsRecorder,
    ModelMetricsSummary,
    ModelPricing,
    TraceRecorder,
    observability_exporter_from_env,
    record_span,
    start_span,
)
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
    trace_id: str = Field(min_length=1)
    model_metrics: ModelMetricsSummary = Field(default_factory=ModelMetricsSummary)
    status: WorkflowStatus
    answer: str | None = None
    intent: str | None = None
    route: str | None = None
    iterations: int = Field(default=0, ge=0)
    error: dict[str, object] | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    token_budget: int | None = None
    confirmation_required: bool = False
    confirmation_id: str | None = None
    confirmation_details: dict[str, object] | None = None


@dataclass(slots=True)
class WorkflowRuntime:
    model_provider: ChatModelProvider
    tool_registry: ToolRegistry
    nodes: tuple[WorkflowNode, ...] | None = None
    checkpoint_store: WorkflowCheckpointStore | None = None
    token_budget: int | None = 2000
    context_layer_manager: ContextLayerManager | None = None
    trace_recorder: TraceRecorder | None = None
    metrics_recorder: ModelMetricsRecorder | None = None
    model_name: str = "configured-model"
    model_pricing: ModelPricing | None = None

    def __post_init__(self) -> None:
        if self.token_budget is not None and self.token_budget < 1:
            raise WorkflowRuntimeError("token_budget must be positive")
        if self.context_layer_manager is None:
            self.context_layer_manager = ContextLayerManager()
        if self.trace_recorder is None:
            self.trace_recorder = InMemoryTraceRecorder()
        if self.metrics_recorder is None:
            self.metrics_recorder = InMemoryModelMetricsRecorder()
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
            token_budget=self.token_budget,
            context_layer_manager=self.context_layer_manager,
            model_name=self.model_name,
            model_pricing=self.model_pricing,
            metrics_recorder=self.metrics_recorder,
        )
        state = WorkflowState(data={
            "messages": [message.model_copy(deep=True) for message in messages],
            "session_id": session_id,
            "request_id": context.request_id,
            "trace_id": context.trace_id,
        })
        context.apply_usage(state)
        root_span_id, root_started_at, root_started_monotonic = start_span()
        for node in self.nodes or ():
            try:
                state = await self._execute_traced_node(
                    node,
                    state,
                    context,
                    root_span_id,
                )
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
        record_span(
            self.trace_recorder,
            trace_id=context.trace_id,
            span_id=root_span_id,
            parent_span_id=None,
            name="workflow.run",
            status=state.status.value,
            started_at=root_started_at,
            started_monotonic=root_started_monotonic,
            attributes={
                "request_id": context.request_id,
                "run_id": context.run_id,
                "workflow_id": state.workflow_id,
            },
        )
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
            trace_id=str(state.data.get("trace_id") or uuid4()),
            session_id=(
                state.data.get("session_id")
                if isinstance(state.data.get("session_id"), str)
                else None
            ),
            token_budget=self.token_budget,
            token_usage=TokenUsage.model_validate(
                state.data.get("token_usage", {})
            ),
            context_layer_manager=self.context_layer_manager,
            model_name=self.model_name,
            model_pricing=self.model_pricing,
            metrics_recorder=self.metrics_recorder,
            model_metrics=ModelMetricsSummary.model_validate(
                state.data.get("model_metrics", {})
            ),
        )
        context.apply_usage(state)
        checkpoint.run_id = context.run_id
        state.data["workflow_run_id"] = context.run_id
        state.data["trace_id"] = context.trace_id
        state.data["confirmation_required"] = False
        root_span_id, root_started_at, root_started_monotonic = start_span()

        if not confirmed:
            state.status = WorkflowStatus.COMPLETED
            state.data["confirmation_status"] = "rejected"
            state.data["answer"] = "已取消此售后申请。"
            self.checkpoint_store.finish(checkpoint, "rejected")
            record_span(
                self.trace_recorder,
                trace_id=context.trace_id,
                span_id=root_span_id,
                parent_span_id=None,
                name="workflow.resume",
                status=state.status.value,
                started_at=root_started_at,
                started_monotonic=root_started_monotonic,
                attributes={"request_id": context.request_id, "run_id": context.run_id},
            )
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
                    state = await self._execute_traced_node(
                        node,
                        state,
                        context,
                        root_span_id,
                    )
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
        record_span(
            self.trace_recorder,
            trace_id=context.trace_id,
            span_id=root_span_id,
            parent_span_id=None,
            name="workflow.resume",
            status=state.status.value,
            started_at=root_started_at,
            started_monotonic=root_started_monotonic,
            attributes={
                "request_id": context.request_id,
                "run_id": context.run_id,
                "workflow_id": state.workflow_id,
            },
        )
        return state

    async def _execute_traced_node(
        self,
        node: WorkflowNode,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
        parent_span_id: str,
    ) -> WorkflowState:
        span_id, started_at, started_monotonic = start_span()
        status = "ok"
        attributes: dict[str, str | int | float | bool] = {
            "request_id": context.request_id,
            "run_id": context.run_id,
        }
        try:
            result = await node.execute(state, context)
            status = result.status.value
            return result
        except Exception as exc:
            status = "error"
            attributes["error_code"] = str(getattr(exc, "code", "workflow_failed"))
            raise
        finally:
            record_span(
                self.trace_recorder,
                trace_id=context.trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name=f"workflow.node.{node.name}",
                status=status,
                started_at=started_at,
                started_monotonic=started_monotonic,
                attributes=attributes,
            )


def workflow_response(state: WorkflowState) -> WorkflowRunResponse:
    error = state.data.get("error")
    return WorkflowRunResponse(
        workflow_id=state.workflow_id,
        run_id=str(state.data.get("workflow_run_id", "unknown")),
        trace_id=str(state.data.get("trace_id", "unknown")),
        status=state.status,
        answer=state.data.get("answer") if isinstance(state.data.get("answer"), str) else None,
        intent=state.data.get("intent") if isinstance(state.data.get("intent"), str) else None,
        route=state.data.get("route") if isinstance(state.data.get("route"), str) else None,
        iterations=int(state.data.get("react_iterations", 0)),
        error=error if isinstance(error, dict) else None,
        token_usage=TokenUsage.model_validate(state.data.get("token_usage", {})),
        model_metrics=ModelMetricsSummary.model_validate(
            state.data.get("model_metrics", {})
        ),
        token_budget=(
            state.data.get("token_budget")
            if isinstance(state.data.get("token_budget"), int)
            else None
        ),
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
    exporter = observability_exporter_from_env()
    return WorkflowRuntime(
        StubModelProvider(),
        ToolRegistry(),
        checkpoint_store=checkpoint_store,
        trace_recorder=exporter,
        metrics_recorder=exporter,
    )
