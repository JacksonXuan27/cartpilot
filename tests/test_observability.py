import pytest

from app.contracts import ChatMessage
from app.observability import InMemoryTraceRecorder
from app.providers import StubModelProvider
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus
from app.workflow_runtime import WorkflowRuntime


class CompleteNode:
    name = "test.complete"

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        state.status = WorkflowStatus.COMPLETED
        state.data["workflow_run_id"] = context.run_id
        return state


class FailingNode:
    name = "test.fail"

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        error = RuntimeError("private downstream detail")
        error.code = "test_node_failed"
        raise error


@pytest.mark.anyio
async def test_runtime_records_root_and_child_spans_without_message_content():
    recorder = InMemoryTraceRecorder()
    runtime = WorkflowRuntime(
        StubModelProvider(),
        ToolRegistry(),
        nodes=(CompleteNode(),),
        trace_recorder=recorder,
    )
    secret_message = "private customer message"

    state = await runtime.run(
        [ChatMessage(role="user", content=secret_message)],
        request_id="request-trace-1",
    )

    spans = recorder.spans(trace_id=state.data["trace_id"])
    assert len(spans) == 2
    root = next(span for span in spans if span.name == "workflow.run")
    child = next(span for span in spans if span.name == "workflow.node.test.complete")
    assert root.parent_span_id is None
    assert child.parent_span_id == root.span_id
    assert root.trace_id == child.trace_id == state.data["trace_id"]
    assert root.attributes["request_id"] == "request-trace-1"
    assert child.status == "completed"
    assert all(secret_message not in span.model_dump_json() for span in spans)
    assert all(span.duration_ms >= 0 for span in spans)


@pytest.mark.anyio
async def test_runtime_records_failed_node_and_root_status():
    recorder = InMemoryTraceRecorder()
    runtime = WorkflowRuntime(
        StubModelProvider(),
        ToolRegistry(),
        nodes=(FailingNode(),),
        trace_recorder=recorder,
    )

    state = await runtime.run([ChatMessage(role="user", content="hello")])

    spans = recorder.spans(trace_id=state.data["trace_id"])
    node_span = next(span for span in spans if span.name == "workflow.node.test.fail")
    root_span = next(span for span in spans if span.name == "workflow.run")
    assert state.status is WorkflowStatus.FAILED
    assert node_span.status == "error"
    assert node_span.attributes["error_code"] == "test_node_failed"
    assert root_span.status == "failed"


def test_trace_recorder_is_bounded_and_filters_by_trace_id():
    from datetime import datetime, timezone

    from app.observability import TraceSpan

    recorder = InMemoryTraceRecorder(max_spans=2)
    for index in range(3):
        recorder.record(
            TraceSpan(
                trace_id=f"trace-{index}",
                span_id=f"span-{index}",
                name="test.span",
                status="ok",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                duration_ms=1,
            )
        )

    assert [span.trace_id for span in recorder.spans()] == ["trace-1", "trace-2"]
    assert [span.span_id for span in recorder.spans(trace_id="trace-2")] == [
        "span-2"
    ]


def test_trace_context_keeps_legacy_positional_arguments():
    context = WorkflowRuntimeContext("request-1", "run-1", "session-1")

    assert context.session_id == "session-1"
    assert context.trace_id


@pytest.mark.anyio
async def test_confirmation_resume_keeps_trace_id_across_runs():
    recorder = InMemoryTraceRecorder()
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-TRACE",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    runtime = WorkflowRuntime(
        provider,
        ToolRegistry(),
        trace_recorder=recorder,
    )

    pending = await runtime.run(
        [ChatMessage(role="user", content="ORD-TRACE 到货破损，我要退款")]
    )
    trace_id = pending.data["trace_id"]
    assert pending.status is WorkflowStatus.AWAITING_CONFIRMATION

    resumed = await runtime.resume(
        str(pending.data["confirmation_id"]),
        confirmed=True,
    )

    spans = recorder.spans(trace_id=trace_id)
    roots = [span for span in spans if span.parent_span_id is None]
    assert resumed.data["trace_id"] == trace_id
    assert {span.name for span in roots} == {"workflow.run", "workflow.resume"}
    assert all(span.trace_id == trace_id for span in spans)
