import json
from datetime import datetime, timezone
from threading import Event

import pytest

from app.contracts import ChatMessage, TokenUsage
from app.observability import (
    HttpObservabilityExporter,
    InMemoryModelMetricsRecorder,
    InMemoryTraceRecorder,
    ModelCallMetric,
    ModelPricing,
    TraceSpan,
)
from app.providers import StubModelProvider
from app.react_loop import ReactLoopNode
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


class RecordingTransport:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def send(self, payload: dict[str, object]) -> None:
        self.payloads.append(payload)


class FailingTransport:
    def send(self, payload: dict[str, object]) -> None:
        raise RuntimeError("observability unavailable")


class BlockingTransport:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def send(self, payload: dict[str, object]) -> None:
        self.started.set()
        self.release.wait(timeout=2)


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


@pytest.mark.anyio
async def test_runtime_reports_token_latency_and_configured_cost_metrics():
    metrics_recorder = InMemoryModelMetricsRecorder()
    provider = StubModelProvider(
        reply="回复完成",
        usage=TokenUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150),
    )
    runtime = WorkflowRuntime(
        provider,
        ToolRegistry(),
        nodes=(
            ReactLoopNode(provider, ToolRegistry()),
        ),
        metrics_recorder=metrics_recorder,
        model_name="test-model",
        model_pricing=ModelPricing(
            input_usd_per_million_tokens=2.0,
            output_usd_per_million_tokens=4.0,
        ),
    )

    state = await runtime.run([ChatMessage(role="user", content="hello")])

    metrics = state.data["model_metrics"]
    assert metrics["call_count"] == 1
    assert metrics["priced_call_count"] == 1
    assert metrics["unpriced_call_count"] == 0
    assert metrics["estimated_cost_usd"] == pytest.approx(0.00036)
    assert metrics["duration_ms"] >= 0
    assert state.data["token_usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }
    recorded = metrics_recorder.metrics(trace_id=state.data["trace_id"])
    assert len(recorded) == 1
    assert recorded[0].model_name == "test-model"
    assert recorded[0].operation == "react_loop.complete"
    assert recorded[0].total_tokens == 150
    assert recorded[0].estimated_cost_usd == pytest.approx(0.00036)


@pytest.mark.anyio
async def test_runtime_does_not_invent_cost_without_pricing():
    provider = StubModelProvider(
        reply="回复完成",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    runtime = WorkflowRuntime(provider, ToolRegistry())

    state = await runtime.run([ChatMessage(role="user", content="hello")])

    metrics = state.data["model_metrics"]
    assert metrics["estimated_cost_usd"] is None
    assert metrics["priced_call_count"] == 0
    assert metrics["unpriced_call_count"] == 1


@pytest.mark.anyio
async def test_metrics_accumulate_across_confirmation_resume():
    metrics_recorder = InMemoryModelMetricsRecorder()
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-METRIC",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        ),
        usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
    )
    pricing = ModelPricing(
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
    )
    runtime = WorkflowRuntime(
        provider,
        ToolRegistry(),
        metrics_recorder=metrics_recorder,
        model_pricing=pricing,
    )

    pending = await runtime.run(
        [ChatMessage(role="user", content="ORD-METRIC 到货破损，我要退款")]
    )
    resumed = await runtime.resume(
        str(pending.data["confirmation_id"]),
        confirmed=True,
    )

    metrics = resumed.data["model_metrics"]
    assert metrics["call_count"] == 2
    assert metrics["priced_call_count"] == 2
    assert metrics["estimated_cost_usd"] == pytest.approx(0.00006)
    recorded = metrics_recorder.metrics(trace_id=pending.data["trace_id"])
    assert len(recorded) == 2


def test_model_pricing_rejects_negative_rates():
    with pytest.raises(ValueError):
        ModelPricing(
            input_usd_per_million_tokens=-1,
            output_usd_per_million_tokens=0,
        )


def test_http_exporter_sends_trace_and_model_metadata_without_message_content():
    transport = RecordingTransport()
    exporter = HttpObservabilityExporter(transport)
    try:
        exporter.record(
            TraceSpan(
                trace_id="trace-export",
                span_id="span-export",
                name="workflow.run",
                status="completed",
                started_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
                ended_at=datetime(2026, 10, 7, 0, 0, 0, 1000, tzinfo=timezone.utc),
                duration_ms=1,
                attributes={"request_id": "request-export"},
            )
        )
        exporter.record(
            ModelCallMetric(
                trace_id="trace-export",
                run_id="run-export",
                operation="react_loop.complete",
                model_name="test-model",
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                duration_ms=4,
                estimated_cost_usd=0.0001,
                created_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
            )
        )
        exporter.flush()
    finally:
        exporter.close()

    assert [payload["event_type"] for payload in transport.payloads] == [
        "trace.span",
        "model.call",
    ]
    serialized = json.dumps(transport.payloads, ensure_ascii=False)
    assert "request-export" in serialized
    assert "customer message" not in serialized
    assert transport.payloads[1]["total_tokens"] == 20


def test_http_exporter_does_not_propagate_transport_failures():
    exporter = HttpObservabilityExporter(FailingTransport())
    try:
        exporter.record(
            TraceSpan(
                trace_id="trace-failed-export",
                span_id="span-failed-export",
                name="workflow.run",
                status="failed",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                duration_ms=0,
            )
        )
        exporter.flush()
        assert exporter.failed_events == 1
    finally:
        exporter.close()


def test_http_exporter_drops_events_instead_of_blocking_when_queue_is_full():
    transport = BlockingTransport()
    exporter = HttpObservabilityExporter(transport, queue_size=1)
    span = TraceSpan(
        trace_id="trace-backpressure",
        span_id="span-backpressure",
        name="workflow.run",
        status="completed",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        duration_ms=0,
    )
    try:
        exporter.record(span)
        assert transport.started.wait(timeout=1)
        exporter.record(span.model_copy(update={"span_id": "span-queued"}))
        exporter.record(span.model_copy(update={"span_id": "span-dropped"}))
        assert exporter.dropped_events == 1
        transport.release.set()
        exporter.flush()
    finally:
        transport.release.set()
        exporter.close()
