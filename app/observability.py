from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone
from queue import Full, Queue
from threading import Event, RLock, Thread
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_usd_per_million_tokens
            + completion_tokens * self.output_usd_per_million_tokens
        ) / 1_000_000


class ModelMetricsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_count: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    priced_call_count: int = Field(default=0, ge=0)
    unpriced_call_count: int = Field(default=0, ge=0)


class ModelCallMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    duration_ms: float = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    created_at: datetime


class ModelMetricsRecorder(Protocol):
    def record(self, metric: ModelCallMetric) -> None:
        """Record model usage and latency without prompt or response content."""


class ObservabilityTransport(Protocol):
    def send(self, payload: dict[str, object]) -> None:
        """Send one observability event to an external service."""


class ObservabilityExportError(RuntimeError):
    pass


class HttpJsonTransport:
    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 1.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        parsed_endpoint = urllib.parse.urlparse(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
            raise ValueError("observability endpoint must be an HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": "cartpilot-observability/1.0",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self._headers.update(headers)

    def send(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if not 200 <= response.status < 300:
                    raise ObservabilityExportError(
                        f"observability service returned HTTP {response.status}"
                    )
        except urllib.error.HTTPError as exc:
            raise ObservabilityExportError(
                f"observability service returned HTTP {exc.code}"
            ) from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ObservabilityExportError(
                "observability service request failed"
            ) from exc


class HttpObservabilityExporter:
    def __init__(
        self,
        transport: ObservabilityTransport,
        *,
        service_name: str = "cartpilot",
        queue_size: int = 1000,
    ) -> None:
        if not service_name.strip():
            raise ValueError("service_name cannot be empty")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.transport = transport
        self.service_name = service_name
        self._events: Queue[dict[str, object] | None] = Queue(maxsize=queue_size)
        self._closed = Event()
        self._lock = RLock()
        self._dropped_events = 0
        self._failed_events = 0
        self._worker = Thread(
            target=self._run,
            name=f"{service_name}-observability-exporter",
            daemon=True,
        )
        self._worker.start()

    def record(self, item: TraceSpan | ModelCallMetric) -> None:
        event = self._build_event(item)
        with self._lock:
            if self._closed.is_set():
                return
            try:
                self._events.put_nowait(event)
            except Full:
                self._dropped_events += 1

    def flush(self) -> None:
        self._events.join()

    def close(self) -> None:
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
        self.flush()
        self._events.put(None)
        self._worker.join(timeout=1.0)

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped_events

    @property
    def failed_events(self) -> int:
        with self._lock:
            return self._failed_events

    def _build_event(self, item: TraceSpan | ModelCallMetric) -> dict[str, object]:
        if isinstance(item, TraceSpan):
            return {
                "service": self.service_name,
                "event_type": "trace.span",
                "trace_id": item.trace_id,
                "span_id": item.span_id,
                "parent_span_id": item.parent_span_id,
                "name": item.name,
                "status": item.status,
                "started_at": item.started_at.isoformat(),
                "ended_at": item.ended_at.isoformat(),
                "duration_ms": item.duration_ms,
                "attributes": dict(item.attributes),
            }
        if isinstance(item, ModelCallMetric):
            return {
                "service": self.service_name,
                "event_type": "model.call",
                "trace_id": item.trace_id,
                "run_id": item.run_id,
                "operation": item.operation,
                "model_name": item.model_name,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "duration_ms": item.duration_ms,
                "estimated_cost_usd": item.estimated_cost_usd,
                "created_at": item.created_at.isoformat(),
            }
        raise TypeError("observability exporter accepts TraceSpan or ModelCallMetric")

    def _run(self) -> None:
        while True:
            event = self._events.get()
            try:
                if event is None:
                    return
                try:
                    self.transport.send(event)
                except Exception:
                    with self._lock:
                        self._failed_events += 1
            finally:
                self._events.task_done()


def observability_exporter_from_env() -> HttpObservabilityExporter | None:
    endpoint = os.getenv("OBSERVABILITY_EXPORT_URL", "").strip()
    if not endpoint:
        return None
    timeout_seconds = float(os.getenv("OBSERVABILITY_TIMEOUT_SECONDS", "1.0"))
    queue_size = int(os.getenv("OBSERVABILITY_QUEUE_SIZE", "1000"))
    return HttpObservabilityExporter(
        HttpJsonTransport(
            endpoint,
            api_key=os.getenv("OBSERVABILITY_API_KEY") or None,
            timeout_seconds=timeout_seconds,
        ),
        service_name=os.getenv("OBSERVABILITY_SERVICE_NAME", "cartpilot"),
        queue_size=queue_size,
    )


class InMemoryModelMetricsRecorder:
    def __init__(self, max_metrics: int = 5000) -> None:
        if max_metrics < 1:
            raise ValueError("max_metrics must be positive")
        self._metrics: deque[ModelCallMetric] = deque(maxlen=max_metrics)
        self._lock = RLock()

    def record(self, metric: ModelCallMetric) -> None:
        with self._lock:
            self._metrics.append(metric.model_copy(deep=True))

    def metrics(self, *, trace_id: str | None = None) -> tuple[ModelCallMetric, ...]:
        with self._lock:
            selected = (
                self._metrics
                if trace_id is None
                else (metric for metric in self._metrics if metric.trace_id == trace_id)
            )
            return tuple(metric.model_copy(deep=True) for metric in selected)


class TraceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    parent_span_id: str | None = None
    name: str = Field(min_length=1)
    status: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime
    duration_ms: float = Field(ge=0)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class TraceRecorder(Protocol):
    def record(self, span: TraceSpan) -> None:
        """Record a completed trace span."""


class InMemoryTraceRecorder:
    def __init__(self, max_spans: int = 5000) -> None:
        if max_spans < 1:
            raise ValueError("max_spans must be positive")
        self._spans: deque[TraceSpan] = deque(maxlen=max_spans)
        self._lock = RLock()

    def record(self, span: TraceSpan) -> None:
        with self._lock:
            self._spans.append(span.model_copy(deep=True))

    def spans(self, *, trace_id: str | None = None) -> tuple[TraceSpan, ...]:
        with self._lock:
            selected = (
                self._spans
                if trace_id is None
                else (span for span in self._spans if span.trace_id == trace_id)
            )
            return tuple(span.model_copy(deep=True) for span in selected)


def start_span() -> tuple[str, datetime, float]:
    return str(uuid4()), datetime.now(timezone.utc), time.monotonic()


def record_span(
    recorder: TraceRecorder,
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    status: str,
    started_at: datetime,
    started_monotonic: float,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> None:
    ended_at = datetime.now(timezone.utc)
    span = TraceSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000),
        attributes=attributes or {},
    )
    try:
        recorder.record(span)
    except Exception:
        return
