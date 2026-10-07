import time
from collections import deque
from datetime import datetime, timezone
from threading import RLock
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
