import asyncio
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.mcp_protocol import MCPCallToolResult
from app.tool_registry import ToolArgumentError, ToolNotFoundError, ToolRegistry


class RetryableToolError(RuntimeError):
    pass


class MCPExecutionConfigurationError(ValueError):
    pass


class MCPToolAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    status: str = Field(min_length=1)
    attempts: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    error_type: str | None = None
    created_at: datetime


class MCPToolAuditSink(Protocol):
    def record(self, entry: MCPToolAuditRecord) -> None:
        """Persist a metadata-only tool execution audit record."""


class InMemoryMCPToolAuditLog:
    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries < 1:
            raise MCPExecutionConfigurationError("max_entries must be positive")
        self._entries: deque[MCPToolAuditRecord] = deque(maxlen=max_entries)
        self._lock = RLock()

    def record(self, entry: MCPToolAuditRecord) -> None:
        with self._lock:
            self._entries.append(entry.model_copy(deep=True))

    def entries(self) -> tuple[MCPToolAuditRecord, ...]:
        with self._lock:
            return tuple(entry.model_copy(deep=True) for entry in self._entries)


@dataclass(slots=True)
class _CircuitState:
    failure_count: int = 0
    opened_at: float | None = None
    probe_in_progress: bool = False


@dataclass(slots=True)
class MCPToolExecutor:
    tool_registry: ToolRegistry
    timeout_seconds: float = 5.0
    max_retries: int = 0
    retry_delay_seconds: float = 0.05
    circuit_failure_threshold: int = 3
    circuit_reset_seconds: float = 30.0
    audit_sink: MCPToolAuditSink = field(default_factory=InMemoryMCPToolAuditLog)
    _circuits: dict[str, _CircuitState] = field(default_factory=dict, init=False)
    _circuit_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise MCPExecutionConfigurationError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise MCPExecutionConfigurationError("max_retries cannot be negative")
        if self.retry_delay_seconds < 0:
            raise MCPExecutionConfigurationError(
                "retry_delay_seconds cannot be negative"
            )
        if self.circuit_failure_threshold < 1:
            raise MCPExecutionConfigurationError(
                "circuit_failure_threshold must be positive"
            )
        if self.circuit_reset_seconds <= 0:
            raise MCPExecutionConfigurationError(
                "circuit_reset_seconds must be positive"
            )

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> MCPCallToolResult:
        started_at = time.monotonic()
        try:
            tool = self.tool_registry.get(name)
        except ToolNotFoundError:
            self._audit(name, "not_found", 0, started_at, "ToolNotFoundError")
            return _tool_error(f"unknown tool: {name}")

        allowed, probe = await self._acquire_circuit(name)
        if not allowed:
            self._audit(name, "circuit_open", 0, started_at, "CircuitOpen")
            return _tool_error("tool is temporarily unavailable")

        retry_safe = bool(getattr(tool, "retry_safe", False))
        retry_limit = self.max_retries if retry_safe else 0
        attempts = 0
        while True:
            attempts += 1
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    output = await self.tool_registry.execute(name, arguments)
                result = MCPCallToolResult.from_output(output)
                await self._record_success(name)
                self._audit(name, "success", attempts, started_at)
                return result
            except (ToolArgumentError, LookupError, ValueError) as exc:
                await self._release_probe(name, probe)
                self._audit(name, "rejected", attempts, started_at, type(exc).__name__)
                return _tool_error(str(exc))
            except asyncio.CancelledError:
                await self._release_probe(name, probe)
                self._audit(name, "cancelled", attempts, started_at, "CancelledError")
                raise
            except (TimeoutError, RetryableToolError, TypeError, ValidationError) as exc:
                retryable_failure = isinstance(
                    exc,
                    (TimeoutError, RetryableToolError),
                )
                if retryable_failure and retry_safe and attempts <= retry_limit:
                    if self.retry_delay_seconds:
                        try:
                            await asyncio.sleep(self.retry_delay_seconds)
                        except asyncio.CancelledError:
                            await self._release_probe(name, probe)
                            self._audit(
                                name,
                                "cancelled",
                                attempts,
                                started_at,
                                "CancelledError",
                            )
                            raise
                    continue
                await self._record_failure(name, probe)
                status = "timed_out" if isinstance(exc, TimeoutError) else "failed"
                self._audit(name, status, attempts, started_at, type(exc).__name__)
                message = (
                    "tool execution timed out"
                    if isinstance(exc, TimeoutError)
                    else "tool execution failed"
                )
                return _tool_error(message)
            except Exception as exc:
                await self._record_failure(name, probe)
                self._audit(name, "failed", attempts, started_at, type(exc).__name__)
                return _tool_error("tool execution failed")

    async def _acquire_circuit(self, name: str) -> tuple[bool, bool]:
        async with self._circuit_lock:
            state = self._circuits.setdefault(name, _CircuitState())
            if state.opened_at is None:
                return True, False
            if time.monotonic() - state.opened_at < self.circuit_reset_seconds:
                return False, False
            if state.probe_in_progress:
                return False, False
            state.probe_in_progress = True
            return True, True

    async def _record_success(self, name: str) -> None:
        async with self._circuit_lock:
            self._circuits[name] = _CircuitState()

    async def _record_failure(self, name: str, probe: bool) -> None:
        async with self._circuit_lock:
            state = self._circuits.setdefault(name, _CircuitState())
            state.probe_in_progress = False
            if probe or state.failure_count + 1 >= self.circuit_failure_threshold:
                state.failure_count = max(
                    self.circuit_failure_threshold,
                    state.failure_count + 1,
                )
                state.opened_at = time.monotonic()
            else:
                state.failure_count += 1

    async def _release_probe(self, name: str, probe: bool) -> None:
        if not probe:
            return
        async with self._circuit_lock:
            state = self._circuits.get(name)
            if state is not None:
                state.probe_in_progress = False

    def _audit(
        self,
        name: str,
        status: str,
        attempts: int,
        started_at: float,
        error_type: str | None = None,
    ) -> None:
        entry = MCPToolAuditRecord(
            tool_name=name,
            status=status,
            attempts=attempts,
            duration_ms=max(0.0, (time.monotonic() - started_at) * 1000),
            error_type=error_type,
            created_at=datetime.now(timezone.utc),
        )
        try:
            self.audit_sink.record(entry)
        except Exception:
            return


def _tool_error(message: str) -> MCPCallToolResult:
    return MCPCallToolResult(
        content=[{"type": "text", "text": message}],
        isError=True,
    )
