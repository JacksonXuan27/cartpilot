import asyncio
from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.mcp_executor import (
    InMemoryMCPToolAuditLog,
    MCPExecutionConfigurationError,
    MCPToolExecutor,
    RetryableToolError,
)
from app.tool_registry import ToolRegistry


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class EchoTool:
    name = "echo"
    description = "Return the supplied text."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, str]:
        query = self.input_model.model_validate(arguments)
        return {"text": query.text}


class FailingTool:
    name = "fail"
    description = "Raise a runtime failure."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> object:
        raise RuntimeError("internal failure")


class RetrySafeTool:
    name = "retry-safe"
    description = "Fails transiently once before succeeding."
    input_model = EchoInput
    retry_safe = True

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, str]:
        self.calls += 1
        if self.calls == 1:
            raise RetryableToolError("temporary failure")
        query = self.input_model.model_validate(arguments)
        return {"text": query.text}


class SlowTool:
    name = "slow"
    description = "Waits longer than the configured tool timeout."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> object:
        await asyncio.sleep(0.05)
        return {"text": "done"}


class RetryAfterTimeoutTool:
    name = "retry-timeout"
    description = "Times out once and then succeeds."
    input_model = EchoInput
    retry_safe = True

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, str]:
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0.02)
        query = self.input_model.model_validate(arguments)
        return {"text": query.text}


class RecoveringTool:
    name = "recovering"
    description = "Fails once and recovers after the circuit opens."
    input_model = EchoInput

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, str]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary outage")
        query = self.input_model.model_validate(arguments)
        return {"text": query.text}


@pytest.fixture
def executor() -> MCPToolExecutor:
    return MCPToolExecutor(ToolRegistry([EchoTool(), FailingTool()]))


@pytest.mark.asyncio
async def test_executor_returns_normalized_output(executor: MCPToolExecutor):
    result = await executor.execute("echo", {"text": "hello"})

    assert result.is_error is False
    assert result.structured_content == {"text": "hello"}
    assert result.content[0].text == '{"text": "hello"}'
    assert isinstance(executor.audit_sink, InMemoryMCPToolAuditLog)
    assert executor.audit_sink.entries()[0].status == "success"


@pytest.mark.asyncio
async def test_executor_normalizes_not_found_and_invalid_arguments(
    executor: MCPToolExecutor,
):
    missing = await executor.execute("missing", {})
    invalid = await executor.execute("echo", {"text": ""})

    assert missing.is_error is True
    assert "unknown tool" in missing.content[0].text
    assert invalid.is_error is True
    assert "invalid arguments" in invalid.content[0].text


@pytest.mark.asyncio
async def test_executor_hides_unexpected_runtime_error_details(
    executor: MCPToolExecutor,
):
    result = await executor.execute("fail", {"text": "run"})

    assert result.is_error is True
    assert result.content[0].text == "tool execution failed"
    assert "internal failure" not in result.content[0].text


@pytest.mark.asyncio
async def test_executor_times_out_slow_tool_and_records_metadata_only_audit():
    audit_log = InMemoryMCPToolAuditLog()
    executor = MCPToolExecutor(
        ToolRegistry([SlowTool()]),
        timeout_seconds=0.001,
        audit_sink=audit_log,
    )

    result = await executor.execute("slow", {"text": "private customer text"})

    assert result.is_error is True
    assert result.content[0].text == "tool execution timed out"
    entries = audit_log.entries()
    assert len(entries) == 1
    assert entries[0].tool_name == "slow"
    assert entries[0].status == "timed_out"
    assert entries[0].attempts == 1
    assert entries[0].duration_ms >= 0
    assert "private customer text" not in entries[0].model_dump_json()


@pytest.mark.asyncio
async def test_executor_retries_only_tools_marked_retry_safe():
    retry_safe = RetrySafeTool()
    unsafe = FailingTool()
    audit_log = InMemoryMCPToolAuditLog()
    executor = MCPToolExecutor(
        ToolRegistry([retry_safe, unsafe]),
        max_retries=1,
        retry_delay_seconds=0,
        audit_sink=audit_log,
    )

    recovered = await executor.execute("retry-safe", {"text": "hello"})
    failed = await executor.execute("fail", {"text": "hello"})

    assert recovered.is_error is False
    assert retry_safe.calls == 2
    assert failed.is_error is True
    assert [entry.attempts for entry in audit_log.entries()] == [2, 1]


@pytest.mark.asyncio
async def test_executor_retries_timeout_for_retry_safe_tool():
    tool = RetryAfterTimeoutTool()
    executor = MCPToolExecutor(
        ToolRegistry([tool]),
        timeout_seconds=0.005,
        max_retries=1,
        retry_delay_seconds=0,
    )

    result = await executor.execute("retry-timeout", {"text": "hello"})

    assert result.is_error is False
    assert tool.calls == 2


@pytest.mark.asyncio
async def test_executor_allows_one_half_open_probe_and_closes_after_recovery():
    tool = RecoveringTool()
    executor = MCPToolExecutor(
        ToolRegistry([tool]),
        circuit_failure_threshold=1,
        circuit_reset_seconds=0.01,
        retry_delay_seconds=0,
    )

    first = await executor.execute("recovering", {"text": "first"})
    assert first.is_error is True
    await asyncio.sleep(0.02)
    recovered = await executor.execute("recovering", {"text": "recovered"})
    following = await executor.execute("recovering", {"text": "following"})

    assert recovered.is_error is False
    assert following.is_error is False
    assert tool.calls == 3


@pytest.mark.asyncio
async def test_executor_opens_circuit_and_rejects_calls_until_reset():
    tool = FailingTool()
    audit_log = InMemoryMCPToolAuditLog()
    executor = MCPToolExecutor(
        ToolRegistry([tool]),
        circuit_failure_threshold=2,
        circuit_reset_seconds=60,
        audit_sink=audit_log,
    )

    first = await executor.execute("fail", {"text": "one"})
    second = await executor.execute("fail", {"text": "two"})
    third = await executor.execute("fail", {"text": "three"})

    assert first.is_error is True
    assert second.is_error is True
    assert third.content[0].text == "tool is temporarily unavailable"
    assert [entry.status for entry in audit_log.entries()] == [
        "failed",
        "failed",
        "circuit_open",
    ]


def test_executor_rejects_invalid_reliability_configuration():
    with pytest.raises(MCPExecutionConfigurationError):
        MCPToolExecutor(ToolRegistry(), timeout_seconds=0)

    with pytest.raises(MCPExecutionConfigurationError):
        MCPToolExecutor(ToolRegistry(), max_retries=-1)


def test_audit_log_keeps_only_the_most_recent_entries():
    from datetime import datetime, timezone

    from app.mcp_executor import MCPToolAuditRecord

    audit_log = InMemoryMCPToolAuditLog(max_entries=1)
    audit_log.record(
        MCPToolAuditRecord(
            tool_name="first",
            status="success",
            attempts=1,
            duration_ms=1,
            created_at=datetime.now(timezone.utc),
        )
    )
    audit_log.record(
        MCPToolAuditRecord(
            tool_name="second",
            status="success",
            attempts=1,
            duration_ms=1,
            created_at=datetime.now(timezone.utc),
        )
    )

    assert [entry.tool_name for entry in audit_log.entries()] == ["second"]
