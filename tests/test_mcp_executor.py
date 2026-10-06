from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.mcp_executor import MCPToolExecutor
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


@pytest.fixture
def executor() -> MCPToolExecutor:
    return MCPToolExecutor(ToolRegistry([EchoTool(), FailingTool()]))


@pytest.mark.asyncio
async def test_executor_returns_normalized_output(executor: MCPToolExecutor):
    result = await executor.execute("echo", {"text": "hello"})

    assert result.is_error is False
    assert result.structured_content == {"text": "hello"}
    assert result.content[0].text == '{"text": "hello"}'


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
