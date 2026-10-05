from collections.abc import AsyncIterator, Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.mcp_discovery import MCPToolDiscovery, MCPToolDiscoveryError
from app.mcp_server import MCPToolServer
from app.tool_registry import ToolRegistry


class DynamicInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class DynamicTool:
    name = "dynamic.echo"
    description = "Return a dynamically discovered value."
    input_model = DynamicInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, str]:
        query = self.input_model.model_validate(arguments)
        return {"value": query.value}


class AsyncSource:
    async def discover_tools(self) -> list[DynamicTool]:
        return [DynamicTool()]


class GeneratorSource:
    def discover_tools(self) -> AsyncIterator[DynamicTool]:
        async def tools() -> AsyncIterator[DynamicTool]:
            yield DynamicTool()

        return tools()


class FailingSource:
    async def discover_tools(self) -> list[DynamicTool]:
        raise RuntimeError("registry unavailable")


class InvalidSource:
    async def discover_tools(self) -> str:
        return "dynamic.echo"


@pytest.mark.asyncio
async def test_discovery_registers_async_tools_at_runtime():
    registry = ToolRegistry()
    discovery = MCPToolDiscovery(registry)

    names = await discovery.discover_and_register(AsyncSource())

    assert names == ("dynamic.echo",)
    assert [item.name for item in registry.definitions()] == ["dynamic.echo"]


@pytest.mark.asyncio
async def test_discovery_supports_async_generators():
    registry = ToolRegistry()

    names = await MCPToolDiscovery(registry).discover_and_register(GeneratorSource())

    assert names == ("dynamic.echo",)


@pytest.mark.asyncio
async def test_server_exposes_discovered_tools_to_tools_list():
    server = MCPToolServer(ToolRegistry())

    await server.discover_and_register(AsyncSource())
    response = await server.handle(
        '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
    )

    assert response.result is not None
    assert response.result["tools"][0]["name"] == "dynamic.echo"


@pytest.mark.asyncio
async def test_discovery_does_not_leave_partial_registration_on_duplicates():
    registry = ToolRegistry()

    class DuplicateSource:
        async def discover_tools(self) -> list[DynamicTool]:
            return [DynamicTool(), DynamicTool()]

    with pytest.raises(MCPToolDiscoveryError, match="already registered"):
        await MCPToolDiscovery(registry).discover_and_register(DuplicateSource())

    assert registry.definitions() == []


@pytest.mark.asyncio
async def test_discovery_wraps_source_and_payload_failures():
    registry = ToolRegistry()

    with pytest.raises(MCPToolDiscoveryError, match="discovery failed"):
        await MCPToolDiscovery(registry).discover_and_register(FailingSource())

    with pytest.raises(MCPToolDiscoveryError, match="tool objects"):
        await MCPToolDiscovery(registry).discover_and_register(InvalidSource())

    assert registry.definitions() == []
