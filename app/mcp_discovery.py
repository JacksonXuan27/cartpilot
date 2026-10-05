import inspect
from collections.abc import AsyncIterable, Awaitable, Iterable
from typing import Protocol

from app.tool_registry import ToolAlreadyRegisteredError, ToolLike, ToolRegistry


ToolCollection = Iterable[ToolLike] | AsyncIterable[ToolLike]


class MCPToolDiscoveryError(RuntimeError):
    pass


class MCPToolSource(Protocol):
    def discover_tools(
        self,
    ) -> ToolCollection | Awaitable[ToolCollection]:
        """Return tools exposed by a runtime MCP tool source."""


class MCPToolDiscovery:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    async def discover_and_register(self, source: MCPToolSource) -> tuple[str, ...]:
        try:
            discovered = source.discover_tools()
            if inspect.isawaitable(discovered):
                discovered = await discovered
            tools = await _materialize_tools(discovered)
            self._tool_registry.register_many(tools)
        except MCPToolDiscoveryError:
            raise
        except ToolAlreadyRegisteredError as exc:
            raise MCPToolDiscoveryError(
                f"discovered tool is already registered: {exc}"
            ) from exc
        except (TypeError, ValueError, AttributeError) as exc:
            raise MCPToolDiscoveryError(
                f"invalid discovered tools: {exc}"
            ) from exc
        except Exception as exc:
            raise MCPToolDiscoveryError("tool discovery failed") from exc
        return tuple(tool.name for tool in tools)


async def _materialize_tools(discovered: ToolCollection) -> list[ToolLike]:
    if isinstance(discovered, (str, bytes)):
        raise MCPToolDiscoveryError("tool source must return tool objects")
    if isinstance(discovered, AsyncIterable):
        return [tool async for tool in discovered]
    if isinstance(discovered, Iterable):
        return list(discovered)
    raise MCPToolDiscoveryError("tool source must return an iterable of tools")
