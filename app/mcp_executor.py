from pydantic import ValidationError

from app.mcp_protocol import MCPCallToolResult
from app.tool_registry import ToolArgumentError, ToolNotFoundError, ToolRegistry


class MCPToolExecutor:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    async def execute(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> MCPCallToolResult:
        try:
            output = await self._tool_registry.execute(name, arguments)
        except ToolNotFoundError:
            return _tool_error(f"unknown tool: {name}")
        except ToolArgumentError as exc:
            return _tool_error(str(exc))
        except (LookupError, ValueError) as exc:
            return _tool_error(str(exc))
        except Exception:
            return _tool_error("tool execution failed")

        try:
            return MCPCallToolResult.from_output(output)
        except (TypeError, ValueError, ValidationError):
            return _tool_error("tool returned a result that cannot be serialized")


def _tool_error(message: str) -> MCPCallToolResult:
    return MCPCallToolResult(
        content=[{"type": "text", "text": message}],
        isError=True,
    )
