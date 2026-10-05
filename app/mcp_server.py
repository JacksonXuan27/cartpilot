import json

from pydantic import ValidationError

from app.mcp_protocol import (
    MCP_INVALID_PARAMS,
    MCPCallToolResult,
    MCPProtocolError,
    MCPRequest,
    MCPResponse,
    MCPToolsCallParams,
    MCPToolsListParams,
    MCPToolsListResult,
    decode_mcp_request,
)
from app.tool_registry import ToolArgumentError, ToolNotFoundError, ToolRegistry


class MCPToolServer:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    async def handle(self, payload: str | bytes) -> MCPResponse:
        request_id: int | str | None = None
        try:
            request = decode_mcp_request(payload)
            request_id = request.id
            result = await self._dispatch(request)
            return MCPResponse.success(request_id, result)
        except MCPProtocolError as exc:
            return MCPResponse.failure(request_id, exc.code, str(exc))

    async def _dispatch(self, request: MCPRequest) -> object:
        params = request.typed_params()
        if isinstance(params, MCPToolsListParams):
            if params.cursor is not None:
                raise MCPProtocolError(
                    "tool list pagination is not supported",
                    code=MCP_INVALID_PARAMS,
                )
            return MCPToolsListResult.from_definitions(
                self._tool_registry.definitions()
            )

        if isinstance(params, MCPToolsCallParams):
            return await self._call_tool(params)

        raise MCPProtocolError("unsupported request parameters")

    async def _call_tool(self, params: MCPToolsCallParams) -> MCPCallToolResult:
        try:
            output = await self._tool_registry.execute(params.name, params.arguments)
        except ToolNotFoundError:
            return _tool_error(f"unknown tool: {params.name}")
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


def encode_mcp_response(response: MCPResponse) -> str:
    return json.dumps(
        response.model_dump(mode="json", by_alias=True, exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
