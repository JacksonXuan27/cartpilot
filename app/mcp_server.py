import json

from app.mcp_discovery import MCPToolDiscovery, MCPToolSource
from app.mcp_executor import MCPToolExecutor
from app.mcp_protocol import (
    MCP_INVALID_PARAMS,
    MCPProtocolError,
    MCPRequest,
    MCPResponse,
    MCPToolsCallParams,
    MCPToolsListParams,
    MCPToolsListResult,
    decode_mcp_request,
)
from app.tool_registry import ToolRegistry


class MCPToolServer:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._tool_discovery = MCPToolDiscovery(tool_registry)
        self._tool_executor = MCPToolExecutor(tool_registry)

    async def discover_and_register(self, source: MCPToolSource) -> tuple[str, ...]:
        return await self._tool_discovery.discover_and_register(source)

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
            return await self._tool_executor.execute(params.name, params.arguments)

        raise MCPProtocolError("unsupported request parameters")


def encode_mcp_response(response: MCPResponse) -> str:
    return json.dumps(
        response.model_dump(mode="json", by_alias=True, exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
