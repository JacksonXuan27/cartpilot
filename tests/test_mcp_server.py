import json
from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.mcp_protocol import (
    MCP_INVALID_PARAMS,
    MCP_METHOD_NOT_FOUND,
    MCPCallToolResult,
    MCPRequest,
    MCPResponse,
    decode_mcp_response,
    encode_mcp_message,
)
from app.mcp_server import MCPToolServer, encode_mcp_response
from app.tool_registry import ToolRegistry


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)


class EchoTool:
    name = "echo"
    description = "Return the supplied message."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, str]:
        query = self.input_model.model_validate(arguments)
        return {"message": query.message}


class BrokenTool:
    name = "broken"
    description = "Always fail during execution."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> object:
        raise RuntimeError("downstream unavailable")


@pytest.fixture
def server() -> MCPToolServer:
    return MCPToolServer(ToolRegistry([EchoTool(), BrokenTool()]))


@pytest.mark.asyncio
async def test_server_lists_registered_business_tools(server: MCPToolServer):
    response = await server.handle(encode_mcp_message(MCPRequest.tools_list(1)))

    assert response.result is not None
    assert [tool["name"] for tool in response.result["tools"]] == ["broken", "echo"]
    assert response.error is None


@pytest.mark.asyncio
async def test_server_calls_tool_and_returns_structured_result(server: MCPToolServer):
    response = await server.handle(
        encode_mcp_message(
            MCPRequest.tools_call(2, "echo", {"message": "hello"})
        )
    )

    assert response.result == {
        "content": [{"type": "text", "text": '{"message": "hello"}'}],
        "isError": False,
        "structuredContent": {"message": "hello"},
    }


@pytest.mark.asyncio
async def test_server_converts_tool_failures_to_error_results(server: MCPToolServer):
    unknown = await server.handle(
        encode_mcp_message(MCPRequest.tools_call(3, "missing.tool"))
    )
    invalid = await server.handle(
        encode_mcp_message(MCPRequest.tools_call(4, "echo", {"message": ""}))
    )
    broken = await server.handle(
        encode_mcp_message(MCPRequest.tools_call(5, "broken", {"message": "run"}))
    )

    for response in (unknown, invalid, broken):
        assert response.error is None
        assert response.result is not None
        assert response.result["isError"] is True
        assert response.result["content"][0]["text"]


@pytest.mark.asyncio
async def test_server_returns_protocol_errors_for_bad_requests(server: MCPToolServer):
    invalid_params = await server.handle(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/list",
                "params": {"cursor": "unsupported"},
            }
        )
    )
    unknown_method = await server.handle(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "resources/list",
                "params": {},
            }
        )
    )

    assert invalid_params.error is not None
    assert invalid_params.error.code == MCP_INVALID_PARAMS
    assert unknown_method.error is not None
    assert unknown_method.error.code == MCP_METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_server_can_encode_and_decode_response_payload(server: MCPToolServer):
    response = await server.handle(encode_mcp_message(MCPRequest.tools_list("x")))
    decoded = decode_mcp_response(encode_mcp_response(response))

    assert decoded == MCPResponse.model_validate(response.model_dump())


def test_server_result_type_is_protocol_model():
    result = MCPCallToolResult(content=[{"type": "text", "text": "ok"}])

    assert result.is_error is False
