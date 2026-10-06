import json
from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.mcp_protocol import (
    MCP_INVALID_PARAMS,
    MCPRequest,
    decode_mcp_response,
    encode_mcp_message,
)
from app.mcp_server import MCPToolServer, encode_mcp_response
from app.tool_registry import ToolRegistry


class CustomerLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=3)


class CustomerLookupTool:
    name = "customer.lookup"
    description = "Look up a customer profile by ID."
    input_model = CustomerLookupInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, str]:
        query = self.input_model.model_validate(arguments)
        return {"customer_id": query.customer_id, "tier": "standard"}


class CustomerToolSource:
    async def discover_tools(self) -> list[CustomerLookupTool]:
        return [CustomerLookupTool()]


@pytest.fixture
def mcp_server() -> MCPToolServer:
    return MCPToolServer(ToolRegistry())


async def round_trip(server: MCPToolServer, request: MCPRequest):
    request_payload = encode_mcp_message(request)
    response = await server.handle(request_payload)
    response_payload = encode_mcp_response(response)
    return decode_mcp_response(response_payload)


@pytest.mark.asyncio
async def test_dynamic_tool_discovery_list_and_call_round_trip(
    mcp_server: MCPToolServer,
):
    registered = await mcp_server.discover_and_register(CustomerToolSource())

    listed = await round_trip(mcp_server, MCPRequest.tools_list("list-1"))

    assert registered == ("customer.lookup",)
    assert listed.error is None
    assert listed.result is not None
    tools = listed.result["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "customer.lookup"
    assert tools[0]["description"] == "Look up a customer profile by ID."
    assert tools[0]["inputSchema"]["required"] == ["customer_id"]
    assert tools[0]["inputSchema"]["properties"]["customer_id"]["minLength"] == 3

    called = await round_trip(
        mcp_server,
        MCPRequest.tools_call(
            "call-1",
            "customer.lookup",
            {"customer_id": "CUS-3001"},
        ),
    )

    assert called.error is None
    assert called.result == {
        "content": [
            {
                "type": "text",
                "text": '{"customer_id": "CUS-3001", "tier": "standard"}',
            }
        ],
        "isError": False,
        "structuredContent": {"customer_id": "CUS-3001", "tier": "standard"},
    }


@pytest.mark.asyncio
async def test_tool_call_validation_error_is_returned_as_mcp_tool_error(
    mcp_server: MCPToolServer,
):
    await mcp_server.discover_and_register(CustomerToolSource())

    response = await round_trip(
        mcp_server,
        MCPRequest.tools_call(
            "call-invalid",
            "customer.lookup",
            {"customer_id": "x"},
        ),
    )

    assert response.error is None
    assert response.result is not None
    assert response.result["isError"] is True
    assert "invalid arguments" in response.result["content"][0]["text"]


@pytest.mark.asyncio
async def test_protocol_parameter_error_survives_full_response_round_trip(
    mcp_server: MCPToolServer,
):
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "invalid-list",
            "method": "tools/list",
            "params": {"cursor": "unsupported"},
        }
    )

    response = await mcp_server.handle(request)
    decoded = decode_mcp_response(encode_mcp_response(response))

    assert decoded.result is None
    assert decoded.error is not None
    assert decoded.error.code == MCP_INVALID_PARAMS
