import json

import pytest
from pydantic import BaseModel, ValidationError

from app.mcp_protocol import (
    JSONRPC_VERSION,
    MCP_INVALID_PARAMS,
    MCP_INVALID_REQUEST,
    MCP_METHOD_NOT_FOUND,
    MCP_PARSE_ERROR,
    MCPProtocolError,
    MCPCallToolResult,
    MCPRequest,
    MCPResponse,
    MCPToolDescriptor,
    MCPToolsCallParams,
    MCPToolsListParams,
    MCPToolsListResult,
    decode_mcp_request,
    decode_mcp_response,
    encode_mcp_message,
)
from app.tool_registry import ToolDefinition


class ExampleOutput(BaseModel):
    order_id: str
    status: str


def test_tool_definition_maps_to_mcp_descriptor_with_protocol_aliases():
    definition = ToolDefinition(
        name="order.query",
        description="查询订单状态",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    )

    descriptor = MCPToolDescriptor.from_definition(definition)

    assert descriptor.name == "order.query"
    assert descriptor.input_schema["required"] == ["order_id"]
    assert descriptor.model_dump(by_alias=True, exclude_none=True) == {
        "name": "order.query",
        "description": "查询订单状态",
        "inputSchema": definition.input_schema,
    }

    with_output = MCPToolDescriptor(
        name="order.query",
        description="查询订单状态",
        inputSchema=definition.input_schema,
        outputSchema={"type": "object"},
    )
    assert with_output.model_dump(by_alias=True)["outputSchema"] == {
        "type": "object"
    }


def test_request_factories_encode_and_parse_typed_parameters():
    list_request = MCPRequest.tools_list("list-1", cursor="cursor-2")
    call_request = MCPRequest.tools_call(
        2,
        "order.query",
        {"order_id": "ORD-2001"},
    )

    assert list_request.typed_params() == MCPToolsListParams(cursor="cursor-2")
    assert call_request.typed_params() == MCPToolsCallParams(
        name="order.query",
        arguments={"order_id": "ORD-2001"},
    )

    decoded = decode_mcp_request(encode_mcp_message(call_request))
    assert decoded == call_request


def test_request_parameter_validation_reports_invalid_params():
    invalid_payloads = [
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": 1,
            "method": "tools/list",
            "params": {"cursor": ""},
        },
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": 2,
            "method": "tools/call",
            "params": {"name": "order.query", "arguments": []},
        },
    ]

    for payload in invalid_payloads:
        with pytest.raises(MCPProtocolError) as error:
            decode_mcp_request(json.dumps(payload))

        assert error.value.code == MCP_INVALID_PARAMS


def test_tools_list_result_converts_definitions():
    definitions = [
        ToolDefinition(
            name="faq.search",
            description="查询 FAQ",
            input_schema={"type": "object"},
        ),
        ToolDefinition(
            name="order.query",
            description="查询订单",
            input_schema={"type": "object"},
        ),
    ]

    result = MCPToolsListResult.from_definitions(
        definitions,
        next_cursor="cursor-3",
    )

    assert [tool.name for tool in result.tools] == ["faq.search", "order.query"]
    assert result.next_cursor == "cursor-3"
    assert result.model_dump(by_alias=True)["nextCursor"] == "cursor-3"


def test_call_tool_result_renders_structured_and_scalar_outputs():
    model_result = MCPCallToolResult.from_output(
        ExampleOutput(order_id="ORD-2001", status="shipped")
    )
    assert model_result.structured_content == {
        "order_id": "ORD-2001",
        "status": "shipped",
    }
    assert json.loads(model_result.content[0].text) == model_result.structured_content

    dictionary_result = MCPCallToolResult.from_output({"count": 2})
    assert dictionary_result.structured_content == {"count": 2}
    assert dictionary_result.content[0].text == '{"count": 2}'

    scalar_result = MCPCallToolResult.from_output("没有匹配订单")
    assert scalar_result.structured_content is None
    assert scalar_result.content[0].text == "没有匹配订单"


def test_success_and_failure_responses_round_trip_through_json():
    success = MCPResponse.success(
        "request-1",
        MCPToolsListResult.from_definitions([]),
    )
    success_decoded = decode_mcp_response(encode_mcp_message(success))
    assert success_decoded.result == {"tools": []}
    assert success_decoded.error is None

    failure = MCPResponse.failure(
        "request-2",
        -32001,
        "tool execution failed",
        {"tool": "order.query"},
    )
    failure_decoded = decode_mcp_response(encode_mcp_message(failure))
    assert failure_decoded.result is None
    assert failure_decoded.error is not None
    assert failure_decoded.error.code == -32001
    assert failure_decoded.error.data == {"tool": "order.query"}


def test_response_requires_exactly_one_of_result_or_error():
    with pytest.raises(ValidationError):
        MCPResponse.model_validate(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": 1,
                "result": {},
                "error": {"code": -32000, "message": "failed"},
            }
        )

    with pytest.raises(ValidationError):
        MCPResponse.model_validate(
            {"jsonrpc": JSONRPC_VERSION, "id": 1}
        )


def test_decode_rejects_malformed_json_non_object_and_invalid_request_fields():
    with pytest.raises(MCPProtocolError) as parse_error:
        decode_mcp_request("not-json")
    assert parse_error.value.code == MCP_PARSE_ERROR

    with pytest.raises(MCPProtocolError) as object_error:
        decode_mcp_request("[]")
    assert object_error.value.code == MCP_INVALID_REQUEST

    invalid_payloads = [
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": 1,
            "method": "tools/unknown",
            "params": {},
        },
        {
            "jsonrpc": "1.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        },
    ]
    for payload in invalid_payloads:
        with pytest.raises(MCPProtocolError) as request_error:
            decode_mcp_request(json.dumps(payload))
        expected_code = (
            MCP_METHOD_NOT_FOUND
            if payload["method"] == "tools/unknown"
            else MCP_INVALID_REQUEST
        )
        assert request_error.value.code == expected_code


def test_decode_rejects_invalid_response_payload():
    with pytest.raises(MCPProtocolError) as error:
        decode_mcp_response(
            json.dumps(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": 1,
                    "result": {},
                    "error": {"code": -32000, "message": "failed"},
                }
            )
        )

    assert error.value.code == MCP_INVALID_REQUEST
