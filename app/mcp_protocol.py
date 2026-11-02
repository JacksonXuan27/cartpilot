import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.tool_registry import ToolDefinition


JSONRPC_VERSION = "2.0"
MCP_TOOLS_LIST_METHOD = "tools/list"
MCP_TOOLS_CALL_METHOD = "tools/call"
MCP_PARSE_ERROR = -32700
MCP_INVALID_REQUEST = -32600
MCP_METHOD_NOT_FOUND = -32601
MCP_INVALID_PARAMS = -32602


class MCPProtocolError(ValueError):
    def __init__(self, message: str, *, code: int = MCP_INVALID_REQUEST) -> None:
        self.code = code
        super().__init__(message)


class MCPToolDescriptor(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(alias="inputSchema")
    output_schema: dict[str, Any] | None = Field(default=None, alias="outputSchema")

    @classmethod
    def from_definition(cls, definition: ToolDefinition) -> "MCPToolDescriptor":
        return cls(
            name=definition.name,
            description=definition.description,
            inputSchema=definition.input_schema,
        )


class MCPToolsListParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cursor: str | None = Field(default=None, min_length=1)


class MCPToolsCallParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal[JSONRPC_VERSION] = JSONRPC_VERSION
    id: int | str
    method: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def tools_list(cls, request_id: int | str, *, cursor: str | None = None) -> "MCPRequest":
        params = {} if cursor is None else {"cursor": cursor}
        return cls(id=request_id, method=MCP_TOOLS_LIST_METHOD, params=params)

    @classmethod
    def tools_call(
        cls,
        request_id: int | str,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> "MCPRequest":
        return cls(
            id=request_id,
            method=MCP_TOOLS_CALL_METHOD,
            params={"name": name, "arguments": arguments or {}},
        )

    def typed_params(self) -> MCPToolsListParams | MCPToolsCallParams:
        try:
            if self.method == MCP_TOOLS_LIST_METHOD:
                return MCPToolsListParams.model_validate(self.params)
            if self.method == MCP_TOOLS_CALL_METHOD:
                return MCPToolsCallParams.model_validate(self.params)
            raise MCPProtocolError(
                f"method not found: {self.method}",
                code=MCP_METHOD_NOT_FOUND,
            )
        except MCPProtocolError:
            raise
        except ValidationError as exc:
            raise MCPProtocolError(
                f"invalid parameters for {self.method}",
                code=MCP_INVALID_PARAMS,
            ) from exc


class MCPToolsListResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    tools: list[MCPToolDescriptor]
    next_cursor: str | None = Field(default=None, alias="nextCursor")

    @classmethod
    def from_definitions(
        cls,
        definitions: list[ToolDefinition],
        *,
        next_cursor: str | None = None,
    ) -> "MCPToolsListResult":
        return cls(
            tools=[MCPToolDescriptor.from_definition(item) for item in definitions],
            nextCursor=next_cursor,
        )


class MCPTextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class MCPCallToolResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    content: list[MCPTextContent] = Field(min_length=1)
    is_error: bool = Field(default=False, alias="isError")
    structured_content: dict[str, Any] | None = Field(
        default=None,
        alias="structuredContent",
    )

    @classmethod
    def from_output(cls, output: Any) -> "MCPCallToolResult":
        if isinstance(output, BaseModel):
            structured = output.model_dump(mode="json")
            rendered = json.dumps(structured, ensure_ascii=False, sort_keys=True)
        elif isinstance(output, dict):
            structured = output
            rendered = json.dumps(output, ensure_ascii=False, sort_keys=True)
        else:
            structured = None
            rendered = str(output)
        return cls(
            content=[MCPTextContent(text=rendered)],
            structuredContent=structured,
        )


class MCPError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: int
    message: str = Field(min_length=1)
    data: Any = None


class MCPResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal[JSONRPC_VERSION] = JSONRPC_VERSION
    id: int | str | None
    result: dict[str, Any] | None = None
    error: MCPError | None = None

    @model_validator(mode="after")
    def validate_result_or_error(self) -> "MCPResponse":
        if (self.result is None) == (self.error is None):
            raise ValueError("MCP response must contain exactly one of result or error")
        return self

    @classmethod
    def success(
        cls,
        request_id: int | str | None,
        result: BaseModel | dict[str, Any],
    ) -> "MCPResponse":
        payload = (
            result.model_dump(mode="json", by_alias=True, exclude_none=True)
            if isinstance(result, BaseModel)
            else result
        )
        return cls(id=request_id, result=payload)

    @classmethod
    def failure(
        cls,
        request_id: int | str | None,
        code: int,
        message: str,
        data: Any = None,
    ) -> "MCPResponse":
        return cls(
            id=request_id,
            error=MCPError(code=code, message=message, data=data),
        )


def encode_mcp_message(message: MCPRequest | MCPResponse) -> str:
    return json.dumps(
        message.model_dump(mode="json", by_alias=True, exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_mcp_request(payload: str | bytes) -> MCPRequest:
    decoded = _decode_json(payload)
    try:
        request = MCPRequest.model_validate(decoded)
        request.typed_params()
        return request
    except MCPProtocolError:
        raise
    except ValidationError as exc:
        raise MCPProtocolError("invalid MCP request", code=MCP_INVALID_REQUEST) from exc


def decode_mcp_response(payload: str | bytes) -> MCPResponse:
    decoded = _decode_json(payload)
    try:
        return MCPResponse.model_validate(decoded)
    except ValidationError as exc:
        raise MCPProtocolError("invalid MCP response", code=MCP_INVALID_REQUEST) from exc


def _decode_json(payload: str | bytes) -> Any:
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MCPProtocolError("invalid JSON", code=MCP_PARSE_ERROR) from exc
    if not isinstance(decoded, dict):
        raise MCPProtocolError("MCP message must be a JSON object")
    return decoded
