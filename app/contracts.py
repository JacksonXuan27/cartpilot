from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


MessageRole = Literal["system", "user", "assistant", "tool"]


class ToolCallMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, object]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=0)
    tool_call_id: str | None = Field(default=None, min_length=1, exclude=True)
    _tool_calls: tuple[ToolCallMessage, ...] = PrivateAttr(default=())

    @property
    def tool_calls(self) -> tuple[ToolCallMessage, ...]:
        return self._tool_calls

    @classmethod
    def assistant_tool_call(
        cls, content: str, tool_calls: list[ToolCallMessage]
    ) -> "ChatMessage":
        validated_calls = tuple(
            ToolCallMessage.model_validate(call) for call in tool_calls
        )
        if not isinstance(content, str) or (not content.strip() and not validated_calls):
            raise ValueError("assistant messages require content or tool calls")
        message = cls.model_construct(role="assistant", content=content)
        message._tool_calls = validated_calls
        return message

    @model_validator(mode="after")
    def validate_tool_call_id(self) -> "ChatMessage":
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require a tool_call_id")
        if not self.content.strip():
            raise ValueError("message content cannot be blank")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    stream: bool = False


AfterSalesIntent = Literal[
    "refund",
    "return",
    "exchange",
    "repair",
    "logistics_issue",
    "other",
    "unknown",
]


class AfterSalesInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: AfterSalesIntent
    order_id: str | None = Field(default=None, min_length=1)
    reason: str | None = Field(default=None, min_length=1)
    requested_action: str | None = Field(default=None, min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AfterSalesExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class AfterSalesExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    data: AfterSalesInfo
    usage: TokenUsage | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_call"] = "stop"
    usage: TokenUsage | None = None


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail
