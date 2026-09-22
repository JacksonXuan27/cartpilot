from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MessageRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1)
    tool_call_id: str | None = Field(default=None, min_length=1, exclude=True)

    @model_validator(mode="after")
    def validate_tool_call_id(self) -> "ChatMessage":
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require a tool_call_id")
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


class AfterSalesExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    data: AfterSalesInfo


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


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
