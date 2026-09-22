from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.contracts import ChatMessage, TokenUsage


class ModelProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, object]
    call_id: str


@dataclass(frozen=True, slots=True)
class ModelResult:
    message: ChatMessage
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelChunk:
    delta: str
    finish_reason: str | None = None


@runtime_checkable
class ChatModelProvider(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[object] = (),
    ) -> ModelResult:
        """Generate one assistant message for a conversation."""

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ModelChunk]:
        """Yield incremental assistant message chunks for a conversation."""


@dataclass(slots=True)
class StubModelProvider:
    reply: str = "I can help with that."
    usage: TokenUsage | None = None
    chunk_size: int = 8
    tool_call_rounds: list[tuple[ToolCall, ...]] = field(default_factory=list)
    calls: list[tuple[ChatMessage, ...]] = field(default_factory=list)
    tool_definition_calls: list[tuple[object, ...]] = field(default_factory=list)
    stream_calls: list[tuple[ChatMessage, ...]] = field(default_factory=list)
    _tool_call_round_index: int = field(default=0, init=False, repr=False)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[object] = (),
    ) -> ModelResult:
        conversation = tuple(message.model_copy(deep=True) for message in messages)
        if not conversation:
            raise ModelProviderError("at least one message is required")

        self.calls.append(conversation)
        self.tool_definition_calls.append(tuple(tools))
        if self._tool_call_round_index < len(self.tool_call_rounds):
            tool_calls = self.tool_call_rounds[self._tool_call_round_index]
            self._tool_call_round_index += 1
            return ModelResult(
                message=ChatMessage(role="assistant", content="I will check that for you."),
                finish_reason="tool_call",
                tool_calls=tool_calls,
            )
        return ModelResult(
            message=ChatMessage(role="assistant", content=self.reply),
            usage=self.usage,
        )

    async def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ModelChunk]:
        conversation = tuple(message.model_copy(deep=True) for message in messages)
        if not conversation:
            raise ModelProviderError("at least one message is required")
        if self.chunk_size < 1:
            raise ModelProviderError("chunk_size must be positive")

        self.stream_calls.append(conversation)
        for offset in range(0, len(self.reply), self.chunk_size):
            yield ModelChunk(delta=self.reply[offset : offset + self.chunk_size])
        yield ModelChunk(delta="", finish_reason="stop")

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ModelChunk]:
        return self._stream(messages)
