from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.contracts import ChatMessage, TokenUsage


class ModelProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelResult:
    message: ChatMessage
    finish_reason: str = "stop"
    usage: TokenUsage | None = None


@runtime_checkable
class ChatModelProvider(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> ModelResult:
        """Generate one assistant message for a conversation."""


@dataclass(slots=True)
class StubModelProvider:
    reply: str = "I can help with that."
    usage: TokenUsage | None = None
    calls: list[tuple[ChatMessage, ...]] = field(default_factory=list)

    async def complete(self, messages: Sequence[ChatMessage]) -> ModelResult:
        conversation = tuple(message.model_copy(deep=True) for message in messages)
        if not conversation:
            raise ModelProviderError("at least one message is required")

        self.calls.append(conversation)
        return ModelResult(
            message=ChatMessage(role="assistant", content=self.reply),
            usage=self.usage,
        )
