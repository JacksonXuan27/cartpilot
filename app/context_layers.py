from collections.abc import Sequence
from dataclasses import dataclass

from app.contracts import ChatMessage
from app.conversation_summary import ExtractiveConversationSummarizer


class ContextLayerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ContextLayers:
    system: tuple[ChatMessage, ...]
    long_term: tuple[ChatMessage, ...]
    short_term: tuple[ChatMessage, ...]
    summary: ChatMessage | None = None
    omitted_message_count: int = 0

    @property
    def prompt_messages(self) -> tuple[ChatMessage, ...]:
        summary = (self.summary,) if self.summary is not None else ()
        return self.system + summary + self.long_term + self.short_term

    def as_state(self) -> dict[str, object]:
        return {
            "system": [message.model_copy(deep=True) for message in self.system],
            "summary": (
                self.summary.model_copy(deep=True)
                if self.summary is not None
                else None
            ),
            "long_term": [
                message.model_copy(deep=True) for message in self.long_term
            ],
            "short_term": [
                message.model_copy(deep=True) for message in self.short_term
            ],
            "omitted_message_count": self.omitted_message_count,
        }


@dataclass(frozen=True, slots=True)
class ContextLayerManager:
    short_term_limit: int = 6
    long_term_limit: int = 20
    summarizer: ExtractiveConversationSummarizer = ExtractiveConversationSummarizer()

    def __post_init__(self) -> None:
        if self.short_term_limit < 1:
            raise ContextLayerError("short_term_limit must be positive")
        if self.long_term_limit < 0:
            raise ContextLayerError("long_term_limit cannot be negative")

    def build(self, messages: Sequence[ChatMessage]) -> ContextLayers:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise ContextLayerError("context messages must be a sequence")

        normalized = tuple(_copy_message(message) for message in messages)
        system = tuple(message for message in normalized if message.role == "system")
        conversation = tuple(
            message for message in normalized if message.role != "system"
        )
        short_term = conversation[-self.short_term_limit :]
        older = conversation[: -self.short_term_limit]
        long_term = older[-self.long_term_limit :] if self.long_term_limit else ()
        omitted_message_count = len(older) - len(long_term)
        summary = (
            self.summarizer.summarize(older[:omitted_message_count])
            if omitted_message_count
            else None
        )
        return ContextLayers(
            system=system,
            long_term=long_term,
            short_term=short_term,
            summary=summary,
            omitted_message_count=omitted_message_count,
        )


def _copy_message(message: ChatMessage) -> ChatMessage:
    if not isinstance(message, ChatMessage):
        raise ContextLayerError("context messages contain an invalid message")
    return message.model_copy(deep=True)
