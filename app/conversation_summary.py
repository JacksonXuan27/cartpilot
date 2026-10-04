from collections.abc import Sequence
from dataclasses import dataclass

from app.contracts import ChatMessage


class ConversationSummaryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExtractiveConversationSummarizer:
    max_message_characters: int = 240
    max_summary_characters: int = 1200

    def __post_init__(self) -> None:
        if self.max_message_characters < 8:
            raise ConversationSummaryError("max_message_characters must be at least 8")
        if self.max_summary_characters < 32:
            raise ConversationSummaryError("max_summary_characters must be at least 32")

    def summarize(self, messages: Sequence[ChatMessage]) -> ChatMessage:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise ConversationSummaryError("summary messages must be a sequence")
        if not messages:
            raise ConversationSummaryError("at least one message is required")
        if any(not isinstance(message, ChatMessage) for message in messages):
            raise ConversationSummaryError("summary messages contain an invalid message")

        header = f"较早对话提取式摘要（{len(messages)} 条消息）："
        available = self.max_summary_characters - len(header) - 1
        if available < 1:
            raise ConversationSummaryError("summary character limit is too small")

        entries: list[str] = []
        for message in reversed(messages):
            role = _role_label(message.role)
            content = " ".join(message.content.split())
            content = _truncate(content, self.max_message_characters)
            entry = f"{role}：{content}"
            if len(entry) <= available:
                entries.append(entry)
                available -= len(entry) + 1
                continue
            if available > len(role) + 3:
                entries.append(_truncate(entry, available))
                available = 0
            break

        summary = header + "\n" + "\n".join(reversed(entries))
        return ChatMessage(role="assistant", content=summary)


def _role_label(role: str) -> str:
    return {
        "user": "用户",
        "assistant": "助手",
        "tool": "工具结果",
    }.get(role, role)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return "…"[:limit]
    head_size = (limit - 1) // 2
    tail_size = limit - head_size - 1
    return f"{value[:head_size]}…{value[-tail_size:]}"
