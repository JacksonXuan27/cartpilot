from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.contracts import ChatMessage, TokenUsage, ToolCallMessage


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
                usage=self.usage,
                tool_calls=tool_calls,
            )
        return ModelResult(
            message=ChatMessage(role="assistant", content=self.reply),
            usage=self.usage,
        )

    async def _stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[ModelChunk]:
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


class OpenAICompatibleChatProvider:
    def __init__(self, model: object) -> None:
        self._model = model

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[object] = (),
    ) -> ModelResult:
        if not messages:
            raise ModelProviderError("at least one message is required")
        try:
            model = self._model
            if tools:
                model = model.bind_tools(_to_openai_tools(tools))
            response = await model.ainvoke(_to_langchain_messages(messages))
        except Exception as exc:
            raise ModelProviderError("chat model request failed") from exc

        tool_calls = _response_tool_calls(response)
        content = _response_text(getattr(response, "content", None))
        if not content and not tool_calls:
            raise ModelProviderError("chat model returned an empty response")

        usage_metadata = getattr(response, "usage_metadata", None) or {}
        usage = None
        if usage_metadata:
            usage = TokenUsage(
                prompt_tokens=int(usage_metadata.get("input_tokens", 0)),
                completion_tokens=int(usage_metadata.get("output_tokens", 0)),
                total_tokens=int(usage_metadata.get("total_tokens", 0)),
            )
        response_metadata = getattr(response, "response_metadata", None) or {}
        finish_reason = "tool_call" if tool_calls else str(
            response_metadata.get("finish_reason") or "stop"
        )
        tool_messages = [
            ToolCallMessage(
                id=call.call_id,
                name=call.name,
                arguments=dict(call.arguments),
            )
            for call in tool_calls
        ]
        message = (
            ChatMessage.assistant_tool_call(content, tool_messages)
            if tool_calls
            else ChatMessage(role="assistant", content=content)
        )
        return ModelResult(
            message=message,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tuple(tool_calls),
        )

    async def _stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[ModelChunk]:
        if not messages:
            raise ModelProviderError("at least one message is required")

        try:
            async for response_chunk in self._model.astream(
                _to_langchain_messages(messages)
            ):
                if getattr(response_chunk, "tool_call_chunks", ()):
                    raise ModelProviderError(
                        "tool calling is not supported by this provider"
                    )
                content = _response_text(getattr(response_chunk, "content", None))
                metadata = getattr(response_chunk, "response_metadata", None) or {}
                finish_reason = metadata.get("finish_reason")
                if content or finish_reason:
                    yield ModelChunk(delta=content, finish_reason=finish_reason)
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError("chat model stream failed") from exc

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ModelChunk]:
        return self._stream(messages)


def create_chat_model_provider(
    settings: Settings,
    model_factory: Callable[..., object] | None = None,
) -> ChatModelProvider:
    if settings.chat_provider == "stub":
        return StubModelProvider()

    missing = [
        name
        for name, value in (
            ("CHAT_MODEL", settings.chat_model),
            ("CHAT_BASE_URL", settings.chat_base_url),
            ("CHAT_API_KEY", settings.chat_api_key),
        )
        if not value or not value.strip()
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            f"{missing_names} required for openai-compatible provider"
        )

    factory = model_factory or ChatOpenAI
    model = factory(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
    )
    return OpenAICompatibleChatProvider(model)


def _to_openai_tools(tools: Sequence[object]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for tool in tools:
        payload = tool.model_dump() if hasattr(tool, "model_dump") else tool
        if not isinstance(payload, Mapping):
            raise ModelProviderError("tool definition must be a mapping")
        name = payload.get("name")
        description = payload.get("description")
        parameters = payload.get("input_schema", payload.get("parameters"))
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(description, str)
            or not isinstance(parameters, Mapping)
        ):
            raise ModelProviderError("tool definition is invalid")
        converted.append(
            {
                "name": name,
                "description": description,
                "parameters": dict(parameters),
            }
        )
    return converted


def _response_tool_calls(response: object) -> list[ToolCall]:
    raw_tool_calls = getattr(response, "tool_calls", None) or ()
    if not isinstance(raw_tool_calls, (list, tuple)):
        raise ModelProviderError("chat model returned invalid tool calls")

    tool_calls = []
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, Mapping):
            raise ModelProviderError("chat model returned an invalid tool call")
        name = raw_call.get("name")
        arguments = raw_call.get("args")
        call_id = raw_call.get("id")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(arguments, Mapping)
            or not isinstance(call_id, str)
            or not call_id.strip()
        ):
            raise ModelProviderError("chat model returned an invalid tool call")
        tool_calls.append(
            ToolCall(name=name, arguments=dict(arguments), call_id=call_id)
        )
    return tool_calls


def _to_langchain_messages(messages: Sequence[ChatMessage]) -> list[object]:
    converted: list[object] = []
    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(
                AIMessage(
                    content=message.content,
                    tool_calls=[
                        {
                            "name": call.name,
                            "args": call.arguments,
                            "id": call.id,
                            "type": "tool_call",
                        }
                        for call in message.tool_calls
                    ],
                )
            )
        else:
            converted.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id or "unknown",
                )
            )
    return converted


def _response_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    return ""