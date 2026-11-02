import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock

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
class ContextCacheStats:
    capacity: int
    size: int
    hits: int
    misses: int
    evictions: int


class ContextLayerCache:
    def __init__(self, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ContextLayerError("cache max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, ContextLayers] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> ContextLayers | None:
        with self._lock:
            layers = self._entries.get(key)
            if layers is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return _copy_layers(layers)

    def put(self, key: str, layers: ContextLayers) -> None:
        with self._lock:
            self._entries[key] = _copy_layers(layers)
            self._entries.move_to_end(key)
            if len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    def stats(self) -> ContextCacheStats:
        with self._lock:
            return ContextCacheStats(
                capacity=self._max_entries,
                size=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0


@dataclass(frozen=True, slots=True)
class ContextLayerManager:
    short_term_limit: int = 6
    long_term_limit: int = 20
    summarizer: ExtractiveConversationSummarizer = ExtractiveConversationSummarizer()
    cache: ContextLayerCache = field(
        default_factory=ContextLayerCache,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.short_term_limit < 1:
            raise ContextLayerError("short_term_limit must be positive")
        if self.long_term_limit < 0:
            raise ContextLayerError("long_term_limit cannot be negative")
        if not isinstance(self.cache, ContextLayerCache):
            raise ContextLayerError("cache must be a ContextLayerCache")

    def build(self, messages: Sequence[ChatMessage]) -> ContextLayers:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise ContextLayerError("context messages must be a sequence")

        normalized = tuple(_copy_message(message) for message in messages)
        cache_key = _cache_key(
            normalized,
            self.short_term_limit,
            self.long_term_limit,
            self.summarizer,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

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
        layers = ContextLayers(
            system=system,
            long_term=long_term,
            short_term=short_term,
            summary=summary,
            omitted_message_count=omitted_message_count,
        )
        self.cache.put(cache_key, layers)
        return layers


def _cache_key(
    messages: Sequence[ChatMessage],
    short_term_limit: int,
    long_term_limit: int,
    summarizer: ExtractiveConversationSummarizer,
) -> str:
    payload = {
        "short_term_limit": short_term_limit,
        "long_term_limit": long_term_limit,
        "max_message_characters": summarizer.max_message_characters,
        "max_summary_characters": summarizer.max_summary_characters,
        "messages": [
            [message.role, message.content, message.tool_call_id]
            for message in messages
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=20).hexdigest()


def _copy_layers(layers: ContextLayers) -> ContextLayers:
    return ContextLayers(
        system=tuple(message.model_copy(deep=True) for message in layers.system),
        long_term=tuple(
            message.model_copy(deep=True) for message in layers.long_term
        ),
        short_term=tuple(
            message.model_copy(deep=True) for message in layers.short_term
        ),
        summary=(
            layers.summary.model_copy(deep=True)
            if layers.summary is not None
            else None
        ),
        omitted_message_count=layers.omitted_message_count,
    )


def _copy_message(message: ChatMessage) -> ChatMessage:
    if not isinstance(message, ChatMessage):
        raise ContextLayerError("context messages contain an invalid message")
    return message.model_copy(deep=True)
