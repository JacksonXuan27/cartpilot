import pytest

from app.context_layers import (
    ContextLayerCache,
    ContextLayerError,
    ContextLayerManager,
)
from app.contracts import ChatMessage


def test_context_layers_keep_system_and_split_history_by_recency():
    manager = ContextLayerManager(short_term_limit=2, long_term_limit=2)
    messages = [
        ChatMessage(role="system", content="You are a support agent."),
        ChatMessage(role="user", content="question-1"),
        ChatMessage(role="assistant", content="answer-1"),
        ChatMessage(role="user", content="question-2"),
        ChatMessage(role="assistant", content="answer-2"),
        ChatMessage(role="user", content="question-3"),
        ChatMessage(role="assistant", content="answer-3"),
    ]

    layers = manager.build(messages)

    assert [message.content for message in layers.system] == [
        "You are a support agent."
    ]
    assert [message.content for message in layers.long_term] == [
        "question-2",
        "answer-2",
    ]
    assert [message.content for message in layers.short_term] == [
        "question-3",
        "answer-3",
    ]
    assert layers.summary is not None
    assert layers.summary.role == "assistant"
    assert "较早对话提取式摘要（2 条消息）" in layers.summary.content
    assert "question-1" in layers.summary.content
    assert [message.content for message in layers.prompt_messages] == [
        "You are a support agent.",
        layers.summary.content,
        "question-2",
        "answer-2",
        "question-3",
        "answer-3",
    ]
    assert layers.omitted_message_count == 2


def test_extractive_summary_normalizes_and_bounds_long_messages():
    from app.conversation_summary import ExtractiveConversationSummarizer

    summarizer = ExtractiveConversationSummarizer(
        max_message_characters=12,
        max_summary_characters=64,
    )
    summary = summarizer.summarize(
        [ChatMessage(role="user", content="  keep the order ORD-12345 and mention damage  ")]
    )

    assert summary.role == "assistant"
    assert len(summary.content) <= 64
    assert "用户" in summary.content
    assert "…" in summary.content


def test_context_layers_do_not_summarize_history_within_retention_limits():
    layers = ContextLayerManager(short_term_limit=2, long_term_limit=2).build(
        [
            ChatMessage(role="user", content="question"),
            ChatMessage(role="assistant", content="answer"),
        ]
    )

    assert layers.summary is None
    assert layers.omitted_message_count == 0


def test_context_layer_cache_reuses_isolated_context_snapshots():
    manager = ContextLayerManager(short_term_limit=1, long_term_limit=1)
    messages = [
        ChatMessage(role="user", content="question-1"),
        ChatMessage(role="assistant", content="answer-1"),
        ChatMessage(role="user", content="question-2"),
        ChatMessage(role="assistant", content="answer-2"),
    ]

    first = manager.build(messages)
    first.summary.content = "mutated cached summary"
    first.short_term[-1].content = "mutated cached message"
    second = manager.build(messages)

    assert "question-1" in second.summary.content
    assert second.short_term[-1].content == "answer-2"
    assert manager.cache.stats().hits == 1
    assert manager.cache.stats().misses == 1


def test_context_layer_cache_invalidates_changed_messages_and_configuration():
    cache = ContextLayerCache(max_entries=4)
    messages = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="second"),
        ChatMessage(role="user", content="third"),
    ]
    manager = ContextLayerManager(short_term_limit=1, long_term_limit=1, cache=cache)

    manager.build(messages)
    messages[0].content = "updated"
    manager.build(messages)
    ContextLayerManager(
        short_term_limit=2,
        long_term_limit=1,
        cache=cache,
    ).build(messages)

    stats = cache.stats()
    assert stats.misses == 3
    assert stats.hits == 0
    assert stats.size == 3


def test_context_layer_cache_evicts_least_recently_used_entries():
    cache = ContextLayerCache(max_entries=2)
    manager = ContextLayerManager(
        short_term_limit=1,
        long_term_limit=0,
        cache=cache,
    )
    first = [ChatMessage(role="user", content="first")]
    second = [ChatMessage(role="user", content="second")]
    third = [ChatMessage(role="user", content="third")]

    manager.build(first)
    manager.build(second)
    manager.build(first)
    manager.build(third)
    manager.build(first)
    manager.build(second)

    stats = cache.stats()
    assert stats.capacity == 2
    assert stats.size == 2
    assert stats.hits == 2
    assert stats.misses == 4
    assert stats.evictions == 2


def test_context_layer_cache_clear_resets_entries_and_statistics():
    manager = ContextLayerManager()
    messages = [ChatMessage(role="user", content="hello")]

    manager.build(messages)
    manager.build(messages)
    manager.cache.clear()
    manager.build(messages)

    stats = manager.cache.stats()
    assert stats.size == 1
    assert stats.hits == 0
    assert stats.misses == 1
    assert stats.evictions == 0


def test_context_layers_reject_invalid_limits_and_messages():
    with pytest.raises(ContextLayerError, match="short_term_limit"):
        ContextLayerManager(short_term_limit=0)

    with pytest.raises(ContextLayerError, match="long_term_limit"):
        ContextLayerManager(long_term_limit=-1)

    with pytest.raises(ContextLayerError, match="invalid message"):
        ContextLayerManager().build([{"role": "user", "content": "hello"}])
