import pytest

from app.context_layers import ContextLayerError, ContextLayerManager
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
    assert [message.content for message in layers.prompt_messages] == [
        "You are a support agent.",
        "question-2",
        "answer-2",
        "question-3",
        "answer-3",
    ]
    assert layers.omitted_message_count == 2


def test_context_layers_reject_invalid_limits_and_messages():
    with pytest.raises(ContextLayerError, match="short_term_limit"):
        ContextLayerManager(short_term_limit=0)

    with pytest.raises(ContextLayerError, match="long_term_limit"):
        ContextLayerManager(long_term_limit=-1)

    with pytest.raises(ContextLayerError, match="invalid message"):
        ContextLayerManager().build([{"role": "user", "content": "hello"}])
