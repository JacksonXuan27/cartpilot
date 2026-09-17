import pytest
from pydantic import ValidationError

from app.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorDetail,
    ErrorResponse,
    TokenUsage,
)


def test_chat_request_accepts_a_conversation_and_stream_flag():
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Where is my order?")],
        session_id="session-123",
        stream=True,
    )

    assert request.messages[0].role == "user"
    assert request.session_id == "session-123"
    assert request.stream is True


def test_chat_request_rejects_empty_messages_and_unknown_fields():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])

    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "hello", "extra": "nope"}])


def test_chat_response_contains_message_and_usage_contract():
    response = ChatResponse(
        request_id="request-123",
        session_id="session-123",
        message={"role": "assistant", "content": "I can help with that."},
        usage={"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
    )

    assert response.message.role == "assistant"
    assert response.usage is not None
    assert response.usage.total_tokens == 14


def test_error_response_exposes_machine_code_and_retryability():
    response = ErrorResponse(
        error=ErrorDetail(
            code="provider_timeout",
            message="The model provider did not respond in time.",
            retryable=True,
        )
    )

    assert response.error.code == "provider_timeout"
    assert response.error.retryable is True


def test_token_usage_rejects_negative_counts():
    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1)
