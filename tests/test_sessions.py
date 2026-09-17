from datetime import timezone
from uuid import UUID

import pytest

from app.contracts import ChatMessage
from app.sessions import (
    InMemorySessionStore,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)


def test_create_generates_a_valid_session_with_utc_timestamps():
    session = InMemorySessionStore().create()

    UUID(session.session_id)
    assert session.messages == []
    assert session.created_at.tzinfo == timezone.utc
    assert session.updated_at == session.created_at


def test_create_accepts_an_explicit_identifier_and_rejects_duplicates():
    store = InMemorySessionStore()
    store.create("session-123")

    with pytest.raises(SessionAlreadyExistsError):
        store.create("session-123")


def test_append_message_updates_session_and_returns_an_isolated_copy():
    store = InMemorySessionStore()
    session = store.create("session-123")

    updated = store.append_message(
        session.session_id,
        ChatMessage(role="user", content="Where is my order?"),
    )
    updated.messages.clear()

    stored = store.get(session.session_id)
    assert len(stored.messages) == 1
    assert stored.messages[0].content == "Where is my order?"
    assert stored.updated_at >= stored.created_at


def test_missing_sessions_raise_a_domain_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        store.get("missing")

    with pytest.raises(SessionNotFoundError):
        store.append_message("missing", ChatMessage(role="user", content="hello"))

    with pytest.raises(SessionNotFoundError):
        store.delete("missing")


def test_delete_removes_an_existing_session():
    store = InMemorySessionStore()
    store.create("session-123")

    store.delete("session-123")

    with pytest.raises(SessionNotFoundError):
        store.get("session-123")
