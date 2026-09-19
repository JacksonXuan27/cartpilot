from fastapi.testclient import TestClient

from app.main import app
from app.providers import StubModelProvider
from app.sessions import InMemorySessionStore
from app.chat import ChatService


def make_client(reply: str = "Your order is on the way.") -> TestClient:
    app.state.chat_service = ChatService(
        session_store=InMemorySessionStore(),
        model_provider=StubModelProvider(reply=reply),
    )
    return TestClient(app)


def test_chat_creates_a_session_and_returns_a_non_streaming_response():
    response = make_client().post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Where is my order?"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["session_id"]
    assert body["message"] == {
        "role": "assistant",
        "content": "Your order is on the way.",
    }
    assert body["finish_reason"] == "stop"


def test_chat_reuses_a_session_for_follow_up_messages():
    client = make_client()
    first = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    session_id = first.json()["session_id"]

    second = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "messages": [{"role": "user", "content": "Can I return it?"}],
        },
    )

    assert second.status_code == 200
    assert second.json()["session_id"] == session_id


def test_chat_rejects_streaming_requests_on_the_non_streaming_endpoint():
    response = make_client().post(
        "/chat",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "streaming_not_supported",
            "message": "stream=true requires the streaming chat endpoint",
            "retryable": False,
        }
    }


def test_chat_returns_not_found_for_an_unknown_session():
    response = make_client().post(
        "/chat",
        json={
            "session_id": "missing",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"
