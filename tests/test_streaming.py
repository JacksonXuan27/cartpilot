from fastapi.testclient import TestClient

from app.chat import ChatService
from app.main import app
from app.providers import StubModelProvider
from app.sessions import InMemorySessionStore


def make_client(reply: str = "Your order is on the way.", chunk_size: int = 8) -> TestClient:
    app.state.chat_service = ChatService(
        session_store=InMemorySessionStore(),
        model_provider=StubModelProvider(reply=reply, chunk_size=chunk_size),
    )
    return TestClient(app)


def test_stream_endpoint_returns_sse_message_chunks_and_done_event():
    response = make_client(chunk_size=5).post(
        "/chat/stream",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Where is my order?"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message" in response.text
    assert "event: done" in response.text
    assert '"finish_reason": "stop"' in response.text
    assert "Your order is on the way." == "".join(
        line.split('"delta": "')[1].split('"')[0]
        for line in response.text.splitlines()
        if '"delta"' in line
    )


def test_stream_endpoint_reuses_a_session():
    client = make_client(reply="Thanks for confirming.")
    first = client.post(
        "/chat/stream",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    session_id = next(
        line.split('"session_id": "')[1].split('"')[0]
        for line in first.text.splitlines()
        if '"session_id"' in line
    )

    second = client.post(
        "/chat/stream",
        json={
            "stream": True,
            "session_id": session_id,
            "messages": [{"role": "user", "content": "Continue"}],
        },
    )

    assert second.status_code == 200
    assert session_id in second.text


def test_stream_endpoint_requires_the_stream_flag():
    response = make_client().post(
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "streaming_required"
