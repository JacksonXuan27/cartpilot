import json

from fastapi.testclient import TestClient

from app.after_sales import AfterSalesExtractor
from app.chat import ChatService
from app.main import app
from app.providers import StubModelProvider
from app.sessions import InMemorySessionStore


def configure_application() -> TestClient:
    provider = StubModelProvider(reply="I can help with your order.", chunk_size=6)
    app.state.chat_service = ChatService(
        session_store=InMemorySessionStore(),
        model_provider=provider,
    )
    app.state.after_sales_extractor = AfterSalesExtractor(provider)
    return TestClient(app)


def test_ch01_public_endpoints_work_together():
    client = configure_application()

    health = client.get("/healthz")
    assert health.json() == {"service": "cartpilot", "status": "ok"}

    chat = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Where is my order?"}]},
    )
    assert chat.status_code == 200
    session_id = chat.json()["session_id"]

    stream = client.post(
        "/chat/stream",
        json={
            "stream": True,
            "session_id": session_id,
            "messages": [{"role": "user", "content": "Any update?"}],
        },
    )
    assert stream.status_code == 200
    assert "event: done" in stream.text
    assert session_id in stream.text

    extraction_provider = StubModelProvider(
        reply=json.dumps(
            {
                "intent": "refund",
                "order_id": "ORD-3003",
                "reason": "damaged",
                "requested_action": "refund",
                "confidence": 0.91,
            }
        )
    )
    app.state.after_sales_extractor = AfterSalesExtractor(extraction_provider)
    extraction = client.post(
        "/after-sales/extract",
        json={
            "messages": [
                {"role": "user", "content": "ORD-3003 arrived damaged"}
            ]
        },
    )
    assert extraction.status_code == 200
    assert extraction.json()["data"]["intent"] == "refund"

