from fastapi.testclient import TestClient

from app.main import app
from app.providers import StubModelProvider
from app.tool_registry import ToolRegistry
from app.workflow_runtime import WorkflowRuntime


def test_workflow_run_returns_execution_summary():
    app.state.workflow_runtime = WorkflowRuntime(StubModelProvider(reply="可以帮你处理。"), ToolRegistry())
    response = TestClient(app).post("/workflow/run", json={
        "session_id": "session-api",
        "messages": [{"role": "user", "content": "你好"}],
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["answer"] == "可以帮你处理。"
    assert payload["intent"] == "smalltalk"
    assert payload["route"] == "fallback_script"
    assert payload["iterations"] == 1
    assert payload["confirmation_required"] is False
    assert payload["run_id"] and payload["workflow_id"]


def test_workflow_run_returns_pending_refund_confirmation():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-6006",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    app.state.workflow_runtime = WorkflowRuntime(provider, ToolRegistry())

    response = TestClient(app).post("/workflow/run", json={
        "messages": [
            {"role": "user", "content": "ORD-6006 到货破损，我要退款"}
        ],
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "awaiting_confirmation"
    assert payload["confirmation_required"] is True
    assert payload["confirmation_id"]
    assert payload["confirmation_details"]["order_id"] == "ORD-6006"
    assert len(provider.calls) == 1


def test_workflow_resume_accepts_confirmation_and_returns_final_state():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-7007",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    app.state.workflow_runtime = WorkflowRuntime(provider, ToolRegistry())
    client = TestClient(app)
    paused = client.post("/workflow/run", json={
        "messages": [
            {"role": "user", "content": "ORD-7007 到货破损，我要退款"}
        ],
    }).json()

    resumed = client.post("/workflow/resume", json={
        "confirmation_id": paused["confirmation_id"],
        "confirmed": True,
    })

    assert resumed.status_code == 200
    assert resumed.json()["status"] == "completed"
    assert resumed.json()["confirmation_required"] is False


def test_workflow_resume_rejects_confirmation_without_calling_model_again():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-7009",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    app.state.workflow_runtime = WorkflowRuntime(provider, ToolRegistry())
    client = TestClient(app)
    paused = client.post("/workflow/run", json={
        "messages": [
            {"role": "user", "content": "ORD-7009 到货破损，我要退款"}
        ],
    }).json()

    rejected = client.post("/workflow/resume", json={
        "confirmation_id": paused["confirmation_id"],
        "confirmed": False,
    })

    assert rejected.status_code == 200
    assert rejected.json()["status"] == "completed"
    assert rejected.json()["answer"] == "已取消此售后申请。"
    assert rejected.json()["confirmation_required"] is False
    assert len(provider.calls) == 1


def test_workflow_resume_rejects_duplicate_confirmation():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-7008",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        )
    )
    app.state.workflow_runtime = WorkflowRuntime(provider, ToolRegistry())
    client = TestClient(app)
    paused = client.post("/workflow/run", json={
        "messages": [
            {"role": "user", "content": "ORD-7008 到货破损，我要退款"}
        ],
    }).json()
    request = {
        "confirmation_id": paused["confirmation_id"],
        "confirmed": True,
    }

    first = client.post("/workflow/resume", json=request)
    duplicate = client.post("/workflow/resume", json=request)

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "checkpoint_already_resumed"


def test_workflow_run_validates_request():
    response = TestClient(app).post("/workflow/run", json={"messages": []})
    assert response.status_code == 422


def test_workflow_failure_is_returned_as_structured_response():
    class BrokenProvider(StubModelProvider):
        async def complete(self, messages, tools=()):
            raise RuntimeError("provider is unavailable")

    app.state.workflow_runtime = WorkflowRuntime(BrokenProvider(), ToolRegistry())
    response = TestClient(app).post("/workflow/run", json={
        "messages": [{"role": "user", "content": "查订单"}],
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "react_loop_failed"
    assert payload["error"]["message"]
