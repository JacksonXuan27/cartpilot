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
    assert payload["run_id"] and payload["workflow_id"]


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
