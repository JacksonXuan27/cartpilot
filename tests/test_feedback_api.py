import pytest
from fastapi.testclient import TestClient

from app.database import DatabaseManager
from app.main import app
from app.repositories import UserFeedbackRepository, initialize_schema


@pytest.fixture
def feedback_repository(tmp_path, monkeypatch):
    database = DatabaseManager(f"sqlite:///{tmp_path / 'feedback.db'}")
    database.open()
    initialize_schema(database)
    repository = UserFeedbackRepository(database)
    monkeypatch.setattr(app.state, "feedback_repository", repository)
    yield repository
    database.close()


@pytest.mark.asyncio
async def test_feedback_endpoint_persists_rating_without_conversation_content(
    feedback_repository,
):
    client = TestClient(app)

    response = client.post(
        "/feedback",
        json={
            "request_id": "knowledge-request-1",
            "trace_id": "trace-1",
            "rating": "unhelpful",
            "reason": "检索结果不相关",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "saved"
    assert body["feedback_id"]
    records = await feedback_repository.list_recent(request_id="knowledge-request-1")
    assert len(records) == 1
    assert records[0].rating == "unhelpful"
    assert records[0].trace_id == "trace-1"
    assert records[0].reason == "检索结果不相关"


def test_feedback_endpoint_validates_rating_and_reason_length(feedback_repository):
    app.state.feedback_repository = feedback_repository
    client = TestClient(app)

    invalid_rating = client.post(
        "/feedback",
        json={"request_id": "request-1", "rating": "neutral"},
    )
    oversized_reason = client.post(
        "/feedback",
        json={
            "request_id": "request-1",
            "rating": "helpful",
            "reason": "x" * 501,
        },
    )

    assert invalid_rating.status_code == 422
    assert oversized_reason.status_code == 422
