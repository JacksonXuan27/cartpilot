import json

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


def test_feedback_endpoint_exports_reviewable_jsonl(feedback_repository):
    client = TestClient(app)
    created = client.post(
        "/feedback",
        json={
            "request_id": "request-export",
            "rating": "unhelpful",
            "reason": "答案不完整",
        },
    ).json()

    reviewed = client.post(
        f"/feedback/{created['feedback_id']}/review",
        json={
            "review_status": "needs_revision",
            "review_note": "补充知识库依据",
            "reviewed_by": "qa-user",
        },
    )

    assert reviewed.status_code == 200
    assert reviewed.json()["review_status"] == "needs_revision"
    exported = client.get(
        "/feedback/export",
        params={"review_status": "needs_revision", "limit": 10},
    )

    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("application/x-ndjson")
    records = [json.loads(line) for line in exported.text.splitlines()]
    assert len(records) == 1
    assert records[0]["feedback_id"] == created["feedback_id"]
    assert records[0]["reviewed_by"] == "qa-user"
    assert records[0]["reason"] == "答案不完整"


def test_feedback_review_reports_missing_feedback(feedback_repository):
    response = TestClient(app).post(
        "/feedback/not-a-uuid/review",
        json={"review_status": "accepted", "reviewed_by": "qa-user"},
    )

    assert response.status_code == 404


def test_feedback_endpoint_validates_rating_and_reason_length(feedback_repository):
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
