import json
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import JSONResponse, StreamingResponse

from app.after_sales import AfterSalesExtractionError, AfterSalesExtractor
from app.chat import ChatService, StreamingRequestError, error_response
from app.data_models import UserFeedback
from app.database import DatabaseManager
from app.embeddings import HashEmbeddingProvider
from app.contracts import (
    AfterSalesExtractionRequest,
    AfterSalesExtractionResponse,
    ChatRequest,
    ChatResponse,
)
from app.knowledge_base import (
    KnowledgeBaseError,
    KnowledgeBaseService,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
)
from app.providers import ModelProviderError, StubModelProvider
from app.sessions import InMemorySessionStore, SessionNotFoundError
from app.repositories import RecordNotFoundError, UserFeedbackRepository, initialize_schema
from app.vector_store import InMemoryVectorStore
from app.workflow_runtime import (
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowResumeRequest,
    WorkflowRuntime,
    default_workflow_runtime,
    workflow_response,
)
from app.workflow_checkpoints import (
    WorkflowCheckpointAlreadyResumedError,
    WorkflowCheckpointNotFoundError,
)


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    rating: Literal["helpful", "unhelpful"]
    reason: str | None = Field(default=None, min_length=1, max_length=500)


class FeedbackSubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    status: str = "saved"


class FeedbackReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_status: Literal["accepted", "needs_revision", "dismissed"]
    review_note: str | None = Field(default=None, max_length=1000)
    reviewed_by: str = Field(min_length=1, max_length=128)


class FeedbackReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    review_status: str
    reviewed_at: datetime


app = FastAPI(title="CartPilot")
feedback_database = DatabaseManager("sqlite:///./data/cartpilot.db")
feedback_database.open()
initialize_schema(feedback_database)
app.state.feedback_repository = UserFeedbackRepository(feedback_database)
default_provider = StubModelProvider()
app.state.chat_service = ChatService(
    session_store=InMemorySessionStore(),
    model_provider=default_provider,
)
app.state.after_sales_extractor = AfterSalesExtractor(default_provider)
default_embedding_provider = HashEmbeddingProvider(dimension=64)
app.state.knowledge_base_service = KnowledgeBaseService(
    embedding_provider=default_embedding_provider,
    vector_store=InMemoryVectorStore(dimension=64),
    model_provider=default_provider,
)
app.state.workflow_runtime = default_workflow_runtime()


@app.post("/feedback", response_model=FeedbackSubmissionResponse, status_code=201)
async def submit_feedback(
    submission: FeedbackSubmission, http_request: Request
) -> FeedbackSubmissionResponse:
    feedback = UserFeedback(
        feedback_id=uuid4(),
        request_id=submission.request_id,
        trace_id=submission.trace_id,
        rating=submission.rating,
        reason=submission.reason,
        created_at=datetime.now(timezone.utc),
    )
    repository: UserFeedbackRepository = http_request.app.state.feedback_repository
    await repository.save(feedback)
    return FeedbackSubmissionResponse(feedback_id=str(feedback.feedback_id))


@app.get("/feedback/export")
async def export_feedback(
    http_request: Request,
    rating: Literal["helpful", "unhelpful"] | None = None,
    review_status: Literal["pending", "accepted", "needs_revision", "dismissed"] | None = None,
    limit: int = Query(default=500, ge=1, le=1000),
) -> StreamingResponse:
    repository: UserFeedbackRepository = http_request.app.state.feedback_repository
    records = await repository.list_recent(
        limit=limit,
        rating=rating,
        review_status=review_status,
    )
    lines = "".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for record in records
    )
    return StreamingResponse(
        iter([lines]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="feedback.jsonl"'},
    )


@app.post(
    "/feedback/{feedback_id}/review",
    response_model=FeedbackReviewResponse,
)
async def review_feedback(
    feedback_id: str,
    submission: FeedbackReviewSubmission,
    http_request: Request,
) -> FeedbackReviewResponse | JSONResponse:
    repository: UserFeedbackRepository = http_request.app.state.feedback_repository
    try:
        existing = await repository.get(UUID(feedback_id))
    except (ValueError, RecordNotFoundError):
        return JSONResponse(status_code=404, content={"detail": "feedback not found"})
    reviewed_at = datetime.now(timezone.utc)
    updated = existing.model_copy(
        update={
            "review_status": submission.review_status,
            "review_note": submission.review_note,
            "reviewed_by": submission.reviewed_by,
            "reviewed_at": reviewed_at,
        }
    )
    await repository.update_review(updated)
    return FeedbackReviewResponse(
        feedback_id=str(updated.feedback_id),
        review_status=updated.review_status,
        reviewed_at=reviewed_at,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"service": "cartpilot", "status": "ok"}


@app.post("/workflow/run", response_model=WorkflowRunResponse)
async def workflow_run(
    request: WorkflowRunRequest, http_request: Request
) -> WorkflowRunResponse:
    runtime: WorkflowRuntime = http_request.app.state.workflow_runtime
    state = await runtime.run(request.messages, session_id=request.session_id)
    return workflow_response(state)


@app.post("/workflow/resume", response_model=WorkflowRunResponse)
async def workflow_resume(
    request: WorkflowResumeRequest, http_request: Request
) -> WorkflowRunResponse | JSONResponse:
    runtime: WorkflowRuntime = http_request.app.state.workflow_runtime
    try:
        state = await runtime.resume(
            request.confirmation_id,
            confirmed=request.confirmed,
        )
    except WorkflowCheckpointNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content=error_response(
                code="checkpoint_not_found",
                message=str(exc),
                retryable=False,
            ).model_dump(mode="json"),
        )
    except WorkflowCheckpointAlreadyResumedError as exc:
        return JSONResponse(
            status_code=409,
            content=error_response(
                code="checkpoint_already_resumed",
                message=str(exc),
                retryable=False,
            ).model_dump(mode="json"),
        )
    return workflow_response(state)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse | JSONResponse:
    service: ChatService = http_request.app.state.chat_service
    try:
        return await service.complete(request)
    except StreamingRequestError as exc:
        return JSONResponse(
            status_code=400,
            content=error_response(
                code="streaming_not_supported",
                message=str(exc),
                retryable=False,
            ).model_dump(mode="json"),
        )
    except SessionNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content=error_response(
                code="session_not_found",
                message=f"session not found: {exc}",
                retryable=False,
            ).model_dump(mode="json"),
        )
    except ModelProviderError as exc:
        return JSONResponse(
            status_code=502,
            content=error_response(
                code="model_provider_error",
                message=str(exc),
                retryable=True,
            ).model_dump(mode="json"),
        )


@app.post("/after-sales/extract", response_model=AfterSalesExtractionResponse)
async def extract_after_sales(
    request: AfterSalesExtractionRequest, http_request: Request
) -> AfterSalesExtractionResponse | JSONResponse:
    extractor: AfterSalesExtractor = http_request.app.state.after_sales_extractor
    try:
        return await extractor.extract(request.messages)
    except AfterSalesExtractionError as exc:
        return JSONResponse(
            status_code=422,
            content=error_response(
                code="invalid_after_sales_output",
                message=str(exc),
                retryable=False,
            ).model_dump(mode="json"),
        )
    except ModelProviderError as exc:
        return JSONResponse(
            status_code=502,
            content=error_response(
                code="model_provider_error",
                message=str(exc),
                retryable=True,
            ).model_dump(mode="json"),
        )


@app.post("/knowledge/query", response_model=KnowledgeQueryResponse)
async def knowledge_query(
    request: KnowledgeQueryRequest, http_request: Request
) -> KnowledgeQueryResponse | JSONResponse:
    service: KnowledgeBaseService = http_request.app.state.knowledge_base_service
    try:
        return await service.query(request)
    except KnowledgeBaseError as exc:
        return JSONResponse(
            status_code=503,
            content=error_response(
                code="knowledge_base_error",
                message=str(exc),
                retryable=True,
            ).model_dump(mode="json"),
        )


@app.post("/chat/stream", response_model=None)
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse | JSONResponse:
    service: ChatService = http_request.app.state.chat_service
    try:
        context = await service.prepare_stream(request)
    except StreamingRequestError as exc:
        return JSONResponse(
            status_code=400,
            content=error_response(
                code="streaming_required",
                message=str(exc),
                retryable=False,
            ).model_dump(mode="json"),
        )
    except SessionNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content=error_response(
                code="session_not_found",
                message=f"session not found: {exc}",
                retryable=False,
            ).model_dump(mode="json"),
        )

    async def events():
        try:
            async for chunk in service.stream(context):
                event = {
                    "request_id": context.request_id,
                    "session_id": context.session_id,
                    "delta": chunk.delta,
                }
                if chunk.finish_reason is not None:
                    event["finish_reason"] = chunk.finish_reason
                yield f"event: message\ndata: {json.dumps(event)}\n\n"
            yield (
                "event: done\n"
                f"data: {json.dumps({'request_id': context.request_id, 'session_id': context.session_id})}\n\n"
            )
        except ModelProviderError as exc:
            error = error_response(
                code="model_provider_error",
                message=str(exc),
                retryable=True,
            ).model_dump(mode="json")
            yield f"event: error\ndata: {json.dumps(error)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
