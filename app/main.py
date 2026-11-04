import json

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.after_sales import AfterSalesExtractionError, AfterSalesExtractor
from app.chat import ChatService, StreamingRequestError, error_response
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
from app.providers import (
    ModelProviderError,
    create_chat_model_provider,
)
from app.sessions import InMemorySessionStore, SessionNotFoundError
from app.tool_registry import ToolRegistry
from app.vector_store import InMemoryVectorStore
from app.workflow_runtime import (
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowRuntime,
    default_workflow_runtime,
    workflow_response,
)


router = APIRouter()


def create_app(
    settings: Settings | None = None,
    *,
    tool_registry: ToolRegistry | None = None,
    model_provider=None,
) -> FastAPI:
    settings = settings if settings is not None else get_settings()
    provider = (
        model_provider
        if model_provider is not None
        else create_chat_model_provider(settings)
    )
    application = FastAPI(title="CartPilot")
    application.state.chat_service = ChatService(
        session_store=InMemorySessionStore(), model_provider=provider
    )
    application.state.after_sales_extractor = AfterSalesExtractor(provider)
    application.state.knowledge_base_service = KnowledgeBaseService(
        embedding_provider=HashEmbeddingProvider(dimension=64),
        vector_store=InMemoryVectorStore(dimension=64),
        model_provider=provider,
    )
    application.state.workflow_runtime = default_workflow_runtime(
        provider, tool_registry
    )
    application.include_router(router)
    return application


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"service": "cartpilot", "status": "ok"}


@router.post("/workflow/run", response_model=WorkflowRunResponse)
async def workflow_run(
    request: WorkflowRunRequest, http_request: Request
) -> WorkflowRunResponse:
    runtime: WorkflowRuntime = http_request.app.state.workflow_runtime
    state = await runtime.run(request.messages, session_id=request.session_id)
    return workflow_response(state)


@router.post("/chat", response_model=ChatResponse)
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


@router.post("/after-sales/extract", response_model=AfterSalesExtractionResponse)
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


@router.post("/knowledge/query", response_model=KnowledgeQueryResponse)
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


@router.post("/chat/stream", response_model=None)
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


app = create_app()
