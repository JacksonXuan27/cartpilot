import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.after_sales import AfterSalesExtractionError, AfterSalesExtractor
from app.chat import ChatService, StreamingRequestError, error_response
from app.embeddings import create_embedding_provider
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
from app.providers import ModelProviderError, create_chat_model_provider
from app.query_rewriting import RuleBasedQueryRewriter
from app.reranking import create_reranker
from app.sessions import InMemorySessionStore, SessionNotFoundError
from app.vector_store import MilvusVectorStore
from app.vector_store_factory import create_vector_store


router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"service": "cartpilot", "status": "ok"}


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else get_settings()
    provider = create_chat_model_provider(settings)
    embedding_provider = create_embedding_provider(settings)
    reranker = create_reranker(settings)
    query_rewriter = RuleBasedQueryRewriter()
    vector_store = create_vector_store(settings, dimension=embedding_provider.dimension)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            if isinstance(vector_store, MilvusVectorStore):
                await asyncio.to_thread(vector_store.client.close)

    application = FastAPI(title="CartPilot", lifespan=lifespan)
    application.state.chat_service = ChatService(
        session_store=InMemorySessionStore(), model_provider=provider
    )
    application.state.after_sales_extractor = AfterSalesExtractor(provider)
    application.state.knowledge_base_service = KnowledgeBaseService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        model_provider=provider,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )
    application.include_router(router)
    return application


app = create_app()
