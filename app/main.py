import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.chat import ChatService, StreamingRequestError, error_response
from app.contracts import ChatRequest, ChatResponse
from app.providers import ModelProviderError, StubModelProvider
from app.sessions import InMemorySessionStore, SessionNotFoundError


app = FastAPI(title="CartPilot")
app.state.chat_service = ChatService(
    session_store=InMemorySessionStore(),
    model_provider=StubModelProvider(),
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"service": "cartpilot", "status": "ok"}


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
