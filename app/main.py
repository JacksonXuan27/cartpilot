from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
