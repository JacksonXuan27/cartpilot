from uuid import uuid4

from app.contracts import ChatRequest, ChatResponse, ErrorDetail, ErrorResponse
from app.providers import ChatModelProvider, ModelProviderError
from app.sessions import InMemorySessionStore, SessionNotFoundError


class StreamingRequestError(ValueError):
    pass


class ChatService:
    def __init__(
        self,
        session_store: InMemorySessionStore,
        model_provider: ChatModelProvider,
    ) -> None:
        self._session_store = session_store
        self._model_provider = model_provider

    async def complete(self, request: ChatRequest) -> ChatResponse:
        if request.stream:
            raise StreamingRequestError(
                "stream=true requires the streaming chat endpoint"
            )

        session = (
            self._session_store.create()
            if request.session_id is None
            else self._session_store.get(request.session_id)
        )
        for message in request.messages:
            session = self._session_store.append_message(session.session_id, message)

        result = await self._model_provider.complete(session.messages)
        self._session_store.append_message(session.session_id, result.message)
        return ChatResponse(
            request_id=str(uuid4()),
            session_id=session.session_id,
            message=result.message,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )


def error_response(code: str, message: str, retryable: bool) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(code=code, message=message, retryable=retryable)
    )
