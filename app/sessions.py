from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.contracts import ChatMessage


class SessionNotFoundError(LookupError):
    pass


class SessionAlreadyExistsError(ValueError):
    pass


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def create(self, session_id: str | None = None) -> Session:
        identifier = session_id or str(uuid4())
        now = datetime.now(timezone.utc)
        session = Session(
            session_id=identifier,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            if identifier in self._sessions:
                raise SessionAlreadyExistsError(identifier)
            self._sessions[identifier] = session
            return session.model_copy(deep=True)

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            return session.model_copy(deep=True)

    def append_message(self, session_id: str, message: ChatMessage) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            session.messages.append(message)
            session.updated_at = datetime.now(timezone.utc)
            return session.model_copy(deep=True)

    def delete(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise SessionNotFoundError(session_id)
