import sqlite3
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


class DatabaseError(RuntimeError):
    pass


class DatabaseConfigurationError(DatabaseError, ValueError):
    pass


class DatabaseNotInitializedError(DatabaseError):
    pass


@dataclass(slots=True)
class DatabaseManager:
    database_url: str
    timeout_seconds: float = 5.0
    _connection: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.database_url.strip():
            raise DatabaseConfigurationError("database_url cannot be empty")
        if self.timeout_seconds <= 0:
            raise DatabaseConfigurationError("timeout_seconds must be positive")

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise DatabaseNotInitializedError("database is not open")
        return self._connection

    def open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection

        path = self._sqlite_path()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            path,
            timeout=self.timeout_seconds,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection

    def close(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connection
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    @asynccontextmanager
    async def lifecycle(self):
        self.open()
        try:
            yield self
        finally:
            self.close()

    def _sqlite_path(self) -> str:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise DatabaseConfigurationError(
                "only sqlite:/// database URLs are supported"
            )

        raw_path = self.database_url[len(prefix) :]
        if not raw_path:
            raise DatabaseConfigurationError("sqlite database path cannot be empty")
        if raw_path == ":memory:":
            return raw_path
        return str(Path(raw_path))
