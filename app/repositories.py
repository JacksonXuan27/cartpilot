import json
from uuid import UUID

from app.data_models import KnowledgeDocument, OrderRecord, RetrievalRecord
from app.database import DatabaseManager


class RepositoryError(RuntimeError):
    pass


class RecordNotFoundError(LookupError):
    pass


def initialize_schema(database: DatabaseManager) -> None:
    with database.transaction() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_knowledge_documents_source
                ON knowledge_documents (source);
            CREATE TABLE IF NOT EXISTS retrieval_records (
                record_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_retrieval_records_created_at
                ON retrieval_records (created_at);
            """
        )


class OrderRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def save(self, order: OrderRecord) -> OrderRecord:
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO orders (order_id, payload) VALUES (?, ?)
                ON CONFLICT(order_id) DO UPDATE SET payload = excluded.payload
                """,
                (order.order_id, order.model_dump_json()),
            )
        return order.model_copy(deep=True)

    async def get(self, order_id: str) -> OrderRecord:
        row = self._database.connection.execute(
            "SELECT payload FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"order not found: {order_id}")
        return OrderRecord.model_validate_json(row["payload"])

    async def list_recent(self, limit: int = 50) -> list[OrderRecord]:
        _validate_limit(limit)
        rows = self._database.connection.execute(
            "SELECT payload FROM orders ORDER BY order_id LIMIT ?", (limit,)
        ).fetchall()
        return [OrderRecord.model_validate_json(row["payload"]) for row in rows]


class KnowledgeDocumentRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def save(self, document: KnowledgeDocument) -> KnowledgeDocument:
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_documents
                    (document_id, title, source, payload) VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    source = excluded.source,
                    payload = excluded.payload
                """,
                (
                    str(document.document_id),
                    document.title,
                    document.source,
                    document.model_dump_json(),
                ),
            )
        return document.model_copy(deep=True)

    async def get(self, document_id: UUID) -> KnowledgeDocument:
        row = self._database.connection.execute(
            "SELECT payload FROM knowledge_documents WHERE document_id = ?",
            (str(document_id),),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"knowledge document not found: {document_id}")
        return KnowledgeDocument.model_validate_json(row["payload"])

    async def search(
        self, query: str = "", source: str | None = None, limit: int = 20
    ) -> list[KnowledgeDocument]:
        _validate_limit(limit)
        filters: list[str] = []
        parameters: list[str | int] = []
        if query.strip():
            filters.append("(title LIKE ? OR payload LIKE ?)")
            pattern = f"%{query.strip()}%"
            parameters.extend((pattern, pattern))
        if source is not None:
            filters.append("source = ?")
            parameters.append(source)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        parameters.append(limit)
        rows = self._database.connection.execute(
            f"SELECT payload FROM knowledge_documents {where_clause} "
            "ORDER BY title, document_id LIMIT ?",
            parameters,
        ).fetchall()
        return [KnowledgeDocument.model_validate_json(row["payload"]) for row in rows]


class RetrievalRecordRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def save(self, record: RetrievalRecord) -> RetrievalRecord:
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_records (record_id, created_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    str(record.record_id),
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )
        return record.model_copy(deep=True)

    async def get(self, record_id: UUID) -> RetrievalRecord:
        row = self._database.connection.execute(
            "SELECT payload FROM retrieval_records WHERE record_id = ?",
            (str(record_id),),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"retrieval record not found: {record_id}")
        return RetrievalRecord.model_validate_json(row["payload"])

    async def list_recent(self, limit: int = 50) -> list[RetrievalRecord]:
        _validate_limit(limit)
        rows = self._database.connection.execute(
            "SELECT payload FROM retrieval_records "
            "ORDER BY created_at DESC, record_id LIMIT ?",
            (limit,),
        ).fetchall()
        return [RetrievalRecord.model_validate_json(row["payload"]) for row in rows]


def _validate_limit(limit: int) -> None:
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
