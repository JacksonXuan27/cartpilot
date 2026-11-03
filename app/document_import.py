import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.database import DatabaseManager
from app.document_ingestion import DocumentIngestionService
from app.embeddings import create_embedding_provider
from app.repositories import KnowledgeDocumentRepository, initialize_schema
from app.vector_store import MilvusVectorStore
from app.vector_store_factory import create_vector_store


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean, chunk, embed, and index a UTF-8 knowledge document."
    )
    parser.add_argument("file", type=Path, help="path to a UTF-8 text or Markdown file")
    parser.add_argument(
        "--source",
        required=True,
        help="stable source identifier used to update this document on reimport",
    )
    parser.add_argument("--title", help="document title; defaults to the filename")
    parser.add_argument("--chunk-size", type=_positive_integer, default=800)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--batch-size", type=_positive_integer, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.overlap_chars < 0 or args.overlap_chars >= args.chunk_size:
        parser.error("--overlap-chars must be non-negative and smaller than --chunk-size")

    settings = get_settings()
    if settings.milvus_uri is None or not settings.milvus_uri.strip():
        parser.error("MILVUS_URI is required for persistent document indexing")

    database = DatabaseManager(
        settings.database_url,
        timeout_seconds=settings.database_timeout_seconds,
    )
    vector_store = None
    try:
        content = args.file.read_text(encoding="utf-8-sig")
        database.open()
        initialize_schema(database)
        embedding_provider = create_embedding_provider(settings)
        vector_store = create_vector_store(
            settings,
            dimension=embedding_provider.dimension,
        )
        service = DocumentIngestionService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            document_repository=KnowledgeDocumentRepository(database),
        )
        result = asyncio.run(
            service.ingest(
                title=args.title or args.file.stem,
                source=args.source,
                content=content,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                overlap_chars=args.overlap_chars,
            )
        )
        print(
            json.dumps(
                {
                    "document_id": str(result.document_id),
                    "version": result.version,
                    "chunk_count": result.chunk_count,
                    "vector_count": result.vector_count,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"document import failed: {exc}", file=sys.stderr)
        return 1
    finally:
        database.close()
        if isinstance(vector_store, MilvusVectorStore):
            close = getattr(vector_store.client, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
