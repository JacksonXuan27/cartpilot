import asyncio
import json

import pytest

from app.document_import import build_parser, main
from app.document_processing import DocumentCleaner


def test_document_import_parser_requires_stable_source_and_sets_chunk_defaults():
    args = build_parser().parse_args(["policies.md", "--source", "refund-policy"])

    assert str(args.file) == "policies.md"
    assert args.source == "refund-policy"
    assert args.title is None
    assert args.chunk_size == 800
    assert args.overlap_chars == 120
    assert args.batch_size == 32


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_document_import_parser_rejects_non_positive_sizes(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["policies.md", "--source", "refund-policy", "--batch-size", value]
        )


def test_document_import_requires_milvus_for_persistent_indexing(monkeypatch):
    from app.config import Settings
    import app.document_import as document_import

    monkeypatch.setattr(
        document_import,
        "get_settings",
        lambda: Settings(_env_file=None, milvus_uri=None),
    )

    with pytest.raises(SystemExit):
        main(["policies.md", "--source", "refund-policy"])


def test_document_import_indexes_utf8_file_and_emits_json_result(
    tmp_path, monkeypatch, capsys
):
    from uuid import UUID

    import app.document_import as document_import
    from app.config import Settings
    from app.database import DatabaseManager
    from app.embeddings import HashEmbeddingProvider
    from app.repositories import KnowledgeDocumentRepository, initialize_schema
    from app.vector_store import InMemoryVectorStore

    source_file = tmp_path / "refund-policy.md"
    source_file.write_text(
        "\ufeff退款申请通过后，款项将在三个工作日内原路退回。",
        encoding="utf-8",
    )
    expected_content = DocumentCleaner().clean(
        source_file.read_text(encoding="utf-8-sig")
    )
    database_url = f"sqlite:///{tmp_path / 'documents.db'}"
    settings = Settings(
        _env_file=None,
        milvus_uri="http://milvus:19530",
        database_url=database_url,
        embedding_dimension=16,
    )
    vector_store = InMemoryVectorStore(dimension=16)
    monkeypatch.setattr(document_import, "get_settings", lambda: settings)
    monkeypatch.setattr(
        document_import,
        "create_embedding_provider",
        lambda current_settings: HashEmbeddingProvider(dimension=16),
    )
    monkeypatch.setattr(
        document_import,
        "create_vector_store",
        lambda current_settings, *, dimension: vector_store,
    )

    exit_code = main(
        [
            str(source_file),
            "--source",
            "refund-policy",
            "--chunk-size",
            "64",
            "--overlap-chars",
            "8",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    database = DatabaseManager(database_url)
    database.open()
    initialize_schema(database)
    try:
        document = asyncio.run(
            KnowledgeDocumentRepository(database).get(UUID(output["document_id"]))
        )
    finally:
        database.close()

    assert exit_code == 0
    assert output["version"] == 1
    assert output["chunk_count"] == output["vector_count"] == 1
    assert document.content == expected_content
    assert document.source == "refund-policy"
