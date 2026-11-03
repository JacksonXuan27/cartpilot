from uuid import UUID, uuid4

import pytest

from app.vector_store import (
    InMemoryVectorStore,
    MilvusVectorStore,
    VectorDimensionError,
    VectorRecord,
    VectorStore,
    VectorStoreResponseError,
)


DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000101")


def make_record(record_id: str, vector: tuple[float, ...]) -> VectorRecord:
    return VectorRecord(
        record_id=record_id,
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content=record_id,
        vector=vector,
        metadata={"source": "test"},
    )


@pytest.mark.asyncio
async def test_in_memory_store_implements_vector_store_and_ranks_matches():
    store = InMemoryVectorStore(dimension=3)
    assert isinstance(store, VectorStore)

    await store.upsert(
        [
            make_record("near", (1.0, 0.0, 0.0)),
            make_record("far", (0.0, 1.0, 0.0)),
        ]
    )

    matches = await store.search((0.9, 0.1, 0.0), top_k=2)

    assert [match.record_id for match in matches] == ["near", "far"]
    assert matches[0].score > matches[1].score


@pytest.mark.asyncio
async def test_in_memory_store_upsert_replaces_existing_record():
    store = InMemoryVectorStore(dimension=2)
    await store.upsert([make_record("same", (1.0, 0.0))])
    await store.upsert([make_record("same", (0.0, 1.0))])

    matches = await store.search((0.0, 1.0))

    assert matches[0].record_id == "same"
    assert matches[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_in_memory_store_deletes_document_chunks_from_requested_index():
    store = InMemoryVectorStore(dimension=2)
    document_id = uuid4()
    records = [
        VectorRecord(
            record_id=f"chunk-{chunk_index}",
            document_id=document_id,
            chunk_index=chunk_index,
            content=f"chunk {chunk_index}",
            vector=(1.0, 0.0),
        )
        for chunk_index in range(3)
    ]
    await store.upsert(records)

    delete_document_chunks = getattr(store, "delete_document_chunks", None)
    assert callable(delete_document_chunks)
    deleted = await delete_document_chunks(document_id, from_chunk_index=2)

    assert deleted == 1
    matches = await store.search((1.0, 0.0), top_k=10)
    assert [match.chunk_index for match in matches] == [0, 1]


class FakeMilvusClient:
    def __init__(self, search_response=None, insert_response=None):
        self.collections: set[str] = set()
        self.created: list[dict[str, object]] = []
        self.inserted: list[dict[str, object]] = []
        self.upserted: list[dict[str, object]] = []
        self.searches: list[dict[str, object]] = []
        self.search_response = search_response if search_response is not None else [[]]
        self.insert_response = insert_response if insert_response is not None else {"code": 0}

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs):
        self.collections.add(str(kwargs["collection_name"]))
        self.created.append(kwargs)
        return {"code": 0}

    def insert(self, **kwargs):
        self.inserted.append(kwargs)
        return self.insert_response

    def upsert(self, **kwargs):
        self.upserted.append(kwargs)
        return self.insert_response

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return self.search_response

    def delete(self, **kwargs):
        self.deleted = kwargs
        return {"delete_count": 1}


@pytest.mark.asyncio
async def test_milvus_store_creates_string_keyed_collection_upserts_and_searches():
    document_id = uuid4()
    client = FakeMilvusClient(
        search_response=[
            [
                {
                    "id": "chunk-1",
                    "distance": 0.91,
                    "entity": {
                        "document_id": str(document_id),
                        "chunk_index": 2,
                        "content": "refund policy",
                        "metadata": '{"source":"policy.md"}',
                    },
                }
            ]
        ]
    )
    store = MilvusVectorStore(client, "cartpilot_chunks", dimension=3)

    written = await store.upsert([make_record("chunk-1", (1.0, 0.0, 0.0))])
    matches = await store.search((1.0, 0.0, 0.0))

    assert written == 1
    assert client.created[0]["dimension"] == 3
    assert client.created[0]["id_type"] == "string"
    assert client.upserted[0]["data"][0]["id"] == "chunk-1"
    assert client.inserted == []
    assert client.searches[0]["limit"] == 5
    assert client.searches[0]["consistency_level"] == "Strong"
    assert matches[0].document_id == document_id
    assert matches[0].metadata == {"source": "policy.md"}


@pytest.mark.asyncio
async def test_milvus_store_validates_dimensions_and_response_errors():
    client = FakeMilvusClient(insert_response={"code": 1, "message": "unavailable"})
    store = MilvusVectorStore(client, "cartpilot_chunks", dimension=2)

    with pytest.raises(VectorDimensionError, match="dimension"):
        await store.upsert([make_record("bad", (1.0,))])

    with pytest.raises(VectorStoreResponseError, match="unavailable"):
        await store.upsert([make_record("ok", (1.0, 0.0))])


@pytest.mark.asyncio
async def test_milvus_store_upsert_reuses_the_primary_key_for_repeated_records():
    client = FakeMilvusClient()
    store = MilvusVectorStore(client, "cartpilot_chunks", dimension=2)
    record = make_record("stable-id", (1.0, 0.0))

    await store.upsert([record])
    await store.upsert([record])

    assert [call["data"][0]["id"] for call in client.upserted] == [
        "stable-id",
        "stable-id",
    ]


@pytest.mark.asyncio
async def test_milvus_store_deletes_only_trailing_document_chunks():
    client = FakeMilvusClient()
    store = MilvusVectorStore(client, "cartpilot_chunks", dimension=2)
    document_id = uuid4()

    deleted = await store.delete_document_chunks(document_id, from_chunk_index=3)

    assert deleted == 1
    assert client.deleted == {
        "collection_name": "cartpilot_chunks",
        "filter": f'document_id == "{document_id}" and chunk_index >= 3',
    }
