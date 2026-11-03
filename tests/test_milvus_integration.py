import asyncio
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.embeddings import HashEmbeddingProvider
from app.knowledge_base import KnowledgeBaseService
from app.main import app
from app.providers import StubModelProvider
from app.vector_store import MilvusVectorStore, VectorRecord
from app.vector_store_factory import create_milvus_client


MILVUS_TEST_URI = os.getenv("CARTPILOT_MILVUS_TEST_URI")


@pytest.mark.skipif(
    not MILVUS_TEST_URI,
    reason="set CARTPILOT_MILVUS_TEST_URI to run the live Milvus integration test",
)
def test_knowledge_query_round_trips_through_live_milvus():
    collection_name = f"cartpilot_it_{uuid4().hex}"
    client = create_milvus_client(MILVUS_TEST_URI)
    embedding_provider = HashEmbeddingProvider(dimension=64)
    vector_store = MilvusVectorStore(
        client=client,
        collection_name=collection_name,
        dimension=embedding_provider.dimension,
    )
    document_id = uuid4()
    record_id = f"{document_id}:0"
    content = "集成测试退款政策：订单退款将在三个工作日内原路退回。"
    previous_service = app.state.knowledge_base_service
    app.state.knowledge_base_service = KnowledgeBaseService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        model_provider=StubModelProvider(reply="退款会原路退回。"),
    )

    try:
        vector = asyncio.run(embedding_provider.embed(content))
        asyncio.run(
            vector_store.upsert(
                [
                    VectorRecord(
                        record_id=record_id,
                        document_id=document_id,
                        chunk_index=0,
                        content=content,
                        vector=vector,
                        metadata={"source": "integration-test"},
                    )
                ]
            )
        )
        matches = asyncio.run(vector_store.search(vector, top_k=1))
        assert matches and matches[0].record_id == record_id, matches
        response = TestClient(app).post(
            "/knowledge/query",
            json={"query": content, "top_k": 1},
        )

        assert response.status_code == 200
        assert response.json()["sources"], response.json()
        assert response.json()["sources"][0]["document_id"] == str(document_id)
        assert response.json()["sources"][0]["metadata"]["source"] == "integration-test"
        assert response.json()["confidence_accepted"] is True
    finally:
        app.state.knowledge_base_service = previous_service
        try:
            client.drop_collection(collection_name)
        finally:
            client.close()
