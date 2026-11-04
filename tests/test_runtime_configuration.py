import json
from uuid import UUID

import httpx
import pytest
from langchain_openai import ChatOpenAI

import app.main as main
import app.providers as providers
from app.config import Settings
from app.providers import StubModelProvider
from app.vector_store import InMemoryVectorStore, VectorRecord


def test_runtime_uses_local_defaults_without_credentials():
    application = main.create_app(Settings(_env_file=None))

    service = application.state.knowledge_base_service
    assert isinstance(service.model_provider, StubModelProvider)
    assert isinstance(service.vector_store, InMemoryVectorStore)
    assert service.embedding_provider.dimension == 64
    assert application.state.knowledge_base_service.reranker is not None
    assert application.state.knowledge_base_service.query_rewriter is not None


def test_settings_loads_runtime_choices_and_short_embedding_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHAT_PROVIDER=openai-compatible\n"
        "CHAT_MODEL=chat-test\n"
        "CHAT_BASE_URL=https://chat.example/v1\n"
        "CHAT_API_KEY=chat-test-key\n"
        "EMBEDDING_PROVIDER=openai-compatible\n"
        "EMBEDDING_API_KEY=\n"
        "EMBED_API_KEY=embedding-test-key\n"
        "EMBEDDING_DIMENSION=3\n"
        "MILVUS_URI=http://milvus:19530\n"
        "RERANK_PROVIDER=siliconflow\n"
        "RERANK_API_KEY=rerank-test-key\n",
        encoding="utf-8",
    )
    for line in env_file.read_text(encoding="utf-8").splitlines():
        monkeypatch.delenv(line.split("=", 1)[0], raising=False)

    settings = Settings(_env_file=env_file)

    assert settings.chat_provider == "openai-compatible"
    assert settings.embedding_provider == "openai-compatible"
    assert settings.embedding_api_key == "embedding-test-key"
    assert settings.embedding_dimension == 3
    assert settings.milvus_uri == "http://milvus:19530"
    assert settings.rerank_provider == "siliconflow"
    assert "chat-test-key" not in repr(settings)
    assert "embedding-test-key" not in repr(settings)
    assert "rerank-test-key" not in repr(settings)


@pytest.mark.parametrize(
    ("overrides", "missing_name"),
    [
        ({"chat_provider": "openai-compatible", "chat_api_key": ""}, "CHAT_API_KEY"),
        (
            {"embedding_provider": "openai-compatible", "embedding_api_key": ""},
            "EMBEDDING_API_KEY",
        ),
        ({"rerank_provider": "siliconflow", "rerank_api_key": ""}, "RERANK_API_KEY"),
    ],
)
def test_remote_runtime_rejects_missing_credentials(overrides, missing_name):
    settings = Settings(
        _env_file=None,
        chat_model="chat-test",
        chat_base_url="https://chat.example/v1",
        **({"chat_api_key": "chat-test-key"} if "chat_api_key" not in overrides else {}),
        **overrides,
    )

    with pytest.raises(ValueError, match=missing_name):
        main.create_app(settings)


class MilvusBoundaryClient:
    def __init__(self):
        self.rows = {}
        self.collections = set()
        self.closed = False

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, **parameters):
        if parameters.get("id_type") != "string":
            raise ValueError("string record IDs require a string primary key")
        self.collections.add(parameters["collection_name"])

    def upsert(self, *, collection_name, data):
        for row in data:
            self.rows[row["id"]] = row
        return {"upsert_count": len(data)}

    def search(self, *, collection_name, data, limit, **parameters):
        return [[{"id": row["id"], "distance": 0.9, "entity": row}
                 for row in list(self.rows.values())[:limit]]]

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_configured_app_serves_chat_stream_and_rag_and_closes_milvus(monkeypatch):
    import app.vector_store_factory as vector_store_factory

    requests = []

    def handle_request(request):
        requests.append(request)
        payload = json.loads(request.content)
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": [1.0, 0.0, 0.0]}
                for index in range(len(payload["input"]))
            ]})
        if request.url.path == "/v1/rerank":
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.95}
            ]})
        assert request.url.path == "/v1/chat/completions"
        if payload.get("stream"):
            chunks = [
                {"id": "chat-test", "object": "chat.completion.chunk", "created": 0,
                 "model": "chat-test", "choices": [{"index": 0,
                 "delta": {"role": "assistant", "content": "物流可查询。"},
                 "finish_reason": None}]},
                {"id": "chat-test", "object": "chat.completion.chunk", "created": 0,
                 "model": "chat-test", "choices": [{"index": 0,
                 "delta": {}, "finish_reason": "stop"}]},
            ]
            content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(200, text=content + "data: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": "chat-test", "object": "chat.completion", "created": 0,
            "model": "chat-test", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "物流可在订单详情查询。"},
            "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        })

    milvus_client = MilvusBoundaryClient()
    connected_uris = []

    def connect_milvus(uri):
        connected_uris.append(uri)
        return milvus_client

    monkeypatch.setattr(vector_store_factory, "create_milvus_client", connect_milvus)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as upstream:
        monkeypatch.setattr(
            providers, "ChatOpenAI",
            lambda **parameters: ChatOpenAI(
                **parameters, http_async_client=upstream, max_retries=0
            ),
        )
        application = main.create_app(Settings(
            _env_file=None,
            chat_provider="openai-compatible", chat_model="chat-test",
            chat_base_url="https://chat.example/v1", chat_api_key="chat-test-key",
            embedding_provider="openai-compatible", embedding_dimension=3,
            embedding_api_key="embedding-test-key",
            embedding_base_url="https://embedding.example/v1",
            milvus_uri="http://milvus:19530",
            rerank_provider="siliconflow", rerank_api_key="rerank-test-key",
            rerank_base_url="https://rerank.example/v1",
        ))
        service = application.state.knowledge_base_service
        service.embedding_provider.client = upstream
        service.reranker.client = upstream

        async with application.router.lifespan_context(application):
            vector = await service.embedding_provider.embed("物流查询")
            record = VectorRecord(
                record_id="logistics-1", document_id=UUID(int=1), chunk_index=0,
                content="物流可在订单详情查询。", vector=vector,
                metadata={"source": "logistics.md"},
            )
            await service.vector_store.upsert([record])
            await service.vector_store.upsert([record])
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                response = await client.post("/chat", json={
                    "messages": [{"role": "user", "content": "你好"}]
                })
                assert response.status_code == 200
                assert response.json()["message"]["content"] == "物流可在订单详情查询。"
                assert response.json()["usage"]["total_tokens"] == 10
                response = await client.post("/chat/stream", json={
                    "messages": [{"role": "user", "content": "查快递"}], "stream": True
                })
                assert response.status_code == 200
                assert "event: done" in response.text
                assert "\\u7269\\u6d41" in response.text
                response = await client.post("/knowledge/query", json={"query": "查快递"})
                assert response.status_code == 200
                assert response.json()["sources"][0]["content"] == "物流可在订单详情查询。"
                assert response.json()["sources"][0]["score"] == 0.95
                assert response.json()["answer"] == "物流可在订单详情查询。"

    assert connected_uris == ["http://milvus:19530"]
    assert len(milvus_client.rows) == 1
    assert milvus_client.closed
    queries = [json.loads(request.content)["input"] for request in requests
               if request.url.path == "/v1/embeddings"]
    assert ["查快递"] in queries
    assert ["查询物流"] in queries
    assert any(request.url.host == "rerank.example" for request in requests)
    assert all(request.headers["authorization"] == "Bearer chat-test-key"
               for request in requests if request.url.host == "chat.example")
