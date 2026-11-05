import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.confidence import ConfidencePolicy, ThresholdConfidencePolicy
from app.contracts import ChatMessage
from app.embeddings import EmbeddingError, EmbeddingProvider
from app.document_processing import DocumentChunker
from app.providers import ChatModelProvider, ModelProviderError
from app.repositories import (
    KnowledgeDocumentRepository,
    RetrievalRecordRepository,
)
from app.data_models import KnowledgeDocument, RetrievalHit, RetrievalRecord
from app.reranking import Reranker, RerankerError
from app.retrieval import RetrievalResult
from app.vector_store import VectorMatch, VectorRecord, VectorStore, VectorStoreError


class KnowledgeBaseError(RuntimeError):
    pass


class KnowledgeQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class KnowledgeQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    query: str
    answer: str = Field(min_length=1)
    sources: list[KnowledgeSource] = Field(default_factory=list)
    confidence_accepted: bool
    confidence_score: float | None = None
    confidence_reason: str = Field(min_length=1)
    low_quality: bool


@dataclass(slots=True)
class KnowledgeBaseService:
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    model_provider: ChatModelProvider
    retrieval_repository: RetrievalRecordRepository | None = None
    document_repository: KnowledgeDocumentRepository | None = None
    reranker: Reranker | None = None
    rerank_candidate_limit: int = 20
    confidence_policy: ConfidencePolicy = field(
        default_factory=ThresholdConfidencePolicy
    )
    low_confidence_message: str = "暂时无法根据知识库确认答案，请补充更多信息。"

    async def query(self, request: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
        request_id = str(uuid4())
        started_at = time.perf_counter()
        try:
            query_vector = await self.embedding_provider.embed(request.query)
            retrieval_limit = (
                max(request.top_k, self.rerank_candidate_limit)
                if self.reranker is not None
                else request.top_k
            )
            matches = await self.vector_store.search(query_vector, retrieval_limit)
            if self.reranker is not None:
                matches = await self._rerank(request.query, matches, request.top_k)
            decision = self.confidence_policy.evaluate(matches)
            if not decision.accepted:
                answer = self.low_confidence_message
            else:
                answer = await self._generate_answer(request.query, matches)
        except (
            EmbeddingError,
            VectorStoreError,
            ModelProviderError,
            RerankerError,
        ) as exc:
            raise KnowledgeBaseError(str(exc)) from exc

        sources = (
            [self._to_source(match) for match in matches]
            if not matches or decision.accepted
            else []
        )
        if self.retrieval_repository is not None:
            await self.retrieval_repository.save(
                RetrievalRecord(
                    record_id=UUID(request_id),
                    query=request.query,
                    top_k=request.top_k,
                    hits=[
                        RetrievalHit(
                            document_id=match.document_id,
                            score=max(0.0, match.score),
                            rank=rank,
                        )
                        for rank, match in enumerate(matches, start=1)
                    ],
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    created_at=datetime.now(timezone.utc),
                )
            )

        return KnowledgeQueryResponse(
            request_id=request_id,
            query=request.query,
            answer=answer,
            sources=sources,
            confidence_accepted=decision.accepted,
            confidence_score=decision.top_score,
            confidence_reason=decision.reason,
            low_quality=not decision.accepted,
        )

    async def index_document(
        self,
        *,
        document_id: UUID,
        title: str,
        content: str,
        source: str,
        metadata: dict[str, str] | None = None,
    ) -> int:
        if self.document_repository is None:
            raise KnowledgeBaseError("document indexing is not configured")
        chunks = DocumentChunker().split(content)
        try:
            vectors = await self.embedding_provider.embed_batch(
                [chunk.content for chunk in chunks]
            )
            records = [
                VectorRecord(
                    record_id=f"{document_id}:{chunk.index}",
                    document_id=document_id,
                    chunk_index=chunk.index,
                    content=chunk.content,
                    vector=vector,
                    metadata={"source": source, **(metadata or {})},
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            await self.vector_store.upsert(records)
            now = datetime.now(timezone.utc)
            await self.document_repository.save(
                KnowledgeDocument(
                    document_id=document_id,
                    title=title,
                    content=content,
                    source=source,
                    metadata=metadata or {},
                    created_at=now,
                    updated_at=now,
                )
            )
        except (EmbeddingError, VectorStoreError, ValueError) as exc:
            raise KnowledgeBaseError(str(exc)) from exc
        return len(records)

    async def _rerank(
        self,
        query: str,
        matches: Sequence[VectorMatch],
        top_k: int,
    ) -> list[VectorMatch]:
        candidates = [
            RetrievalResult(
                record_id=match.record_id,
                document_id=match.document_id,
                chunk_index=match.chunk_index,
                content=match.content,
                score=match.score,
                method="vector",
                metadata=match.metadata,
            )
            for match in matches
        ]
        ranked = await self.reranker.rerank(query, candidates, top_k=top_k)
        return [
            VectorMatch(
                record_id=result.record_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
            )
            for result in ranked
        ]

    async def _generate_answer(
        self, query: str, matches: Sequence[VectorMatch]
    ) -> str:
        context = "\n\n".join(
            f"[来源 {index}] {match.content}" for index, match in enumerate(matches, 1)
        )
        if not context:
            context = "未检索到相关知识片段。"
        result = await self.model_provider.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "你是电商客服知识库助手。仅根据给定知识片段回答，"
                        "无法确认时明确说明信息不足。\n\n"
                        f"知识片段：\n{context}"
                    ),
                ),
                ChatMessage(role="user", content=query),
            ]
        )
        return result.message.content

    @staticmethod
    def _to_source(match: VectorMatch) -> KnowledgeSource:
        return KnowledgeSource(
            document_id=match.document_id,
            chunk_index=match.chunk_index,
            content=match.content,
            score=match.score,
            metadata=dict(match.metadata),
        )
