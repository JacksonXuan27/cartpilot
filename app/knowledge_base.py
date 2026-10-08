import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.confidence import ConfidencePolicy, ThresholdConfidencePolicy
from app.contracts import ChatMessage
from app.embeddings import EmbeddingError, EmbeddingProvider
from app.providers import ChatModelProvider, ModelProviderError
from app.repositories import RetrievalRecordRepository
from app.data_models import RetrievalHit, RetrievalRecord
from app.vector_store import VectorMatch, VectorStore, VectorStoreError


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
    confidence_policy: ConfidencePolicy = field(
        default_factory=ThresholdConfidencePolicy
    )
    low_confidence_message: str = "暂时无法根据知识库确认答案，请补充更多信息。"

    async def query(self, request: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
        request_id = str(uuid4())
        started_at = time.perf_counter()
        try:
            query_vector = await self.embedding_provider.embed(request.query)
            matches = await self.vector_store.search(query_vector, request.top_k)
            decision = self.confidence_policy.evaluate(matches)
            if not decision.accepted:
                answer = self.low_confidence_message
            else:
                answer = await self._generate_answer(request.query, matches)
        except (EmbeddingError, VectorStoreError, ModelProviderError) as exc:
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
