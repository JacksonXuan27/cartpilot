from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.order_tools import OrderStatus


class OrderRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    status: OrderStatus
    product_name: str = Field(min_length=1, max_length=200)
    quantity: int = Field(ge=1)
    total_amount: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    placed_at: datetime
    created_at: datetime
    updated_at: datetime


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=500)
    version: int = Field(default=1, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    score: float = Field(ge=0)
    rank: int = Field(ge=1)


class RetrievalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: UUID
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(ge=1, le=100)
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    created_at: datetime


DataModelKind = Literal["order", "knowledge_document", "retrieval_record"]
