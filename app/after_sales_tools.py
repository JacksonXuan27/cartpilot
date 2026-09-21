from collections.abc import Mapping
from datetime import datetime, timezone
from threading import RLock
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.order_tools import Order, OrderRepository, OrderStatus


AfterSalesAction = Literal["return", "exchange", "repair"]
AfterSalesRequestStatus = Literal["pending_review", "approved", "rejected", "completed"]


class AfterSalesToolError(ValueError):
    pass


class RefundNotEligibleError(AfterSalesToolError):
    pass


class RefundAmountError(AfterSalesToolError):
    pass


class AfterSalesRequestNotFoundError(LookupError):
    pass


class RefundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    reason: str = Field(min_length=1)
    status: AfterSalesRequestStatus = "pending_review"
    created_at: datetime
    updated_at: datetime


class AfterSalesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    action: AfterSalesAction
    reason: str = Field(min_length=1)
    status: AfterSalesRequestStatus = "pending_review"
    created_at: datetime
    updated_at: datetime


class RefundRepository(Protocol):
    async def create(self, request: RefundRequest) -> RefundRequest:
        """Persist and return a refund request."""

    async def get(self, request_id: str) -> RefundRequest:
        """Return a refund request or raise AfterSalesRequestNotFoundError."""


class AfterSalesRepository(Protocol):
    async def create(self, request: AfterSalesRequest) -> AfterSalesRequest:
        """Persist and return an after-sales request."""


class InMemoryRefundRepository:
    def __init__(self) -> None:
        self._requests: dict[str, RefundRequest] = {}
        self._lock = RLock()

    async def create(self, request: RefundRequest) -> RefundRequest:
        with self._lock:
            self._requests[request.request_id] = request.model_copy(deep=True)
            return request.model_copy(deep=True)

    async def get(self, request_id: str) -> RefundRequest:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise AfterSalesRequestNotFoundError(request_id)
            return request.model_copy(deep=True)


class InMemoryAfterSalesRepository:
    def __init__(self) -> None:
        self._requests: dict[str, AfterSalesRequest] = {}
        self._lock = RLock()

    async def create(self, request: AfterSalesRequest) -> AfterSalesRequest:
        with self._lock:
            self._requests[request.request_id] = request.model_copy(deep=True)
            return request.model_copy(deep=True)


class RefundRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    reason: str = Field(min_length=1)


class AfterSalesRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    action: AfterSalesAction
    reason: str = Field(min_length=1)


class RefundTool:
    name = "refund.create"
    description = "Create a reviewable refund request for an eligible order."
    input_model = RefundRequestInput

    def __init__(
        self,
        order_repository: OrderRepository,
        request_repository: RefundRepository,
    ) -> None:
        self._orders = order_repository
        self._requests = request_repository

    async def execute(self, arguments: Mapping[str, object]) -> RefundRequest:
        query = self.input_model.model_validate(arguments)
        order = await self._orders.get(query.order_id)
        _validate_refund_order(order)
        if query.currency.upper() != order.currency.upper():
            raise RefundAmountError("refund currency does not match the order")
        if query.amount > order.total_amount:
            raise RefundAmountError("refund amount cannot exceed the order total")

        now = datetime.now(timezone.utc)
        request = RefundRequest(
            request_id=f"refund-{uuid4()}",
            order_id=order.order_id,
            amount=query.amount,
            currency=order.currency,
            reason=query.reason,
            created_at=now,
            updated_at=now,
        )
        return await self._requests.create(request)


class AfterSalesTool:
    name = "after_sales.create"
    description = "Create a reviewable return, exchange, or repair request."
    input_model = AfterSalesRequestInput

    def __init__(
        self,
        order_repository: OrderRepository,
        request_repository: AfterSalesRepository,
    ) -> None:
        self._orders = order_repository
        self._requests = request_repository

    async def execute(self, arguments: Mapping[str, object]) -> AfterSalesRequest:
        query = self.input_model.model_validate(arguments)
        order = await self._orders.get(query.order_id)
        _validate_after_sales_order(order)
        now = datetime.now(timezone.utc)
        request = AfterSalesRequest(
            request_id=f"after-sales-{uuid4()}",
            order_id=order.order_id,
            action=query.action,
            reason=query.reason,
            created_at=now,
            updated_at=now,
        )
        return await self._requests.create(request)


def _validate_refund_order(order: Order) -> None:
    if order.status not in {"paid", "shipped", "delivered"}:
        raise RefundNotEligibleError(
            f"order status does not allow a refund request: {order.status}"
        )


def _validate_after_sales_order(order: Order) -> None:
    if order.status in {"cancelled", "refunded"}:
        raise RefundNotEligibleError(
            f"order status does not allow an after-sales request: {order.status}"
        )
