from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


OrderStatus = Literal[
    "pending",
    "paid",
    "shipped",
    "delivered",
    "cancelled",
    "refunded",
]


class OrderNotFoundError(LookupError):
    pass


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    status: OrderStatus
    product_name: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    total_amount: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    placed_at: datetime


class OrderRepository(Protocol):
    async def get(self, order_id: str) -> Order:
        """Return an order or raise OrderNotFoundError."""


class InMemoryOrderRepository:
    def __init__(self, orders: Mapping[str, Order] | None = None) -> None:
        self._orders = {
            order_id: order.model_copy(deep=True)
            for order_id, order in (orders or {}).items()
        }

    async def get(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise OrderNotFoundError(order_id)
        return order.model_copy(deep=True)


class OrderQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")


class OrderQueryTool:
    name = "order.query"
    description = "Query the current status and summary of an e-commerce order."
    input_model = OrderQueryInput

    def __init__(self, repository: OrderRepository) -> None:
        self._repository = repository

    async def execute(self, arguments: Mapping[str, object]) -> Order:
        query = self.input_model.model_validate(arguments)
        return await self._repository.get(query.order_id)
