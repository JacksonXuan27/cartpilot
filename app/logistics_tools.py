from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


ShipmentStatus = Literal[
    "pending",
    "in_transit",
    "out_for_delivery",
    "delivered",
    "exception",
]


class ShipmentNotFoundError(LookupError):
    pass


class TrackingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occurred_at: datetime
    location: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Shipment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    carrier: str = Field(min_length=1)
    tracking_number: str = Field(min_length=3)
    status: ShipmentStatus
    estimated_delivery: datetime | None = None
    events: list[TrackingEvent] = Field(default_factory=list)


class ShipmentRepository(Protocol):
    async def get(self, order_id: str) -> Shipment:
        """Return shipment details or raise ShipmentNotFoundError."""


class InMemoryShipmentRepository:
    def __init__(self, shipments: Mapping[str, Shipment] | None = None) -> None:
        self._shipments = {
            order_id: shipment.model_copy(deep=True)
            for order_id, shipment in (shipments or {}).items()
        }

    async def get(self, order_id: str) -> Shipment:
        shipment = self._shipments.get(order_id)
        if shipment is None:
            raise ShipmentNotFoundError(order_id)
        return shipment.model_copy(deep=True)


class LogisticsQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")


class LogisticsQueryTool:
    name = "logistics.query"
    description = "Query shipment status and tracking events for an e-commerce order."
    input_model = LogisticsQueryInput

    def __init__(self, repository: ShipmentRepository) -> None:
        self._repository = repository

    async def execute(self, arguments: Mapping[str, object]) -> Shipment:
        query = self.input_model.model_validate(arguments)
        return await self._repository.get(query.order_id)
