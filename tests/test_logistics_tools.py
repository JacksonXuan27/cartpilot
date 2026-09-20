from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.logistics_tools import (
    InMemoryShipmentRepository,
    LogisticsQueryInput,
    LogisticsQueryTool,
    Shipment,
    ShipmentNotFoundError,
    TrackingEvent,
)


def make_shipment(order_id: str = "ORD-1001") -> Shipment:
    return Shipment(
        order_id=order_id,
        carrier="CartPilot Express",
        tracking_number="CPX1001001",
        status="in_transit",
        estimated_delivery=datetime(2026, 9, 22, 18, 0, tzinfo=timezone.utc),
        events=[
            TrackingEvent(
                occurred_at=datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc),
                location="Shanghai",
                description="Package accepted by carrier",
            )
        ],
    )


@pytest.mark.asyncio
async def test_logistics_query_tool_returns_shipment_details():
    repository = InMemoryShipmentRepository({"ORD-1001": make_shipment()})
    tool = LogisticsQueryTool(repository)

    result = await tool.execute({"order_id": "ORD-1001"})

    assert tool.name == "logistics.query"
    assert result.status == "in_transit"
    assert result.carrier == "CartPilot Express"
    assert result.events[0].location == "Shanghai"


@pytest.mark.asyncio
async def test_logistics_query_tool_rejects_unknown_orders():
    tool = LogisticsQueryTool(InMemoryShipmentRepository())

    with pytest.raises(ShipmentNotFoundError, match="ORD-404"):
        await tool.execute({"order_id": "ORD-404"})


@pytest.mark.asyncio
async def test_logistics_query_tool_validates_arguments():
    tool = LogisticsQueryTool(InMemoryShipmentRepository())

    with pytest.raises(ValidationError):
        await tool.execute({"order_id": "bad id"})

    with pytest.raises(ValidationError):
        await tool.execute({"order_id": "ORD-1001", "unexpected": True})


@pytest.mark.asyncio
async def test_repository_returns_an_isolated_shipment_copy():
    repository = InMemoryShipmentRepository({"ORD-1001": make_shipment()})

    result = await repository.get("ORD-1001")
    result.events[0].description = "changed locally"

    stored = await repository.get("ORD-1001")
    assert stored.events[0].description == "Package accepted by carrier"


def test_shipment_rejects_invalid_status_and_tracking_number():
    with pytest.raises(ValidationError):
        Shipment(
            order_id="ORD-1001",
            carrier="CartPilot Express",
            tracking_number="CPX1001001",
            status="unknown",
        )

    with pytest.raises(ValidationError):
        LogisticsQueryInput(order_id="")
