from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.faq_tools import FAQEntry, FAQSearchTool, InMemoryFAQRepository
from app.logistics_tools import (
    InMemoryShipmentRepository,
    LogisticsQueryTool,
    Shipment,
)
from app.order_tools import InMemoryOrderRepository, Order, OrderQueryTool
from app.tool_registry import (
    ToolAlreadyRegisteredError,
    ToolArgumentError,
    ToolNotFoundError,
    ToolRegistry,
)


def make_order() -> Order:
    return Order(
        order_id="ORD-4001",
        status="shipped",
        product_name="Mechanical keyboard",
        quantity=1,
        total_amount=599.0,
        currency="CNY",
        placed_at=datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc),
    )


def make_shipment() -> Shipment:
    return Shipment(
        order_id="ORD-4001",
        carrier="CartPilot Express",
        tracking_number="CPX4001",
        status="in_transit",
    )


def make_faq() -> FAQEntry:
    return FAQEntry(
        faq_id="FAQ-4001",
        category="refund",
        title="Refund timing",
        answer="Refunds usually arrive in three to five business days.",
        keywords=("refund", "timing"),
    )


@pytest.mark.asyncio
async def test_registry_discovers_and_executes_registered_tools():
    registry = ToolRegistry(
        [
            OrderQueryTool(InMemoryOrderRepository({"ORD-4001": make_order()})),
            LogisticsQueryTool(
                InMemoryShipmentRepository({"ORD-4001": make_shipment()})
            ),
            FAQSearchTool(InMemoryFAQRepository([make_faq()])),
        ]
    )

    definitions = registry.definitions()
    assert [definition.name for definition in definitions] == [
        "faq.search",
        "logistics.query",
        "order.query",
    ]
    assert definitions[-1].input_schema["properties"]["order_id"]["type"] == "string"

    order = await registry.execute("order.query", {"order_id": "ORD-4001"})
    assert order.order_id == "ORD-4001"


def test_registry_rejects_duplicate_and_missing_tools():
    tool = OrderQueryTool(InMemoryOrderRepository())
    registry = ToolRegistry([tool])

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(tool)

    with pytest.raises(ToolNotFoundError):
        registry.get("unknown.tool")


@pytest.mark.asyncio
async def test_registry_wraps_validation_errors_before_execution():
    registry = ToolRegistry(
        [OrderQueryTool(InMemoryOrderRepository({"ORD-4001": make_order()}))]
    )

    with pytest.raises(ToolArgumentError) as error:
        await registry.execute("order.query", {"order_id": "bad id"})

    assert error.value.tool_name == "order.query"
    assert error.value.errors
    assert isinstance(error.value.__cause__, ValidationError)


def test_registry_definitions_are_safe_to_mutate():
    registry = ToolRegistry(
        [OrderQueryTool(InMemoryOrderRepository({"ORD-4001": make_order()}))]
    )

    definitions = registry.definitions()
    definitions[0].input_schema["properties"]["order_id"]["type"] = "integer"

    fresh = registry.definitions()
    assert fresh[-1].input_schema["properties"]["order_id"]["type"] == "string"
