from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.order_tools import (
    InMemoryOrderRepository,
    Order,
    OrderNotFoundError,
    OrderQueryInput,
    OrderQueryTool,
)


def make_order(order_id: str = "ORD-1001") -> Order:
    return Order(
        order_id=order_id,
        status="shipped",
        product_name="Wireless keyboard",
        quantity=1,
        total_amount=199.0,
        currency="CNY",
        placed_at=datetime(2026, 9, 20, 10, 30, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_order_query_tool_returns_a_matching_order():
    repository = InMemoryOrderRepository({"ORD-1001": make_order()})
    tool = OrderQueryTool(repository)

    result = await tool.execute({"order_id": "ORD-1001"})

    assert tool.name == "order.query"
    assert result.status == "shipped"
    assert result.product_name == "Wireless keyboard"


@pytest.mark.asyncio
async def test_order_query_tool_rejects_unknown_orders():
    tool = OrderQueryTool(InMemoryOrderRepository())

    with pytest.raises(OrderNotFoundError, match="ORD-404"):
        await tool.execute({"order_id": "ORD-404"})


@pytest.mark.asyncio
async def test_order_query_tool_validates_arguments_before_repository_access():
    tool = OrderQueryTool(InMemoryOrderRepository())

    with pytest.raises(ValidationError):
        await tool.execute({"order_id": "bad id"})

    with pytest.raises(ValidationError):
        await tool.execute({"order_id": "ORD-1001", "unexpected": True})


@pytest.mark.asyncio
async def test_repository_returns_an_isolated_order_copy():
    repository = InMemoryOrderRepository({"ORD-1001": make_order()})

    result = await repository.get("ORD-1001")
    result.product_name = "changed locally"

    stored = await repository.get("ORD-1001")
    assert stored.product_name == "Wireless keyboard"


def test_order_rejects_invalid_status_and_non_positive_quantity():
    with pytest.raises(ValidationError):
        Order(
            order_id="ORD-1001",
            status="unknown",
            product_name="Wireless keyboard",
            quantity=1,
            total_amount=199.0,
            currency="CNY",
            placed_at=datetime(2026, 9, 20, 10, 30, tzinfo=timezone.utc),
        )

    with pytest.raises(ValidationError):
        Order(
            order_id="ORD-1001",
            status="shipped",
            product_name="Wireless keyboard",
            quantity=0,
            total_amount=199.0,
            currency="CNY",
            placed_at=datetime(2026, 9, 20, 10, 30, tzinfo=timezone.utc),
        )
