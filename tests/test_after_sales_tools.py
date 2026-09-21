from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.after_sales_tools import (
    AfterSalesRequestInput,
    AfterSalesTool,
    InMemoryAfterSalesRepository,
    InMemoryRefundRepository,
    RefundAmountError,
    RefundRequestInput,
    RefundNotEligibleError,
    RefundTool,
)
from app.order_tools import InMemoryOrderRepository, Order


def make_order(status: str = "delivered") -> Order:
    return Order(
        order_id="ORD-2001",
        status=status,
        product_name="Noise-cancelling headphones",
        quantity=1,
        total_amount=399.0,
        currency="CNY",
        placed_at=datetime(2026, 9, 20, 10, 30, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_refund_tool_creates_a_reviewable_request():
    tool = RefundTool(
        InMemoryOrderRepository({"ORD-2001": make_order()}),
        InMemoryRefundRepository(),
    )

    result = await tool.execute(
        {
            "order_id": "ORD-2001",
            "amount": 399,
            "currency": "CNY",
            "reason": "item arrived damaged",
        }
    )

    assert tool.name == "refund.create"
    assert result.request_id.startswith("refund-")
    assert result.status == "pending_review"
    assert result.amount == 399


@pytest.mark.asyncio
async def test_refund_tool_rejects_amount_currency_and_order_status_errors():
    tool = RefundTool(
        InMemoryOrderRepository({"ORD-2001": make_order(status="cancelled")}),
        InMemoryRefundRepository(),
    )

    with pytest.raises(RefundNotEligibleError):
        await tool.execute(
            {
                "order_id": "ORD-2001",
                "amount": 10,
                "currency": "CNY",
                "reason": "changed my mind",
            }
        )

    eligible_tool = RefundTool(
        InMemoryOrderRepository({"ORD-2001": make_order()}),
        InMemoryRefundRepository(),
    )
    with pytest.raises(RefundAmountError, match="exceed"):
        await eligible_tool.execute(
            {
                "order_id": "ORD-2001",
                "amount": 400,
                "currency": "CNY",
                "reason": "damaged",
            }
        )

    with pytest.raises(RefundAmountError, match="currency"):
        await eligible_tool.execute(
            {
                "order_id": "ORD-2001",
                "amount": 10,
                "currency": "USD",
                "reason": "damaged",
            }
        )


@pytest.mark.asyncio
async def test_after_sales_tool_creates_a_return_request():
    tool = AfterSalesTool(
        InMemoryOrderRepository({"ORD-2001": make_order()}),
        InMemoryAfterSalesRepository(),
    )

    result = await tool.execute(
        {
            "order_id": "ORD-2001",
            "action": "return",
            "reason": "wrong size",
        }
    )

    assert tool.name == "after_sales.create"
    assert result.request_id.startswith("after-sales-")
    assert result.action == "return"
    assert result.status == "pending_review"


@pytest.mark.asyncio
async def test_after_sales_tool_rejects_cancelled_orders():
    tool = AfterSalesTool(
        InMemoryOrderRepository({"ORD-2001": make_order(status="cancelled")}),
        InMemoryAfterSalesRepository(),
    )

    with pytest.raises(RefundNotEligibleError):
        await tool.execute(
            {"order_id": "ORD-2001", "action": "repair", "reason": "broken"}
        )


def test_request_inputs_reject_unknown_fields_and_invalid_amounts():
    with pytest.raises(ValidationError):
        RefundRequestInput(
            order_id="ORD-2001",
            amount=0,
            currency="CNY",
            reason="damaged",
        )

    with pytest.raises(ValidationError):
        AfterSalesRequestInput(
            order_id="ORD-2001",
            action="refund",
            reason="not a supported action",
        )
