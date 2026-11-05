from datetime import datetime, timezone

import pytest

from app.default_tools import default_tool_registry
from app.providers import StubModelProvider
from app.workflow_runtime import default_workflow_runtime


def test_default_tool_registry_exposes_seeded_business_tools():
    registry = default_tool_registry()

    assert [definition.name for definition in registry.definitions()] == [
        "after_sales.create",
        "faq.search",
        "logistics.query",
        "order.query",
        "refund.create",
    ]


@pytest.mark.asyncio
async def test_default_tools_can_query_demo_order_and_shipping_data():
    registry = default_tool_registry()

    order = await registry.execute("order.query", {"order_id": "ORD-DEMO-1001"})
    shipment = await registry.execute(
        "logistics.query", {"order_id": "ORD-DEMO-1001"}
    )
    faq = await registry.execute(
        "faq.search", {"query": "退款多久到账", "category": "refund"}
    )

    assert order.order_id == "ORD-DEMO-1001"
    assert shipment.order_id == order.order_id
    assert faq.entries


def test_default_workflow_runtime_uses_the_default_business_registry():
    runtime = default_workflow_runtime(StubModelProvider())

    assert {definition.name for definition in runtime.tool_registry.definitions()} == {
        "after_sales.create",
        "faq.search",
        "logistics.query",
        "order.query",
        "refund.create",
    }
