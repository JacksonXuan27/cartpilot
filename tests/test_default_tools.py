from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.default_tools import default_tool_registry
from app.data_models import OrderRecord
from app.database import DatabaseManager
from app.repositories import OrderRepository, initialize_schema
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


@pytest.mark.asyncio
async def test_default_order_tool_reads_updates_from_sqlite_across_registry_rebuilds(
    tmp_path,
):
    database = DatabaseManager(f"sqlite:///{tmp_path / 'orders.db'}")
    database.open()
    initialize_schema(database)
    repository = OrderRepository(database)
    first_registry = default_tool_registry(database=database)

    order = await first_registry.execute(
        "order.query", {"order_id": "ORD-DEMO-1001"}
    )
    updated = OrderRecord(
        order_id=order.order_id,
        status="delivered",
        product_name=order.product_name,
        quantity=order.quantity,
        total_amount=order.total_amount,
        currency=order.currency,
        placed_at=order.placed_at,
        created_at=order.created_at,
        updated_at=datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc),
    )
    await repository.save(updated)

    second_registry = default_tool_registry(database=database)
    persisted = await second_registry.execute(
        "order.query", {"order_id": "ORD-DEMO-1001"}
    )

    assert persisted.status == "delivered"
    assert persisted.updated_at == updated.updated_at
    database.close()


def test_default_workflow_runtime_uses_the_default_business_registry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "app.workflow_runtime.get_settings",
        lambda: SimpleNamespace(
            database_url=f"sqlite:///{tmp_path / 'registry.db'}",
            database_timeout_seconds=5.0,
            token_budget=2000,
            chat_model="configured-test-model",
        ),
    )
    runtime = default_workflow_runtime(StubModelProvider())

    assert {definition.name for definition in runtime.tool_registry.definitions()} == {
        "after_sales.create",
        "faq.search",
        "logistics.query",
        "order.query",
        "refund.create",
    }
    runtime.checkpoint_store.close()


@pytest.mark.asyncio
async def test_default_workflow_runtime_uses_configured_sqlite_for_orders(
    tmp_path, monkeypatch
):
    database_url = f"sqlite:///{tmp_path / 'runtime-orders.db'}"

    monkeypatch.setattr(
        "app.workflow_runtime.get_settings",
        lambda: SimpleNamespace(
            database_url=database_url,
            database_timeout_seconds=5.0,
            token_budget=2000,
            chat_model="configured-test-model",
        ),
    )
    first_runtime = default_workflow_runtime(StubModelProvider())
    first_order = await first_runtime.tool_registry.execute(
        "order.query", {"order_id": "ORD-DEMO-1001"}
    )

    database = DatabaseManager(database_url)
    database.open()
    repository = OrderRepository(database)
    updated = OrderRecord(
        order_id=first_order.order_id,
        status="delivered",
        product_name=first_order.product_name,
        quantity=first_order.quantity,
        total_amount=first_order.total_amount,
        currency=first_order.currency,
        placed_at=first_order.placed_at,
        created_at=first_order.created_at,
        updated_at=datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc),
    )
    await repository.save(updated)
    database.close()
    first_runtime.checkpoint_store.close()

    second_runtime = default_workflow_runtime(StubModelProvider())
    persisted = await second_runtime.tool_registry.execute(
        "order.query", {"order_id": "ORD-DEMO-1001"}
    )

    assert persisted.status == "delivered"
    assert persisted.updated_at == updated.updated_at
    second_runtime.checkpoint_store.close()
