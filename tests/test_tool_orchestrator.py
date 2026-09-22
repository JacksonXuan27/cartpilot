from datetime import datetime, timezone

import pytest

from app.contracts import ChatMessage
from app.order_tools import InMemoryOrderRepository, Order, OrderQueryTool
from app.providers import StubModelProvider, ToolCall
from app.tool_orchestrator import ToolOrchestrationError, ToolOrchestrator
from app.tool_registry import ToolArgumentError, ToolRegistry


def make_order() -> Order:
    return Order(
        order_id="ORD-5001",
        status="shipped",
        product_name="Wireless mouse",
        quantity=1,
        total_amount=129.0,
        currency="CNY",
        placed_at=datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_orchestrator_executes_model_tool_call_and_returns_final_reply():
    provider = StubModelProvider(
        reply="Your order ORD-5001 has shipped.",
        tool_call_rounds=[
            (
                ToolCall(
                    name="order.query",
                    arguments={"order_id": "ORD-5001"},
                    call_id="call-order-1",
                ),
            )
        ],
    )
    registry = ToolRegistry(
        [OrderQueryTool(InMemoryOrderRepository({"ORD-5001": make_order()}))]
    )

    result = await ToolOrchestrator(provider, registry).complete(
        [ChatMessage(role="user", content="Where is order ORD-5001?")]
    )

    assert result.message.content == "Your order ORD-5001 has shipped."
    assert result.finish_reason == "stop"
    assert len(provider.calls) == 2
    assert provider.calls[1][-1].role == "tool"
    assert provider.calls[1][-1].tool_call_id == "call-order-1"
    assert '"order_id": "ORD-5001"' in provider.calls[1][-1].content
    assert provider.tool_definition_calls[0][0].name == "order.query"


@pytest.mark.asyncio
async def test_orchestrator_executes_multiple_tool_calls_in_one_round():
    provider = StubModelProvider(
        tool_call_rounds=[
            (
                ToolCall(
                    name="order.query",
                    arguments={"order_id": "ORD-5001"},
                    call_id="call-order-1",
                ),
                ToolCall(
                    name="order.query",
                    arguments={"order_id": "ORD-5001"},
                    call_id="call-order-2",
                ),
            )
        ]
    )
    registry = ToolRegistry(
        [OrderQueryTool(InMemoryOrderRepository({"ORD-5001": make_order()}))]
    )

    await ToolOrchestrator(provider, registry).complete(
        [ChatMessage(role="user", content="Check my order twice")]
    )

    assert [message.tool_call_id for message in provider.calls[1][-2:]] == [
        "call-order-1",
        "call-order-2",
    ]


@pytest.mark.asyncio
async def test_orchestrator_propagates_tool_argument_errors():
    provider = StubModelProvider(
        tool_call_rounds=[
            (
                ToolCall(
                    name="order.query",
                    arguments={"order_id": "bad id"},
                    call_id="call-invalid-1",
                ),
            )
        ]
    )
    registry = ToolRegistry([OrderQueryTool(InMemoryOrderRepository())])

    with pytest.raises(ToolArgumentError):
        await ToolOrchestrator(provider, registry).complete(
            [ChatMessage(role="user", content="Check this order")]
        )


@pytest.mark.asyncio
async def test_orchestrator_stops_runaway_tool_call_rounds():
    provider = StubModelProvider(
        tool_call_rounds=[
            (
                ToolCall(
                    name="order.query",
                    arguments={"order_id": "ORD-5001"},
                    call_id=f"call-order-{index}",
                ),
            )
            for index in range(3)
        ]
    )
    registry = ToolRegistry(
        [OrderQueryTool(InMemoryOrderRepository({"ORD-5001": make_order()}))]
    )

    with pytest.raises(ToolOrchestrationError, match="round limit"):
        await ToolOrchestrator(provider, registry, max_rounds=2).complete(
            [ChatMessage(role="user", content="Keep checking")]
        )


def test_tool_message_requires_a_call_id():
    with pytest.raises(ValueError):
        ChatMessage(role="tool", content="{}")
