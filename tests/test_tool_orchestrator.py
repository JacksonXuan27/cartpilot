import asyncio
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import ChatMessage
from app.order_tools import InMemoryOrderRepository, Order, OrderQueryTool
from app.providers import ModelProviderError, ModelResult, StubModelProvider, ToolCall
from app.tool_orchestrator import (
    ToolExecutionError,
    ToolOrchestrationError,
    ToolOrchestrator,
    ToolTimeoutError,
)
from app.tool_registry import (
    ToolArgumentError,
    ToolNotFoundError,
    ToolRegistry,
)


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


class FailureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class FailingTool:
    name = "test.fail"
    description = "A tool used to verify execution failures."
    input_model = FailureInput

    async def execute(self, arguments: dict[str, object]) -> None:
        raise RuntimeError("backend unavailable")


class SlowTool:
    name = "test.slow"
    description = "A tool used to verify execution timeouts."
    input_model = FailureInput

    async def execute(self, arguments: dict[str, object]) -> None:
        await asyncio.sleep(0.05)


class FailingProvider:
    async def complete(self, messages, tools=()) -> ModelResult:
        raise ModelProviderError("model request failed")

    def stream(self, messages):
        raise NotImplementedError


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
async def test_orchestrator_preserves_unknown_tool_errors():
    provider = StubModelProvider(
        tool_call_rounds=[
            (
                ToolCall(
                    name="unknown.tool",
                    arguments={},
                    call_id="call-unknown-1",
                ),
            )
        ]
    )

    with pytest.raises(ToolNotFoundError):
        await ToolOrchestrator(provider, ToolRegistry()).complete(
            [ChatMessage(role="user", content="Use an unknown tool")]
        )


@pytest.mark.asyncio
async def test_orchestrator_wraps_tool_runtime_errors_with_tool_context():
    provider = StubModelProvider(
        tool_call_rounds=[
            (
                ToolCall(
                    name="test.fail",
                    arguments={"value": "request"},
                    call_id="call-fail-1",
                ),
            )
        ]
    )

    with pytest.raises(ToolExecutionError) as error:
        await ToolOrchestrator(provider, ToolRegistry([FailingTool()])).complete(
            [ChatMessage(role="user", content="Call the failing tool")]
        )

    assert error.value.tool_name == "test.fail"
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_orchestrator_cancels_tools_that_exceed_the_timeout():
    provider = StubModelProvider(
        tool_call_rounds=[
            (
                ToolCall(
                    name="test.slow",
                    arguments={"value": "request"},
                    call_id="call-slow-1",
                ),
            )
        ]
    )

    with pytest.raises(ToolTimeoutError, match="test.slow"):
        await ToolOrchestrator(
            provider,
            ToolRegistry([SlowTool()]),
            tool_timeout_seconds=0.001,
        ).complete([ChatMessage(role="user", content="Call the slow tool")])


@pytest.mark.asyncio
async def test_orchestrator_propagates_model_call_failures():
    with pytest.raises(ModelProviderError, match="model request failed"):
        await ToolOrchestrator(FailingProvider(), ToolRegistry()).complete(
            [ChatMessage(role="user", content="Ask the model")]
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
