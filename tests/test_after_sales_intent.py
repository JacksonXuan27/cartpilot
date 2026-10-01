import pytest

from app.after_sales import AfterSalesExtractor
from app.after_sales_intent import AfterSalesIntentNode, AfterSalesIntentNodeError
from app.contracts import ChatMessage
from app.providers import StubModelProvider
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus
from app.workflow_runtime import WorkflowRuntime


def make_state(intent: str) -> WorkflowState:
    return WorkflowState(
        data={
            "intent": intent,
            "route": "refund_flow",
            "messages": [ChatMessage(role="user", content="商品坏了，想退款")],
        }
    )


@pytest.mark.asyncio
async def test_after_sales_intent_node_records_fine_grained_intent():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-1001",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.92}'
        )
    )
    state = make_state("refund_return")

    result = await AfterSalesIntentNode(AfterSalesExtractor(provider)).execute(
        state,
        WorkflowRuntimeContext(request_id="request-1", run_id="run-1"),
    )

    assert result.status is WorkflowStatus.RUNNING
    assert result.current_node == "after-sales-intent"
    assert result.data["after_sales_intent"] == "refund"
    assert result.data["after_sales_info"]["order_id"] == "ORD-1001"
    assert result.data["route"] == "refund_flow"
    assert result.data["workflow_run_id"] == "run-1"


@pytest.mark.asyncio
async def test_after_sales_intent_node_skips_unrelated_intents():
    provider = StubModelProvider()
    state = make_state("order")

    await AfterSalesIntentNode(AfterSalesExtractor(provider)).execute(
        state,
        WorkflowRuntimeContext(request_id="request-2"),
    )

    assert state.data["after_sales_intent_skipped"] is True
    assert "after_sales_intent" not in state.data
    assert provider.calls == []


@pytest.mark.asyncio
async def test_after_sales_intent_node_rejects_missing_messages():
    state = WorkflowState(data={"intent": "after_sales"})
    node = AfterSalesIntentNode(AfterSalesExtractor(StubModelProvider()))

    with pytest.raises(AfterSalesIntentNodeError, match="at least one message"):
        await node.execute(state, WorkflowRuntimeContext(request_id="request-3"))

    assert state.status is WorkflowStatus.FAILED


@pytest.mark.asyncio
async def test_workflow_runtime_surfaces_invalid_after_sales_classification():
    runtime = WorkflowRuntime(
        StubModelProvider(reply="not JSON"),
        ToolRegistry(),
    )

    state = await runtime.run(
        [ChatMessage(role="user", content="商品坏了，想退款")]
    )

    assert state.status is WorkflowStatus.FAILED
    assert state.current_node == "after-sales-intent"
    assert state.data["error"]["code"] == "after_sales_intent_failed"


@pytest.mark.asyncio
async def test_workflow_runtime_runs_after_sales_classification_before_reply():
    provider = StubModelProvider(
        reply=(
            '{"intent":"return","order_id":null,"reason":"wrong size",'
            '"requested_action":"return","confidence":0.81}'
        )
    )
    runtime = WorkflowRuntime(provider, ToolRegistry())

    state = await runtime.run(
        [ChatMessage(role="user", content="尺码不合适，想退货")]
    )

    assert state.status is WorkflowStatus.COMPLETED
    assert state.data["after_sales_intent"] == "return"
    assert len(provider.calls) == 2
    assert state.data["answer"] == provider.reply
