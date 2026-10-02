import pytest

from app.after_sales import AfterSalesExtractor
from app.after_sales_intent import AfterSalesIntentNode, AfterSalesIntentNodeError
from app.after_sales_routing import AfterSalesRoutingError, AfterSalesRoutingNode
from app.contracts import ChatMessage
from app.context_reference_resolution import ContextReferenceResolutionNode
from app.refund_interruption import RefundInterruptionNode
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
async def test_context_reference_resolution_uses_latest_prior_order_id():
    state = WorkflowState(
        data={
            "intent": "refund_return",
            "messages": [
                ChatMessage(role="user", content="查询订单 ORD-1001"),
                ChatMessage(role="assistant", content="查到了 ORD-1001"),
                ChatMessage(role="user", content="另一个订单 ORD-2002 有物流吗"),
                ChatMessage(role="assistant", content="ORD-2002 正在配送"),
                ChatMessage(role="user", content="这个订单我想退款"),
            ],
        }
    )

    await ContextReferenceResolutionNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-context-1", run_id="run-context-1"),
    )

    assert state.data["context_reference_resolution"] == "resolved"
    assert state.data["resolved_order_context"] == {
        "order_id": "ORD-2002",
        "reference": "这个订单",
        "source_message_index": 3,
        "source_role": "assistant",
    }
    assert state.data["messages"][-1].content == "这个订单我想退款"


@pytest.mark.asyncio
async def test_context_reference_resolution_does_not_guess_without_history():
    state = WorkflowState(
        data={
            "intent": "after_sales",
            "messages": [ChatMessage(role="user", content="它有质量问题")],
        }
    )

    await ContextReferenceResolutionNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-context-2"),
    )

    assert state.data["context_reference_resolution"] == "no_prior_order_id"
    assert state.data["unresolved_order_reference"] == "它"
    assert "resolved_order_context" not in state.data


@pytest.mark.asyncio
async def test_context_reference_resolution_preserves_explicit_current_order_id():
    state = WorkflowState(
        data={
            "intent": "refund_return",
            "messages": [
                ChatMessage(role="user", content="订单 ORD-1001 有问题"),
                ChatMessage(role="user", content="这个订单号是 ORD-2002，我要退款"),
            ],
        }
    )

    await ContextReferenceResolutionNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-context-3"),
    )

    assert state.data["context_reference_resolution"] == "explicit_order_id"
    assert "resolved_order_context" not in state.data


@pytest.mark.asyncio
async def test_context_reference_resolution_does_not_guess_between_prior_orders():
    state = WorkflowState(
        data={
            "intent": "refund_return",
            "messages": [
                ChatMessage(
                    role="assistant",
                    content="订单 ORD-1001 和 ORD-2002 都符合条件。",
                ),
                ChatMessage(role="user", content="这个订单我要退款"),
            ],
        }
    )

    await ContextReferenceResolutionNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-context-4"),
    )

    assert state.data["context_reference_resolution"] == "ambiguous_prior_order_ids"
    assert state.data["ambiguous_order_ids"] == ["ORD-1001", "ORD-2002"]
    assert "resolved_order_context" not in state.data


@pytest.mark.asyncio
async def test_workflow_fills_missing_extracted_order_id_from_resolved_context():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":null,"reason":"damaged",'
            '"requested_action":"refund","confidence":0.9}'
        )
    )
    runtime = WorkflowRuntime(provider, ToolRegistry())

    state = await runtime.run(
        [
            ChatMessage(role="user", content="订单 ORD-3003 到货了"),
            ChatMessage(role="assistant", content="收到，订单 ORD-3003"),
            ChatMessage(role="user", content="这个订单坏了想退款"),
        ]
    )

    assert state.status is WorkflowStatus.AWAITING_CONFIRMATION
    assert state.data["resolved_order_context"]["order_id"] == "ORD-3003"
    assert state.data["after_sales_info"]["order_id"] == "ORD-3003"
    assert state.data["after_sales_order_id_source"] == "conversation_context"
    assert state.data["confirmation_details"]["order_id"] == "ORD-3003"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("after_sales_intent", "expected_route", "requires_confirmation"),
    [
        ("refund", "refund_flow", True),
        ("return", "return_flow", True),
        ("exchange", "exchange_flow", True),
        ("repair", "repair_flow", False),
        ("logistics_issue", "logistics_flow", False),
    ],
)
async def test_after_sales_routing_selects_scenario_route(
    after_sales_intent: str,
    expected_route: str,
    requires_confirmation: bool,
):
    state = WorkflowState(
        data={
            "intent": "after_sales",
            "after_sales_intent": after_sales_intent,
        }
    )

    result = await AfterSalesRoutingNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-route-1", run_id="run-route-1"),
    )

    assert result.status is WorkflowStatus.RUNNING
    assert result.current_node == "after-sales-routing"
    assert result.data["after_sales_scenario"] == after_sales_intent
    assert result.data["after_sales_route"] == expected_route
    assert result.data["route"] == expected_route
    assert result.data["after_sales_requires_confirmation"] is requires_confirmation
    assert result.data["after_sales_routing"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("after_sales_intent", ["other", "unknown", "future_intent"])
async def test_after_sales_routing_sends_uncertain_intents_to_manual_review(
    after_sales_intent: str,
):
    state = WorkflowState(
        data={
            "intent": "refund_return",
            "after_sales_intent": after_sales_intent,
        }
    )

    await AfterSalesRoutingNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-route-2"),
    )

    assert state.data["after_sales_route"] == "manual_review"
    assert state.data["after_sales_requires_confirmation"] is True
    assert state.data["after_sales_handoff_reason"] == "unrecognized_after_sales_intent"


@pytest.mark.asyncio
async def test_after_sales_routing_requires_extracted_intent():
    state = WorkflowState(data={"intent": "after_sales"})

    with pytest.raises(AfterSalesRoutingError, match="intent is required"):
        await AfterSalesRoutingNode().execute(
            state,
            WorkflowRuntimeContext(request_id="request-route-3"),
        )

    assert state.status is WorkflowStatus.FAILED


@pytest.mark.asyncio
async def test_workflow_runs_after_sales_routing_before_reply():
    provider = StubModelProvider(
        reply=(
            '{"intent":"repair","order_id":"ORD-4004",'
            '"reason":"broken","requested_action":"repair",'
            '"confidence":0.86}'
        )
    )
    runtime = WorkflowRuntime(provider, ToolRegistry())

    state = await runtime.run(
        [ChatMessage(role="user", content="订单 ORD-4004 坏了需要维修")]
    )

    assert state.status is WorkflowStatus.COMPLETED
    assert state.data["after_sales_route"] == "repair_flow"
    assert state.data["route"] == "repair_flow"
    assert state.data["after_sales_requires_confirmation"] is False


@pytest.mark.asyncio
async def test_refund_interruption_pauses_before_agent_tools_run():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-5005",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.93}'
        ),
        tool_call_rounds=[],
    )
    runtime = WorkflowRuntime(provider, ToolRegistry())

    state = await runtime.run(
        [ChatMessage(role="user", content="订单 ORD-5005 到货破损，我要退款")]
    )

    assert state.status is WorkflowStatus.AWAITING_CONFIRMATION
    assert state.current_node == "refund-confirmation-interrupt"
    assert state.data["confirmation_required"] is True
    assert state.data["confirmation_status"] == "pending"
    assert state.data["confirmation_details"] == {
        "scenario": "refund",
        "order_id": "ORD-5005",
        "reason": "damaged",
        "requested_action": "refund",
    }
    assert len(provider.calls) == 1
    assert state.data["answer"] == "请确认是否继续此售后申请。"


@pytest.mark.asyncio
async def test_refund_interruption_node_leaves_non_confirming_flow_running():
    state = WorkflowState(
        data={"after_sales_requires_confirmation": False}
    )

    await RefundInterruptionNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-refund-interrupt-1"),
    )

    assert state.status is WorkflowStatus.RUNNING
    assert state.data["confirmation_required"] is False
    assert "confirmation_id" not in state.data


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
async def test_workflow_runtime_pauses_return_for_confirmation_before_reply():
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

    assert state.status is WorkflowStatus.AWAITING_CONFIRMATION
    assert state.data["after_sales_intent"] == "return"
    assert state.data["after_sales_route"] == "return_flow"
    assert state.data["confirmation_required"] is True
    assert len(provider.calls) == 1
