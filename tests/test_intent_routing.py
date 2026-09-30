import pytest

from app.contracts import ChatMessage
from app.intent_routing import (
    IntentCategory,
    IntentRoutingError,
    IntentRouterNode,
    route_for_intent,
)
from app.workflow import WorkflowNode, WorkflowRuntimeContext, WorkflowState, WorkflowStatus


@pytest.mark.asyncio
async def test_router_classifies_logistics_and_selects_business_route():
    node = IntentRouterNode()
    state = WorkflowState(data={"query": "我的快递现在到哪了"})

    result = await node.execute(
        state,
        WorkflowRuntimeContext(request_id="request-1"),
    )

    assert isinstance(node, WorkflowNode)
    assert result.status is WorkflowStatus.RUNNING
    assert result.current_node == "intent-router"
    assert result.data["intent"] == IntentCategory.LOGISTICS.value
    assert result.data["route"] == "business"
    assert result.data["intent_confidence"] > 0
    assert "快递" in result.data["matched_intent_terms"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "intent", "route"),
    [
        ("这个商品还能退款吗", IntentCategory.REFUND_RETURN, "refund_flow"),
        ("收到的商品坏了怎么维修", IntentCategory.AFTER_SALES, "refund_flow"),
        ("这个产品是什么材质", IntentCategory.PRODUCT, "knowledge"),
        ("我要转人工客服", IntentCategory.HUMAN, "business"),
        ("我要投诉这个服务", IntentCategory.COMPLAINT, "escalate"),
    ],
)
async def test_router_maps_business_intents_to_expected_routes(query, intent, route):
    state = WorkflowState(data={"query": query})

    result = await IntentRouterNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-2"),
    )

    assert result.data["intent"] == intent.value
    assert result.data["route"] == route


@pytest.mark.asyncio
async def test_router_uses_latest_user_message_when_query_is_missing():
    state = WorkflowState(
        data={
            "messages": [
                ChatMessage(role="user", content="你好"),
                ChatMessage(role="assistant", content="你好，请问有什么可以帮您？"),
                ChatMessage(role="user", content="订单现在是什么状态"),
            ]
        }
    )

    result = await IntentRouterNode().execute(
        state,
        WorkflowRuntimeContext(request_id="request-3"),
    )

    assert result.data["query"] == "订单现在是什么状态"
    assert result.data["intent"] == IntentCategory.ORDER.value


@pytest.mark.asyncio
async def test_router_sends_unknown_and_ambiguous_queries_to_fallback():
    node = IntentRouterNode()
    unknown = await node.execute(
        WorkflowState(data={"query": "今天天气怎么样"}),
        WorkflowRuntimeContext(request_id="request-4"),
    )
    ambiguous = await node.execute(
        WorkflowState(data={"query": "订单物流信息"}),
        WorkflowRuntimeContext(request_id="request-5"),
    )

    assert unknown.data["intent"] == IntentCategory.OTHER.value
    assert unknown.data["route"] == "fallback_script"
    assert unknown.data["intent_confidence"] == 0.0
    assert ambiguous.data["intent"] == IntentCategory.LOGISTICS.value
    assert ambiguous.data["route"] == "business"


@pytest.mark.parametrize(
    "state",
    [
        WorkflowState(),
        WorkflowState(data={"query": " "}),
        WorkflowState(data={"messages": []}),
        WorkflowState(data={"messages": [ChatMessage(role="assistant", content="hello")]}),
    ],
)
@pytest.mark.asyncio
async def test_router_rejects_state_without_user_query(state):
    with pytest.raises(IntentRoutingError):
        await IntentRouterNode().execute(
            state,
            WorkflowRuntimeContext(request_id="request-6"),
        )


@pytest.mark.parametrize(
    ("intent", "route"),
    [
        (IntentCategory.LOGISTICS, "business"),
        (IntentCategory.PRODUCT, "knowledge"),
        (IntentCategory.OTHER, "fallback_script"),
    ],
)
def test_route_for_intent_is_total_for_supported_categories(intent, route):
    assert route_for_intent(intent) == route
