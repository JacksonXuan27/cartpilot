from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import ChatMessage, TokenUsage
from app.context_layers import ContextLayerManager
from app.intent_routing import IntentRouterNode
from app.providers import ModelResult, StubModelProvider, ToolCall
from app.react_loop import ReactLoopNode
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowStatus
from app.workflow_runtime import WorkflowRuntime, WorkflowRuntimeError


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class EchoTool:
    name = "test.echo"
    description = "Echo a value for workflow runtime tests."
    input_model = EchoInput

    async def execute(self, arguments: Mapping[str, object]) -> dict[str, object]:
        return {"echo": arguments["value"]}


def make_tool_call(call_id: str) -> ToolCall:
    return ToolCall(
        name="test.echo",
        arguments={"value": "hello"},
        call_id=call_id,
    )


@pytest.mark.anyio
async def test_runtime_runs_router_and_react_nodes():
    runtime = WorkflowRuntime(StubModelProvider(reply="回复内容"), ToolRegistry())
    state = await runtime.run([ChatMessage(role="user", content="你好")], session_id="session-1")

    assert state.status is WorkflowStatus.COMPLETED
    assert state.data["intent"] == "smalltalk"
    assert state.data["route"] == "fallback_script"
    assert state.data["answer"] == "回复内容"
    assert state.data["session_id"] == "session-1"
    assert state.data["react_iterations"] == 1
    assert state.data["confirmation_required"] is False


@pytest.mark.anyio
async def test_runtime_exposes_token_usage_and_budget():
    provider = StubModelProvider(
        reply="回复内容",
        usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    runtime = WorkflowRuntime(provider, ToolRegistry(), token_budget=5)

    state = await runtime.run([ChatMessage(role="user", content="你好")])

    assert state.status is WorkflowStatus.COMPLETED
    assert state.data["token_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert state.data["token_budget"] == 5


@pytest.mark.anyio
async def test_runtime_uses_context_layers_for_model_prompt_and_preserves_history():
    provider = StubModelProvider(reply="acknowledged")
    context_layers = ContextLayerManager(short_term_limit=2, long_term_limit=1)
    runtime = WorkflowRuntime(
        provider,
        ToolRegistry(),
        nodes=(ReactLoopNode(provider, ToolRegistry()),),
        context_layer_manager=context_layers,
    )

    messages = [
        ChatMessage(role="system", content="system-rule"),
        ChatMessage(role="user", content="message-1"),
        ChatMessage(role="assistant", content="message-2"),
        ChatMessage(role="user", content="message-3"),
        ChatMessage(role="assistant", content="message-4"),
        ChatMessage(role="user", content="message-5"),
    ]
    state = await runtime.run(messages)

    prompt = provider.calls[0]
    assert prompt[0].content == "system-rule"
    assert prompt[1].role == "assistant"
    assert "message-1" in prompt[1].content
    assert [message.content for message in prompt[2:]] == [
        "message-3",
        "message-4",
        "message-5",
    ]
    assert len(state.data["messages"]) == len(messages) + 1
    assert [message.content for message in state.data["context_layers"]["short_term"]] == [
        "message-4",
        "message-5",
    ]
    assert state.data["context_layers"]["omitted_message_count"] == 2
    assert state.data["context_layers"]["summary"].role == "assistant"
    assert "message-1" in state.data["context_layers"]["summary"].content


@pytest.mark.anyio
async def test_runtime_reuses_context_cache_between_after_sales_and_agent():
    provider = StubModelProvider(
        reply=(
            '{"intent":"repair","order_id":"ORD-CACHE",'
            '"reason":"broken","requested_action":"repair",'
            '"confidence":0.9}'
        )
    )
    context_layers = ContextLayerManager()
    runtime = WorkflowRuntime(
        provider,
        ToolRegistry(),
        context_layer_manager=context_layers,
    )

    state = await runtime.run(
        [ChatMessage(role="user", content="订单 ORD-CACHE 坏了需要维修")]
    )

    assert state.status is WorkflowStatus.COMPLETED
    assert len(provider.calls) == 2
    assert context_layers.cache.stats().misses == 1
    assert context_layers.cache.stats().hits == 1


@pytest.mark.anyio
async def test_runtime_completes_normal_tool_call_path():
    provider = StubModelProvider(
        reply="工具查询完成",
        tool_call_rounds=[(make_tool_call("call-1"),)],
    )
    runtime = WorkflowRuntime(provider, ToolRegistry([EchoTool()]))

    state = await runtime.run([ChatMessage(role="user", content="查订单")])

    assert state.status is WorkflowStatus.COMPLETED
    assert state.current_node == "react-loop"
    assert state.data["intent"] == "order"
    assert state.data["route"] == "business"
    assert state.data["react_iterations"] == 2
    assert state.data["tool_history"][0]["result"] == {"echo": "hello"}


@pytest.mark.anyio
async def test_runtime_shares_budget_between_after_sales_and_resume():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-TOKEN",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        ),
        usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    runtime = WorkflowRuntime(provider, ToolRegistry(), token_budget=10)

    paused = await runtime.run(
        [ChatMessage(role="user", content="ORD-TOKEN 到货破损，我要退款")]
    )
    failed = await runtime.resume(
        str(paused.data["confirmation_id"]),
        confirmed=True,
    )

    assert paused.status is WorkflowStatus.AWAITING_CONFIRMATION
    assert paused.data["token_usage"]["total_tokens"] == 6
    assert failed.status is WorkflowStatus.FAILED
    assert failed.data["error"]["code"] == "token_budget_exceeded"
    assert failed.data["token_usage"]["total_tokens"] == 12
    assert len(provider.calls) == 2


@pytest.mark.anyio
async def test_runtime_returns_structured_failure_when_after_sales_exceeds_budget():
    provider = StubModelProvider(
        reply=(
            '{"intent":"refund","order_id":"ORD-BUDGET",'
            '"reason":"damaged","requested_action":"refund",'
            '"confidence":0.9}'
        ),
        usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    runtime = WorkflowRuntime(provider, ToolRegistry(), token_budget=5)

    state = await runtime.run(
        [ChatMessage(role="user", content="ORD-BUDGET 到货破损，我要退款")]
    )

    assert state.status is WorkflowStatus.FAILED
    assert state.current_node == "after-sales-intent"
    assert state.data["error"]["code"] == "token_budget_exceeded"
    assert state.data["token_usage"]["total_tokens"] == 6
    assert state.data["token_budget"] == 5
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_runtime_converts_provider_failure_to_failed_state():
    class BrokenProvider:
        async def complete(self, messages, tools=()) -> ModelResult:
            raise RuntimeError("provider is unavailable")

    runtime = WorkflowRuntime(BrokenProvider(), ToolRegistry())

    state = await runtime.run([ChatMessage(role="user", content="查订单")])

    assert state.status is WorkflowStatus.FAILED
    assert state.current_node == "react-loop"
    assert state.data["error"]["code"] == "react_loop_failed"
    assert state.data["error"]["message"] == "react loop failed: provider is unavailable"


@pytest.mark.anyio
async def test_runtime_converts_react_iteration_limit_to_failed_state():
    provider = StubModelProvider(
        tool_call_rounds=[(make_tool_call("call-1"),), (make_tool_call("call-2"),)]
    )
    registry = ToolRegistry([EchoTool()])
    runtime = WorkflowRuntime(
        provider,
        registry,
        nodes=(IntentRouterNode(), ReactLoopNode(provider, registry, max_iterations=2)),
    )

    state = await runtime.run([ChatMessage(role="user", content="查订单")])

    assert state.status is WorkflowStatus.FAILED
    assert state.data["error"]["code"] == "react_iteration_limit"
    assert state.data["error"]["iteration"] == 2
    assert state.data["react_iterations"] == 2
    assert len(provider.calls) == 2


def test_runtime_rejects_empty_or_duplicate_node_configuration():
    with pytest.raises(WorkflowRuntimeError, match="at least one node"):
        WorkflowRuntime(StubModelProvider(), ToolRegistry(), nodes=())

    class Node:
        name = "same"

    with pytest.raises(WorkflowRuntimeError, match="duplicate"):
        WorkflowRuntime(StubModelProvider(), ToolRegistry(), nodes=(Node(), Node()))


@pytest.mark.anyio
async def test_runtime_rejects_empty_messages():
    runtime = WorkflowRuntime(StubModelProvider(), ToolRegistry())
    with pytest.raises(WorkflowRuntimeError, match="at least one message"):
        await runtime.run([])
