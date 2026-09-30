import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.contracts import ChatMessage
from app.providers import ToolCall
from app.tool_registry import ToolRegistry
from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class ToolCallNodeError(RuntimeError):
    def __init__(self, message: str, tool_name: str | None = None) -> None:
        self.tool_name = tool_name
        super().__init__(message)


@dataclass(slots=True)
class ToolCallNode:
    tool_registry: ToolRegistry
    name: str = field(default="tool-call", init=False)

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        tool_call: ToolCall | None = None
        try:
            tool_call = _coerce_tool_call(state.data.get("tool_call"))
            output = await self.tool_registry.execute(
                tool_call.name,
                dict(tool_call.arguments),
            )
            normalized_output, serialized_output = _serialize_output(output)
            _append_tool_message(state, tool_call.call_id, serialized_output)
        except ToolCallNodeError as exc:
            _mark_failed(state, self.name, tool_call, exc)
            raise
        except Exception as exc:
            tool_name = tool_call.name if tool_call is not None else None
            error = ToolCallNodeError(
                f"tool call failed: {tool_name or 'unknown'}",
                tool_name=tool_name,
            )
            _mark_failed(state, self.name, tool_call, error)
            raise error from exc

        state.status = WorkflowStatus.RUNNING
        state.current_node = self.name
        state.data["tool_name"] = tool_call.name
        state.data["tool_call_id"] = tool_call.call_id
        state.data["tool_arguments"] = dict(tool_call.arguments)
        state.data["tool_result"] = normalized_output
        state.data["tool_result_json"] = serialized_output
        state.data["workflow_run_id"] = context.run_id
        return state


def _coerce_tool_call(raw_call: object) -> ToolCall:
    if isinstance(raw_call, ToolCall):
        _validate_tool_call_fields(raw_call.name, raw_call.call_id, raw_call.arguments)
        return ToolCall(
            name=raw_call.name.strip(),
            arguments=dict(raw_call.arguments),
            call_id=raw_call.call_id.strip(),
        )
    if not isinstance(raw_call, Mapping):
        raise ToolCallNodeError("workflow state must contain a tool_call mapping")

    name = raw_call.get("name")
    arguments = raw_call.get("arguments")
    call_id = raw_call.get("call_id")
    _validate_tool_call_fields(name, call_id, arguments)
    return ToolCall(
        name=name.strip(),
        arguments=dict(arguments),
        call_id=call_id.strip(),
    )


def _validate_tool_call_fields(
    name: object,
    call_id: object,
    arguments: object,
) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ToolCallNodeError("tool call name cannot be empty")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ToolCallNodeError("tool call ID cannot be empty", tool_name=name)
    if not isinstance(arguments, Mapping):
        raise ToolCallNodeError(
            "tool call arguments must be a mapping", tool_name=name
        )


def _serialize_output(output: Any) -> tuple[object, str]:
    payload = output.model_dump(mode="json") if isinstance(output, BaseModel) else output
    try:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return json.loads(serialized), serialized
    except (TypeError, ValueError) as exc:
        raise ToolCallNodeError("tool result is not JSON serializable") from exc


def _append_tool_message(
    state: WorkflowState,
    call_id: str,
    serialized_output: str,
) -> None:
    raw_messages = state.data.get("messages")
    if raw_messages is None:
        return
    if isinstance(raw_messages, (str, bytes)) or not isinstance(raw_messages, Sequence):
        raise ToolCallNodeError("workflow messages must be a sequence")

    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        try:
            messages.append(
                raw_message
                if isinstance(raw_message, ChatMessage)
                else ChatMessage.model_validate(raw_message)
            )
        except (ValidationError, TypeError) as exc:
            raise ToolCallNodeError("workflow messages contain an invalid message") from exc
    messages.append(
        ChatMessage(role="tool", tool_call_id=call_id, content=serialized_output)
    )
    state.data["messages"] = messages


def _mark_failed(
    state: WorkflowState,
    node_name: str,
    tool_call: ToolCall | None,
    error: ToolCallNodeError,
) -> None:
    state.status = WorkflowStatus.FAILED
    state.current_node = node_name
    if tool_call is not None:
        state.data["tool_name"] = tool_call.name
        state.data["tool_call_id"] = tool_call.call_id
        state.data["tool_arguments"] = dict(tool_call.arguments)
    state.data["error"] = {
        "code": "tool_call_failed",
        "message": str(error),
        "tool_name": error.tool_name,
    }
