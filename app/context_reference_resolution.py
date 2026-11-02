import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.contracts import ChatMessage
from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class ContextReferenceResolutionError(RuntimeError):
    code = "context_reference_resolution_failed"


_REFERENCE_PATTERN = re.compile(
    r"这个订单|该订单|此订单|这笔订单|那笔订单|刚才的订单|上一笔订单|"
    r"同一笔订单|同一个订单|这件商品|那个商品|它|这个|那个|"
    r"the same order|this order|that order|the order|this item|that item|"
    r"\bit\b|\bthis\b|\bthat\b",
    re.IGNORECASE,
)
_ORDER_ID_PATTERN = re.compile(
    r"\b(?:ORD|ORDER)[-_][A-Z0-9][A-Z0-9_-]{0,62}\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ContextReferenceResolutionNode:
    name: str = field(default="context-reference-resolution", init=False)

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        state.status = WorkflowStatus.RUNNING
        state.current_node = self.name
        state.data.pop("resolved_order_context", None)
        state.data.pop("unresolved_order_reference", None)
        state.data.pop("ambiguous_order_ids", None)

        if state.data.get("intent") not in {"refund_return", "after_sales"}:
            state.data["context_reference_resolution"] = "skipped"
            state.data["workflow_run_id"] = context.run_id
            return state

        messages = _coerce_messages(state.data.get("messages"))
        latest_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].role == "user"
            ),
            None,
        )
        if latest_user_index is None:
            raise ContextReferenceResolutionError("workflow requires a user message")

        latest_user_message = messages[latest_user_index]
        reference_match = _REFERENCE_PATTERN.search(latest_user_message.content)
        if reference_match is None:
            state.data["context_reference_resolution"] = "not_needed"
            state.data["workflow_run_id"] = context.run_id
            return state

        if _ORDER_ID_PATTERN.search(latest_user_message.content):
            state.data["context_reference_resolution"] = "explicit_order_id"
            state.data["workflow_run_id"] = context.run_id
            return state

        for message_index in range(latest_user_index - 1, -1, -1):
            message = messages[message_index]
            if message.role not in {"user", "assistant", "tool"}:
                continue
            matches = list(_ORDER_ID_PATTERN.finditer(message.content))
            if not matches:
                continue
            order_ids = list(dict.fromkeys(match.group(0) for match in matches))
            if len(order_ids) > 1:
                state.data["unresolved_order_reference"] = reference_match.group(0)
                state.data["ambiguous_order_ids"] = order_ids
                state.data["context_reference_resolution"] = "ambiguous_prior_order_ids"
                state.data["workflow_run_id"] = context.run_id
                return state
            state.data["resolved_order_context"] = {
                "order_id": order_ids[0],
                "reference": reference_match.group(0),
                "source_message_index": message_index,
                "source_role": message.role,
            }
            state.data["context_reference_resolution"] = "resolved"
            state.data["workflow_run_id"] = context.run_id
            return state

        state.data["unresolved_order_reference"] = reference_match.group(0)
        state.data["context_reference_resolution"] = "no_prior_order_id"
        state.data["workflow_run_id"] = context.run_id
        return state


def _coerce_messages(raw_messages: object) -> list[ChatMessage]:
    if isinstance(raw_messages, (str, bytes)) or not isinstance(
        raw_messages, Sequence
    ):
        raise ContextReferenceResolutionError("workflow messages must be a sequence")
    if not raw_messages:
        raise ContextReferenceResolutionError("workflow requires at least one message")

    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        try:
            messages.append(
                raw_message
                if isinstance(raw_message, ChatMessage)
                else ChatMessage.model_validate(raw_message)
            )
        except (ValidationError, TypeError) as exc:
            raise ContextReferenceResolutionError(
                "workflow messages contain an invalid message"
            ) from exc
    return messages
