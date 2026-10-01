from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.after_sales import AfterSalesExtractor
from app.contracts import ChatMessage
from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class AfterSalesIntentNodeError(RuntimeError):
    code = "after_sales_intent_failed"


@dataclass(slots=True)
class AfterSalesIntentNode:
    extractor: AfterSalesExtractor
    name: str = field(default="after-sales-intent", init=False)

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        state.status = WorkflowStatus.RUNNING
        state.current_node = self.name
        if state.data.get("intent") not in {"refund_return", "after_sales"}:
            state.data["after_sales_intent_skipped"] = True
            state.data["workflow_run_id"] = context.run_id
            return state

        state.data.pop("after_sales_intent_skipped", None)
        try:
            messages = _coerce_messages(state.data.get("messages"))
            extraction = await self.extractor.extract(messages)
        except AfterSalesIntentNodeError:
            state.status = WorkflowStatus.FAILED
            raise
        except Exception as exc:
            error = AfterSalesIntentNodeError("after-sales intent recognition failed")
            state.status = WorkflowStatus.FAILED
            raise error from exc

        state.data["after_sales_intent"] = extraction.data.intent
        state.data["after_sales_info"] = extraction.data.model_dump(mode="json")
        state.data["after_sales_extraction_request_id"] = extraction.request_id
        state.data["workflow_run_id"] = context.run_id
        return state


def _coerce_messages(raw_messages: object) -> list[ChatMessage]:
    if raw_messages is None:
        raise AfterSalesIntentNodeError("workflow requires at least one message")
    if isinstance(raw_messages, (str, bytes)) or not isinstance(raw_messages, Sequence):
        raise AfterSalesIntentNodeError("workflow messages must be a sequence")
    if not raw_messages:
        raise AfterSalesIntentNodeError("workflow requires at least one message")

    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        try:
            messages.append(
                raw_message
                if isinstance(raw_message, ChatMessage)
                else ChatMessage.model_validate(raw_message)
            )
        except (ValidationError, TypeError) as exc:
            raise AfterSalesIntentNodeError(
                "workflow messages contain an invalid message"
            ) from exc
    return messages

