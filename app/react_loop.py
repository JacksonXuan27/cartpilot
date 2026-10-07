from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import time

from pydantic import ValidationError

from app.contracts import ChatMessage
from app.providers import ChatModelProvider, ModelResult
from app.tool_calling import ToolCallNode, ToolCallNodeError
from app.tool_registry import ToolRegistry
from app.workflow import (
    WorkflowRuntimeContext,
    WorkflowState,
    WorkflowStatus,
    WorkflowTokenBudgetExceededError,
)


class ReactLoopError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "react_loop_failed",
        iteration: int = 0,
    ) -> None:
        self.code = code
        self.iteration = iteration
        super().__init__(message)


@dataclass(slots=True)
class ReactLoopNode:
    model_provider: ChatModelProvider
    tool_registry: ToolRegistry
    max_iterations: int = 4
    tool_call_node: ToolCallNode | None = field(default=None, repr=False)
    name: str = field(default="react-loop", init=False)

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ReactLoopError("max_iterations must be positive", code="invalid_configuration")
        if self.tool_call_node is None:
            self.tool_call_node = ToolCallNode(self.tool_registry)

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        state.status = WorkflowStatus.RUNNING
        state.current_node = self.name
        state.data.pop("error", None)
        state.data.pop("answer", None)
        state.data["react_iterations"] = 0
        state.data.setdefault("tool_history", [])
        context.apply_usage(state)

        try:
            messages = _coerce_messages(state.data.get("messages"))
            if not isinstance(state.data["tool_history"], list):
                raise ReactLoopError("tool_history must be a list")

            for iteration in range(1, self.max_iterations + 1):
                state.data["react_iterations"] = iteration
                prompt_messages = context.prepare_context(state, messages)
                model_call_started = time.monotonic()
                try:
                    result = await self.model_provider.complete(
                        prompt_messages,
                        tools=self.tool_registry.definitions(),
                    )
                except Exception:
                    context.record_model_call(
                        "react_loop.complete",
                        None,
                        (time.monotonic() - model_call_started) * 1000,
                    )
                    context.apply_usage(state)
                    raise
                if result.usage is not None:
                    state.data["last_model_usage"] = result.usage.model_dump(mode="json")
                try:
                    context.record_model_call(
                        "react_loop.complete",
                        result.usage,
                        (time.monotonic() - model_call_started) * 1000,
                    )
                except WorkflowTokenBudgetExceededError as exc:
                    context.apply_usage(state)
                    raise ReactLoopError(
                        str(exc),
                        code=exc.code,
                        iteration=iteration,
                    ) from exc
                context.apply_usage(state)
                messages.append(_coerce_model_message(result))
                state.data["messages"] = messages
                state.data["last_model_finish_reason"] = result.finish_reason

                if not result.tool_calls:
                    state.status = WorkflowStatus.COMPLETED
                    state.current_node = self.name
                    state.data["answer"] = messages[-1].content
                    state.data["workflow_run_id"] = context.run_id
                    return state

                for tool_call in result.tool_calls:
                    state.data["tool_call"] = tool_call
                    await self.tool_call_node.execute(state, context)
                    state.data["tool_history"].append(
                        {
                            "call_id": state.data["tool_call_id"],
                            "name": state.data["tool_name"],
                            "arguments": deepcopy(state.data["tool_arguments"]),
                            "result": deepcopy(state.data["tool_result"]),
                        }
                    )
                    messages = _coerce_messages(state.data.get("messages"))
                    state.current_node = self.name

            error = ReactLoopError(
                f"react iteration limit reached: {self.max_iterations}",
                code="react_iteration_limit",
                iteration=self.max_iterations,
            )
            _mark_failed(state, error)
            raise error
        except ReactLoopError as exc:
            _mark_failed(state, exc)
            raise
        except ToolCallNodeError as exc:
            error = ReactLoopError(
                f"react tool call failed: {exc}",
                code="react_tool_call_failed",
                iteration=int(state.data.get("react_iterations", 0)),
            )
            _mark_failed(state, error)
            raise error from exc
        except Exception as exc:
            error = ReactLoopError(
                f"react loop failed: {exc}",
                iteration=int(state.data.get("react_iterations", 0)),
            )
            _mark_failed(state, error)
            raise error from exc


def _coerce_messages(raw_messages: object) -> list[ChatMessage]:
    if (
        isinstance(raw_messages, (str, bytes))
        or not isinstance(raw_messages, Sequence)
        or not raw_messages
    ):
        raise ReactLoopError("workflow state must contain non-empty messages")

    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        try:
            messages.append(
                raw_message
                if isinstance(raw_message, ChatMessage)
                else ChatMessage.model_validate(raw_message)
            )
        except (ValidationError, TypeError) as exc:
            raise ReactLoopError("workflow messages contain an invalid message") from exc
    return [message.model_copy(deep=True) for message in messages]


def _coerce_model_message(result: ModelResult) -> ChatMessage:
    try:
        return ChatMessage.model_validate(result.message).model_copy(deep=True)
    except (ValidationError, TypeError) as exc:
        raise ReactLoopError("model returned an invalid assistant message") from exc


def _mark_failed(state: WorkflowState, error: ReactLoopError) -> None:
    state.status = WorkflowStatus.FAILED
    state.current_node = "react-loop"
    state.data["error"] = {
        "code": error.code,
        "message": str(error),
        "iteration": error.iteration,
    }
