import json
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.contracts import ChatMessage
from app.providers import ChatModelProvider, ModelResult, ToolCall
from app.tool_registry import ToolArgumentError, ToolNotFoundError, ToolRegistry


class ToolOrchestrationError(RuntimeError):
    pass


class ToolExecutionError(ToolOrchestrationError):
    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(message)


class ToolTimeoutError(ToolExecutionError):
    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        super().__init__(
            tool_name,
            f"tool execution timed out: {tool_name} after {timeout_seconds:g}s",
        )


@dataclass(slots=True)
class ToolOrchestrator:
    model_provider: ChatModelProvider
    tool_registry: ToolRegistry
    max_rounds: int = 4
    tool_timeout_seconds: float = 5.0

    async def complete(self, messages: Sequence[ChatMessage]) -> ModelResult:
        if not messages:
            raise ToolOrchestrationError("at least one message is required")
        if self.max_rounds < 1:
            raise ToolOrchestrationError("max_rounds must be positive")
        if self.tool_timeout_seconds <= 0:
            raise ToolOrchestrationError("tool_timeout_seconds must be positive")

        conversation = [message.model_copy(deep=True) for message in messages]
        for _ in range(self.max_rounds):
            result = await self.model_provider.complete(
                conversation,
                tools=self.tool_registry.definitions(),
            )
            conversation.append(result.message.model_copy(deep=True))
            if not result.tool_calls:
                return result

            for tool_call in result.tool_calls:
                output = await self._execute(tool_call)
                conversation.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=tool_call.call_id,
                        content=self._serialize_output(output),
                    )
                )

        raise ToolOrchestrationError(
            f"tool call round limit exceeded: {self.max_rounds}"
        )

    async def _execute(self, tool_call: ToolCall) -> Any:
        try:
            async with asyncio.timeout(self.tool_timeout_seconds):
                return await self.tool_registry.execute(
                    tool_call.name, tool_call.arguments
                )
        except (ToolArgumentError, ToolNotFoundError):
            raise
        except TimeoutError as exc:
            raise ToolTimeoutError(
                tool_call.name, self.tool_timeout_seconds
            ) from exc
        except Exception as exc:
            raise ToolExecutionError(
                tool_call.name,
                f"tool execution failed: {tool_call.name}",
            ) from exc

    @staticmethod
    def _serialize_output(output: Any) -> str:
        if isinstance(output, BaseModel):
            output = output.model_dump(mode="json")
        return json.dumps(output, ensure_ascii=False, sort_keys=True)
