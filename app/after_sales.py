import json
import re
from collections.abc import Sequence
from uuid import uuid4

from pydantic import ValidationError

from app.contracts import (
    AfterSalesExtractionResponse,
    AfterSalesInfo,
    ChatMessage,
)
from app.providers import ChatModelProvider


class AfterSalesExtractionError(ValueError):
    pass


_EXTRACTION_INSTRUCTION = """Extract after-sales information from the conversation.
Return only a JSON object with these fields:
intent: one of refund, return, exchange, repair, logistics_issue, other, unknown
order_id: string or null
reason: string or null
requested_action: string or null
confidence: number from 0 to 1
Do not add any other fields."""


class AfterSalesExtractor:
    def __init__(self, model_provider: ChatModelProvider) -> None:
        self._model_provider = model_provider

    async def extract(
        self, messages: Sequence[ChatMessage]
    ) -> AfterSalesExtractionResponse:
        if not messages:
            raise AfterSalesExtractionError("at least one message is required")

        normalized_messages = tuple(
            ChatMessage.model_validate(message) for message in messages
        )

        result = await self._model_provider.complete(
            [
                ChatMessage(role="system", content=_EXTRACTION_INSTRUCTION),
                *normalized_messages,
            ]
        )
        try:
            payload = json.loads(_extract_json(result.message.content))
            info = AfterSalesInfo.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, AfterSalesExtractionError) as exc:
            raise AfterSalesExtractionError(
                "model response is not valid after-sales JSON"
            ) from exc

        return AfterSalesExtractionResponse(request_id=str(uuid4()), data=info)


def _extract_json(content: str) -> str:
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise AfterSalesExtractionError("model response does not contain a JSON object")
    return cleaned[start : end + 1]
