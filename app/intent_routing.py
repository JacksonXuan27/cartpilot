import unicodedata
from collections.abc import Mapping, Sequence
from enum import Enum

from pydantic import ValidationError

from app.contracts import ChatMessage
from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class IntentRoutingError(ValueError):
    pass


class IntentCategory(str, Enum):
    LOGISTICS = "logistics"
    ORDER = "order"
    PRODUCT = "product"
    REFUND_RETURN = "refund_return"
    AFTER_SALES = "after_sales"
    COMPLAINT = "complaint"
    HUMAN = "human"
    SMALLTALK = "smalltalk"
    OTHER = "other"


INTENT_ROUTES: dict[IntentCategory, str] = {
    IntentCategory.LOGISTICS: "business",
    IntentCategory.ORDER: "business",
    IntentCategory.PRODUCT: "knowledge",
    IntentCategory.REFUND_RETURN: "refund_flow",
    IntentCategory.AFTER_SALES: "refund_flow",
    IntentCategory.COMPLAINT: "escalate",
    IntentCategory.HUMAN: "business",
    IntentCategory.SMALLTALK: "fallback_script",
    IntentCategory.OTHER: "fallback_script",
}

DEFAULT_INTENT_TERMS: dict[IntentCategory, tuple[str, ...]] = {
    IntentCategory.HUMAN: (
        "转人工",
        "人工客服",
        "真人客服",
        "找人工",
        "人工",
        "human agent",
    ),
    IntentCategory.COMPLAINT: (
        "我要投诉",
        "投诉",
        "举报",
        "服务态度差",
        "complaint",
    ),
    IntentCategory.REFUND_RETURN: (
        "退款",
        "退货",
        "退钱",
        "退换",
        "七天无理由",
        "refund",
        "return",
    ),
    IntentCategory.AFTER_SALES: (
        "售后",
        "维修",
        "换新",
        "补发",
        "破损",
        "少件",
        "质量问题",
        "after sales",
    ),
    IntentCategory.LOGISTICS: (
        "物流",
        "快递",
        "运单",
        "配送",
        "发货",
        "签收",
        "揽收",
        "tracking",
        "shipment",
    ),
    IntentCategory.ORDER: (
        "订单状态",
        "订单信息",
        "订单查询",
        "订单",
        "下单",
        "order",
    ),
    IntentCategory.PRODUCT: (
        "商品咨询",
        "商品信息",
        "产品",
        "材质",
        "怎么用",
        "商品",
        "product",
    ),
    IntentCategory.SMALLTALK: (
        "你好",
        "您好",
        "谢谢",
        "再见",
        "hello",
        "thanks",
    ),
}

_INTENT_PRIORITY = tuple(DEFAULT_INTENT_TERMS)


def route_for_intent(intent: IntentCategory | str) -> str:
    try:
        category = intent if isinstance(intent, IntentCategory) else IntentCategory(intent)
    except ValueError as exc:
        raise IntentRoutingError(f"unsupported intent: {intent}") from exc
    return INTENT_ROUTES[category]


class IntentRouterNode:
    name = "intent-router"

    def __init__(
        self,
        terms: Mapping[IntentCategory, Sequence[str]] | None = None,
    ) -> None:
        configured_terms = terms if terms is not None else DEFAULT_INTENT_TERMS
        self._terms: dict[IntentCategory, tuple[str, ...]] = {}
        for raw_category, raw_terms in configured_terms.items():
            try:
                category = (
                    raw_category
                    if isinstance(raw_category, IntentCategory)
                    else IntentCategory(raw_category)
                )
            except ValueError as exc:
                raise IntentRoutingError(
                    f"unsupported intent category: {raw_category}"
                ) from exc
            if category is IntentCategory.OTHER:
                raise IntentRoutingError("other is reserved for unmatched queries")
            normalized_terms = tuple(
                _normalize_text(term) for term in raw_terms
            )
            if not normalized_terms or any(not term for term in normalized_terms):
                raise IntentRoutingError("each intent requires non-empty terms")
            if len(set(normalized_terms)) != len(normalized_terms):
                raise IntentRoutingError(f"duplicate terms for intent: {category.value}")
            self._terms[category] = normalized_terms

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        query = _extract_query(state.data)
        category, matched_terms, confidence = self._classify(query)
        state.status = WorkflowStatus.RUNNING
        state.current_node = self.name
        state.data["query"] = query
        state.data["intent"] = category.value
        state.data["route"] = route_for_intent(category)
        state.data["intent_confidence"] = confidence
        state.data["matched_intent_terms"] = list(matched_terms)
        state.data["workflow_run_id"] = context.run_id
        return state

    def _classify(
        self, query: str
    ) -> tuple[IntentCategory, tuple[str, ...], float]:
        normalized_query = _normalize_text(query)
        candidates: list[tuple[int, int, IntentCategory, tuple[str, ...]]] = []
        for priority, category in enumerate(_INTENT_PRIORITY):
            terms = self._terms.get(category, ())
            matched_spans = _non_overlapping_matches(normalized_query, terms)
            if matched_spans:
                score = sum(end - start for start, end, _ in matched_spans)
                matched_terms = tuple(term for _, _, term in matched_spans)
                candidates.append((score, -priority, category, matched_terms))

        if not candidates:
            return IntentCategory.OTHER, (), 0.0

        score, _, category, matched_terms = max(candidates, key=lambda item: (item[0], item[1]))
        confidence = min(0.87, 0.55 + 0.08 * min(score, 4))
        return category, matched_terms, round(confidence, 2)


def _extract_query(data: Mapping[str, object]) -> str:
    if "query" in data:
        query = data["query"]
        if not isinstance(query, str) or not query.strip():
            raise IntentRoutingError("query must be a non-empty string")
        return query.strip()

    messages = data.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise IntentRoutingError("workflow state must contain query or user messages")
    for raw_message in reversed(messages):
        try:
            message = (
                raw_message
                if isinstance(raw_message, ChatMessage)
                else ChatMessage.model_validate(raw_message)
            )
        except (ValidationError, TypeError) as exc:
            raise IntentRoutingError("workflow messages contain an invalid message") from exc
        if message.role == "user":
            return message.content.strip()
    raise IntentRoutingError("workflow state must contain a user message")


def _non_overlapping_matches(
    query: str, terms: Sequence[str]
) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    for term in terms:
        start = 0
        while (offset := query.find(term, start)) >= 0:
            matches.append((offset, offset + len(term), term))
            start = offset + 1
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))

    selected: list[tuple[int, int, str]] = []
    for candidate in matches:
        start, end, _ = candidate
        if all(end <= selected_start or start >= selected_end for selected_start, selected_end, _ in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item[0], item[2]))


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise IntentRoutingError("intent terms and query must be strings")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
