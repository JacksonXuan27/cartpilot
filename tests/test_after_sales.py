import pytest
from fastapi.testclient import TestClient

from app.after_sales import AfterSalesExtractionError, AfterSalesExtractor
from app.chat import ChatService
from app.main import app
from app.providers import StubModelProvider
from app.sessions import InMemorySessionStore


def make_extractor(reply: str) -> AfterSalesExtractor:
    return AfterSalesExtractor(StubModelProvider(reply=reply))


@pytest.mark.asyncio
async def test_extractor_parses_structured_after_sales_information():
    extractor = make_extractor(
        '{"intent":"refund","order_id":"ORD-1001",'
        '"reason":"damaged","requested_action":"refund",'
        '"confidence":0.95}'
    )

    result = await extractor.extract(
        [{"role": "user", "content": "订单 ORD-1001 到货破损，想退款"}]
    )

    assert result.data.intent == "refund"
    assert result.data.order_id == "ORD-1001"
    assert result.data.confidence == 0.95


@pytest.mark.asyncio
async def test_extractor_accepts_a_markdown_json_code_block():
    extractor = make_extractor(
        "```json\n"
        '{"intent":"return","order_id":null,"reason":"wrong size",'
        '"requested_action":"return","confidence":0.8}\n'
        "```"
    )

    result = await extractor.extract(
        [{"role": "user", "content": "The size is wrong."}]
    )

    assert result.data.intent == "return"
    assert result.data.order_id is None


@pytest.mark.asyncio
async def test_extractor_rejects_invalid_model_output():
    extractor = make_extractor('{"intent":"not-supported"}')

    with pytest.raises(AfterSalesExtractionError):
        await extractor.extract(
            [{"role": "user", "content": "I need help with an order."}]
        )


@pytest.mark.asyncio
async def test_extractor_requires_messages():
    extractor = make_extractor("{}")

    with pytest.raises(AfterSalesExtractionError, match="at least one message"):
        await extractor.extract([])


def test_after_sales_endpoint_returns_structured_data():
    provider = StubModelProvider(
        reply=(
            '{"intent":"exchange","order_id":"ORD-2002",'
            '"reason":"wrong color","requested_action":"exchange",'
            '"confidence":0.88}'
        )
    )
    app.state.chat_service = ChatService(
        session_store=InMemorySessionStore(), model_provider=provider
    )
    app.state.after_sales_extractor = AfterSalesExtractor(provider)

    response = TestClient(app).post(
        "/after-sales/extract",
        json={
            "messages": [
                {"role": "user", "content": "ORD-2002 换成其他颜色"}
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["intent"] == "exchange"
    assert response.json()["data"]["order_id"] == "ORD-2002"


def test_after_sales_endpoint_reports_invalid_model_output():
    provider = StubModelProvider(reply="not json")
    app.state.after_sales_extractor = AfterSalesExtractor(provider)

    response = TestClient(app).post(
        "/after-sales/extract",
        json={"messages": [{"role": "user", "content": "Please help"}]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_after_sales_output"
