import pytest
from pydantic import ValidationError

from app.faq_tools import (
    FAQEntry,
    FAQNotFoundError,
    FAQSearchInput,
    FAQSearchTool,
    InMemoryFAQRepository,
)


def make_entries() -> list[FAQEntry]:
    return [
        FAQEntry(
            faq_id="FAQ-REFUND-001",
            category="refund",
            title="退款到账时间",
            answer="审核通过后，原路退款通常需要三到五个工作日到账。",
            keywords=("退款", "到账", "原路退回"),
        ),
        FAQEntry(
            faq_id="FAQ-RETURN-001",
            category="return",
            title="退货申请条件",
            answer="商品保持完整并符合售后期限时，可以提交退货申请。",
            keywords=("退货", "售后期限", "申请条件"),
        ),
        FAQEntry(
            faq_id="FAQ-LOGISTICS-001",
            category="logistics",
            title="物流异常处理",
            answer="物流连续三天没有更新时，可以提交物流异常反馈。",
            keywords=("物流", "异常", "没有更新"),
        ),
    ]


@pytest.mark.asyncio
async def test_faq_tool_returns_matching_policy_entries():
    tool = FAQSearchTool(InMemoryFAQRepository(make_entries()))

    result = await tool.execute({"query": "退款多久到账"})

    assert tool.name == "faq.search"
    assert result.entries[0].faq_id == "FAQ-REFUND-001"
    assert "三到五个工作日" in result.entries[0].answer


@pytest.mark.asyncio
async def test_faq_tool_filters_by_category_and_limit():
    tool = FAQSearchTool(InMemoryFAQRepository(make_entries()))

    result = await tool.execute(
        {"query": "申请", "category": "return", "limit": 1}
    )

    assert len(result.entries) == 1
    assert result.entries[0].category == "return"


@pytest.mark.asyncio
async def test_faq_tool_raises_when_no_policy_matches():
    tool = FAQSearchTool(InMemoryFAQRepository(make_entries()))

    with pytest.raises(FAQNotFoundError, match="没有相关政策"):
        await tool.execute({"query": "没有相关政策"})


@pytest.mark.asyncio
async def test_repository_returns_isolated_entries():
    repository = InMemoryFAQRepository(make_entries())

    results = await repository.search("退款")
    results[0].keywords = ("changed",)

    stored = await repository.search("退款")
    assert stored[0].keywords == ("退款", "到账", "原路退回")


def test_faq_input_rejects_empty_queries_invalid_categories_and_limits():
    with pytest.raises(ValidationError):
        FAQSearchInput(query="")

    with pytest.raises(ValidationError):
        FAQSearchInput(query="refund", category="unknown")

    with pytest.raises(ValidationError):
        FAQSearchInput(query="refund", limit=11)
