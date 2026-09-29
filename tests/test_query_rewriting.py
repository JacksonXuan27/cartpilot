import pytest

from app.query_rewriting import (
    QueryRewriter,
    QueryRewriteError,
    RuleBasedQueryRewriter,
)


@pytest.mark.asyncio
async def test_rule_based_rewriter_returns_normalized_query_first():
    rewriter = RuleBasedQueryRewriter(
        rules={"express": ("shipping", "delivery")}
    )

    variants = await rewriter.rewrite("  CHECK   EXPRESS  ")

    assert isinstance(rewriter, QueryRewriter)
    assert variants == ["check express", "check shipping", "check delivery"]


@pytest.mark.asyncio
async def test_rewriter_expands_short_phrase_and_deduplicates_variants():
    rewriter = RuleBasedQueryRewriter(
        rules={"express": ("shipping", "shipping")}
    )

    assert await rewriter.rewrite("check express") == [
        "check express",
        "check shipping",
    ]


@pytest.mark.asyncio
async def test_rewriter_respects_max_variants_and_keeps_rule_order():
    rewriter = RuleBasedQueryRewriter(
        rules={"express": ("shipping", "delivery")}
    )

    variants = await rewriter.rewrite("check express", max_variants=2)

    assert variants == ["check express", "check shipping"]


@pytest.mark.asyncio
async def test_rewriter_returns_normalized_query_when_no_rule_matches():
    rewriter = RuleBasedQueryRewriter(rules={"express": ("shipping",)})

    assert await rewriter.rewrite("  REFUND   POLICY ") == ["refund policy"]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", None])
async def test_rewriter_rejects_invalid_query(query):
    with pytest.raises(QueryRewriteError):
        await RuleBasedQueryRewriter().rewrite(query)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_variants", [0, -1])
async def test_rewriter_rejects_invalid_limit(max_variants):
    with pytest.raises(QueryRewriteError, match="max_variants"):
        await RuleBasedQueryRewriter().rewrite("query", max_variants=max_variants)


@pytest.mark.parametrize(
    "rules",
    [
        {"source": ()},
        {"source": ("source",)},
        {"": ("replacement",)},
    ],
)
def test_rewriter_rejects_invalid_rules(rules):
    with pytest.raises(QueryRewriteError):
        RuleBasedQueryRewriter(rules=rules)
