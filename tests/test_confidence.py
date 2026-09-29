from uuid import UUID

import pytest

from app.confidence import (
    ConfidencePolicy,
    ConfidencePolicyError,
    ThresholdConfidencePolicy,
)
from app.vector_store import VectorMatch


DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000701")


def make_match(score: float) -> VectorMatch:
    return VectorMatch(
        record_id=f"record-{score}",
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content="退款政策",
        score=score,
        metadata={},
    )


def test_threshold_policy_implements_protocol_and_accepts_relevant_matches():
    policy = ThresholdConfidencePolicy(min_score=0.7)

    decision = policy.evaluate([make_match(0.8), make_match(0.4)])

    assert isinstance(policy, ConfidencePolicy)
    assert decision.accepted is True
    assert decision.top_score == 0.8
    assert decision.matched_count == 2
    assert decision.reason == "accepted"


def test_threshold_policy_rejects_low_confidence_matches():
    decision = ThresholdConfidencePolicy(min_score=0.7).evaluate(
        [make_match(0.69)]
    )

    assert decision.accepted is False
    assert decision.reason == "score_below_threshold"


def test_threshold_policy_marks_empty_results_without_fabricating_score():
    decision = ThresholdConfidencePolicy().evaluate([])

    assert decision.accepted is False
    assert decision.top_score is None
    assert decision.matched_count == 0
    assert decision.reason == "no_matches"


@pytest.mark.parametrize("min_score", [-0.1, 1.1, float("nan")])
def test_threshold_policy_rejects_invalid_threshold(min_score):
    with pytest.raises(ConfidencePolicyError, match="min_score"):
        ThresholdConfidencePolicy(min_score=min_score)


def test_threshold_policy_rejects_non_finite_match_score():
    with pytest.raises(ConfidencePolicyError, match="finite"):
        ThresholdConfidencePolicy().evaluate([make_match(float("nan"))])
