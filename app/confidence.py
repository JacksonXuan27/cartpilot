import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.vector_store import VectorMatch


class ConfidencePolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConfidenceDecision:
    accepted: bool
    top_score: float | None
    matched_count: int
    reason: str


@runtime_checkable
class ConfidencePolicy(Protocol):
    def evaluate(self, matches: Sequence[VectorMatch]) -> ConfidenceDecision:
        """Decide whether retrieved context is reliable enough to answer."""


@dataclass(frozen=True, slots=True)
class ThresholdConfidencePolicy:
    min_score: float = 0.35

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ConfidencePolicyError("min_score must be finite and between 0 and 1")

    def evaluate(self, matches: Sequence[VectorMatch]) -> ConfidenceDecision:
        if not matches:
            return ConfidenceDecision(
                accepted=False,
                top_score=None,
                matched_count=0,
                reason="no_matches",
            )

        top_score = max(match.score for match in matches)
        if not math.isfinite(top_score):
            raise ConfidencePolicyError("match scores must be finite")
        accepted = top_score >= self.min_score
        return ConfidenceDecision(
            accepted=accepted,
            top_score=top_score,
            matched_count=len(matches),
            reason="accepted" if accepted else "score_below_threshold",
        )
