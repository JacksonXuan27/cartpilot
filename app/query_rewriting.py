import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class QueryRewriteError(ValueError):
    pass


DEFAULT_REWRITE_RULES: dict[str, tuple[str, ...]] = {
    "怎么查快递": ("如何查询物流", "物流查询"),
    "查快递": ("查询物流",),
    "快递": ("物流",),
    "退钱": ("退款",),
    "多久到账": ("到账时间",),
    "几天到账": ("到账时间",),
    "不想要": ("退货",),
}


@runtime_checkable
class QueryRewriter(Protocol):
    async def rewrite(self, query: str, max_variants: int = 3) -> list[str]:
        """Return deterministic query variants, including the normalized query."""


@dataclass(slots=True)
class RuleBasedQueryRewriter:
    rules: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: dict(DEFAULT_REWRITE_RULES)
    )
    _normalized_rules: tuple[tuple[str, tuple[str, ...]], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        normalized_rules: list[tuple[str, tuple[str, ...]]] = []
        for source, replacements in self.rules.items():
            normalized_source = _normalize_query(source)
            if not replacements:
                raise QueryRewriteError("each rule must have a replacement")
            normalized_replacements = tuple(
                _normalize_query(replacement) for replacement in replacements
            )
            if any(
                replacement == normalized_source
                for replacement in normalized_replacements
            ):
                raise QueryRewriteError("a rule replacement cannot equal its source")
            normalized_rules.append((normalized_source, normalized_replacements))
        object.__setattr__(self, "_normalized_rules", tuple(normalized_rules))

    async def rewrite(self, query: str, max_variants: int = 3) -> list[str]:
        if max_variants < 1:
            raise QueryRewriteError("max_variants must be positive")
        normalized_query = _normalize_query(query)
        variants: list[str] = [normalized_query]
        seen = {normalized_query}

        for source, replacements in self._normalized_rules:
            if source not in normalized_query:
                continue
            for replacement in replacements:
                variant = normalized_query.replace(source, replacement)
                if variant not in seen:
                    variants.append(variant)
                    seen.add(variant)
                if len(variants) >= max_variants:
                    return variants
        return variants


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise QueryRewriteError("query must be a string")
    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise QueryRewriteError("query cannot be empty")
    return normalized
