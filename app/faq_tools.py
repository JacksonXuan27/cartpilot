from collections.abc import Iterable, Mapping
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


FAQCategory = Literal[
    "refund",
    "return",
    "exchange",
    "repair",
    "logistics",
    "payment",
    "general",
]


class FAQNotFoundError(LookupError):
    pass


class FAQEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faq_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    category: FAQCategory
    title: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    keywords: tuple[str, ...] = Field(min_length=1)


class FAQSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    category: FAQCategory | None = None
    limit: int = Field(default=3, ge=1, le=10)


class FAQSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    entries: list[FAQEntry] = Field(min_length=1)


class FAQRepository(Protocol):
    async def search(
        self,
        query: str,
        category: FAQCategory | None = None,
        limit: int = 3,
    ) -> list[FAQEntry]:
        """Return matching FAQ entries ordered by relevance."""


class InMemoryFAQRepository:
    def __init__(self, entries: Iterable[FAQEntry] = ()) -> None:
        self._entries = [entry.model_copy(deep=True) for entry in entries]

    async def search(
        self,
        query: str,
        category: FAQCategory | None = None,
        limit: int = 3,
    ) -> list[FAQEntry]:
        normalized_query = query.casefold().strip()
        candidates = [
            entry
            for entry in self._entries
            if category is None or entry.category == category
        ]

        ranked = sorted(
            (
                (self._score(entry, normalized_query), index, entry)
                for index, entry in enumerate(candidates)
                if self._score(entry, normalized_query) > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [entry.model_copy(deep=True) for _, _, entry in ranked[:limit]]

    @staticmethod
    def _score(entry: FAQEntry, query: str) -> int:
        title_and_answer = [entry.title, entry.answer]
        score = sum(1 for value in title_and_answer if query in value.casefold())
        score += sum(
            1
            for keyword in entry.keywords
            if keyword.casefold() in query or query in keyword.casefold()
        )
        return score


class FAQSearchTool:
    name = "faq.search"
    description = "Search approved customer-support policies and FAQ answers."
    input_model = FAQSearchInput

    def __init__(self, repository: FAQRepository) -> None:
        self._repository = repository

    async def execute(self, arguments: Mapping[str, object]) -> FAQSearchResult:
        query = self.input_model.model_validate(arguments)
        entries = await self._repository.search(
            query=query.query,
            category=query.category,
            limit=query.limit,
        )
        if not entries:
            raise FAQNotFoundError(f"no FAQ matched query: {query.query}")
        return FAQSearchResult(query=query.query, entries=entries)
