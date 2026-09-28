import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID


class KeywordRetrievalError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class KeywordDocument:
    record_id: str
    document_id: UUID
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KeywordMatch:
    record_id: str
    document_id: UUID
    content: str
    score: float
    metadata: dict[str, str]


@dataclass(slots=True)
class BM25Index:
    k1: float = 1.2
    b: float = 0.75
    _documents: dict[str, KeywordDocument] = field(default_factory=dict, init=False)
    _term_frequencies: dict[str, Counter[str]] = field(default_factory=dict, init=False)
    _document_frequencies: Counter[str] = field(default_factory=Counter, init=False)
    _document_lengths: dict[str, int] = field(default_factory=dict, init=False)
    _total_document_length: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise KeywordRetrievalError("k1 must be non-negative")
        if not 0 <= self.b <= 1:
            raise KeywordRetrievalError("b must be between 0 and 1")

    @property
    def size(self) -> int:
        return len(self._documents)

    def upsert(self, documents: list[KeywordDocument]) -> int:
        for document in documents:
            tokens = _tokenize(document.content)
            if not tokens:
                raise KeywordRetrievalError(
                    f"document content cannot be empty: {document.record_id}"
                )
        for document in documents:
            self._remove_statistics(document.record_id)
            self._documents[document.record_id] = document
            self._add_statistics(document.record_id, document.content)
        return len(documents)

    def remove(self, record_id: str) -> bool:
        if record_id not in self._documents:
            return False
        self._remove_statistics(record_id)
        del self._documents[record_id]
        return True

    def search(self, query: str, top_k: int = 5) -> list[KeywordMatch]:
        if not isinstance(query, str) or not query.strip():
            raise KeywordRetrievalError("query cannot be empty")
        if top_k < 1:
            raise KeywordRetrievalError("top_k must be positive")

        query_terms = _tokenize(query)
        if not query_terms or not self._documents:
            return []
        document_count = len(self._documents)
        average_length = self._total_document_length / document_count
        scored: list[tuple[float, KeywordDocument]] = []
        for record_id, document in self._documents.items():
            term_frequencies = self._term_frequencies[record_id]
            document_length = self._document_lengths[record_id]
            score = sum(
                self._term_score(
                    term,
                    term_frequencies.get(term, 0),
                    document_length,
                    average_length,
                    document_count,
                )
                for term in query_terms
            )
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: (-item[0], item[1].record_id))
        return [
            KeywordMatch(
                record_id=document.record_id,
                document_id=document.document_id,
                content=document.content,
                score=score,
                metadata=dict(document.metadata),
            )
            for score, document in scored[:top_k]
        ]

    def _add_statistics(self, record_id: str, content: str) -> None:
        frequencies = Counter(_tokenize(content))
        self._term_frequencies[record_id] = frequencies
        self._document_lengths[record_id] = sum(frequencies.values())
        self._total_document_length += self._document_lengths[record_id]
        for term in frequencies:
            self._document_frequencies[term] += 1

    def _remove_statistics(self, record_id: str) -> None:
        frequencies = self._term_frequencies.pop(record_id, None)
        if frequencies is None:
            return
        self._total_document_length -= self._document_lengths.pop(record_id)
        for term in frequencies:
            self._document_frequencies[term] -= 1
            if self._document_frequencies[term] == 0:
                del self._document_frequencies[term]

    def _term_score(
        self,
        term: str,
        term_frequency: int,
        document_length: int,
        average_length: float,
        document_count: int,
    ) -> float:
        if term_frequency == 0:
            return 0.0
        document_frequency = self._document_frequencies.get(term, 0)
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        normalization = 1 - self.b + self.b * document_length / average_length
        return inverse_document_frequency * (
            term_frequency * (self.k1 + 1)
            / (term_frequency + self.k1 * normalization)
        )


def _tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for segment in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized):
        tokens.append(segment)
    return tokens
