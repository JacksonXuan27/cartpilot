import html
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser


class DocumentProcessingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    index: int
    content: str
    start_char: int
    end_char: int


class _VisibleTextParser(HTMLParser):
    _ignored_tags = frozenset({"script", "style", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in self._ignored_tags:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and normalized_tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._ignored_depth == 0 and normalized_tag in {
            "div",
            "li",
            "p",
            "section",
            "article",
            "h1",
            "h2",
            "h3",
        }:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0 and data.strip():
            self.parts.append(data)


class DocumentCleaner:
    def clean(self, content: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise DocumentProcessingError("document content cannot be empty")

        normalized = unicodedata.normalize("NFKC", content)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        if re.search(r"<\/?[a-z][^>]*>", normalized, re.IGNORECASE):
            parser = _VisibleTextParser()
            parser.feed(normalized)
            parser.close()
            normalized = "".join(parser.parts)
        normalized = html.unescape(normalized).replace("\xa0", " ")
        normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n[ \t]+", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()


class DocumentChunker:
    def __init__(self, max_chars: int = 800, overlap_chars: int = 120) -> None:
        if max_chars < 1:
            raise DocumentProcessingError("max_chars must be positive")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise DocumentProcessingError(
                "overlap_chars must be non-negative and smaller than max_chars"
            )
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, content: str) -> list[DocumentChunk]:
        cleaned = DocumentCleaner().clean(content)
        chunks: list[DocumentChunk] = []
        cursor = 0
        index = 0
        while cursor < len(cleaned):
            end = min(cursor + self.max_chars, len(cleaned))
            if end < len(cleaned):
                boundary = self._find_boundary(cleaned, cursor, end)
                if boundary > cursor:
                    end = boundary
            chunk_content = cleaned[cursor:end].strip()
            if chunk_content:
                actual_start = cursor + len(cleaned[cursor:end]) - len(
                    cleaned[cursor:end].lstrip()
                )
                actual_end = actual_start + len(chunk_content)
                chunks.append(
                    DocumentChunk(
                        index=index,
                        content=chunk_content,
                        start_char=actual_start,
                        end_char=actual_end,
                    )
                )
                index += 1
            if end >= len(cleaned):
                break
            next_cursor = max(end - self.overlap_chars, cursor + 1)
            cursor = next_cursor
        return chunks

    def _find_boundary(self, content: str, start: int, end: int) -> int:
        window = content[start:end]
        paragraph_boundary = window.rfind("\n\n")
        if paragraph_boundary >= max(1, len(window) // 3):
            return start + paragraph_boundary
        whitespace_boundary = max(window.rfind(" "), window.rfind("\n"))
        if whitespace_boundary >= max(1, len(window) // 2):
            return start + whitespace_boundary
        return end
