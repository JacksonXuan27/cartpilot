import pytest

from app.document_processing import (
    DocumentChunker,
    DocumentCleaner,
    DocumentProcessingError,
)


def test_cleaner_normalizes_html_entities_whitespace_and_ignored_content():
    content = """
    <h1>退款&nbsp;政策</h1>
    <p>原路退款通常需要&nbsp;3 个工作日。</p>
    <script>tracking_id = 'secret';</script>
    """

    cleaned = DocumentCleaner().clean(content)

    assert cleaned == "退款 政策\n原路退款通常需要 3 个工作日。"
    assert "tracking_id" not in cleaned


def test_cleaner_normalizes_unicode_and_line_endings():
    cleaned = DocumentCleaner().clean("ＡＩ\r\n\r\n客服\u200b \t系统")

    assert cleaned == "AI\n\n客服 系统"


def test_chunker_prefers_paragraph_boundaries_and_keeps_overlap():
    content = "第一段包含订单查询说明。\n\n第二段包含退款政策说明。\n\n第三段包含物流说明。"

    chunks = DocumentChunker(max_chars=18, overlap_chars=5).split(content)

    assert len(chunks) >= 3
    assert chunks[0].index == 0
    assert chunks[0].content.startswith("第一段")
    assert chunks[0].end_char > chunks[0].start_char
    assert any(
        set(previous.content) & set(current.content)
        for previous, current in zip(chunks, chunks[1:])
    )


def test_chunker_splits_long_paragraphs_without_losing_content():
    content = "这是一个很长的段落，" * 30
    chunks = DocumentChunker(max_chars=40, overlap_chars=8).split(content)

    assert len(chunks) > 1
    assert chunks[0].content
    assert chunks[-1].content
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(content)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DocumentCleaner().clean(" "),
        lambda: DocumentChunker(max_chars=0),
        lambda: DocumentChunker(max_chars=10, overlap_chars=10),
    ],
)
def test_document_processing_rejects_invalid_inputs(factory):
    with pytest.raises(DocumentProcessingError):
        factory()
