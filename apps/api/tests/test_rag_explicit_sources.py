from app.services.rag import RAGService
from app.services.retrieval import RetrievedChunk


def _chunk(text: str, chunk_id: int = 3) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-1",
        filename="Interview_Questions.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        chunk_id=chunk_id,
        text=text,
        start_char=0,
        end_char=len(text),
        score=0.9,
    )


def test_explicit_source_tokens_are_normalized_at_output():
    assert RAGService._normalize_answer(
        "The text explains S3. ⟦SRC:4⟧"
    ) == "The text explains S3. [4]"


def test_context_distinguishes_question_numbers_from_citations():
    context = RAGService._format_context([
        _chunk("16. What is Amazon S3?", chunk_id=15)
    ])
    assert "\nCitation token: ⟦SRC:1⟧\n" in context
    assert "\nFilename: Interview_Questions.docx\n" in context
    assert "16. What is Amazon S3?" in context


def test_list_requests_trigger_whole_document_retrieval():
    assert RAGService.REQUESTED_COUNT_PATTERN.search(
        "List top 10 services mentioned in the document."
    ).group(1) == "10"
    assert not RAGService.REQUESTED_COUNT_PATTERN.search(
        "What does Qdrant do?"
    )
