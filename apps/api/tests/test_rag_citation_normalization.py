from app.services.rag import RAGService


def test_normalizes_nested_brackets_around_cited_answer():
    answer = (
        "[RAG combines document retrieval with a large "
        "language model.[1]]"
    )

    assert RAGService._normalize_answer(answer) == (
        "RAG combines document retrieval with a large "
        "language model.[1]"
    )


def test_preserves_standalone_citation():
    assert RAGService._normalize_answer("[1]") == "[1]"


def test_removes_corrected_answer_tags():
    answer = (
        "<corrected_answer>\n"
        "Qdrant is used for vector storage.[1]\n"
        "</corrected_answer>"
    )

    assert RAGService._normalize_answer(answer) == (
        "Qdrant is used for vector storage.[1]"
    )


def test_removes_markdown_code_fence():
    answer = (
        "```text\n"
        "Qdrant is used for vector storage.[1]\n"
        "```"
    )

    assert RAGService._normalize_answer(answer) == (
        "Qdrant is used for vector storage.[1]"
    )


def test_accepts_valid_citation():
    assert RAGService._validate_citations(
        "Qdrant is used for vector storage.[1]",
        source_count=1,
    ) == "Qdrant is used for vector storage.[1]"


def test_rejects_missing_citations():
    assert RAGService._validate_citations(
        "RAG combines document retrieval with a large language model.",
        source_count=1,
    ) == RAGService.INVALID_CITATION_ANSWER


def test_rejects_out_of_range_citations():
    assert RAGService._validate_citations(
        "Qdrant is used for vector storage.[2]",
        source_count=1,
    ) == RAGService.INVALID_CITATION_ANSWER


def test_rejects_mixed_valid_and_invalid_citations():
    assert RAGService._validate_citations(
        "Qdrant is used for vector storage.[1][2]",
        source_count=1,
    ) == RAGService.INVALID_CITATION_ANSWER


def test_accepts_multiple_valid_citations():
    answer = "The documents describe both systems.[1][2]"

    assert RAGService._validate_citations(
        answer,
        source_count=2,
    ) == answer

def test_normalizes_explicit_source_tokens_to_public_citations():
    answer = "S3 is described in the source. ⟦SRC:4⟧"

    assert RAGService._normalize_answer(answer) == (
        "S3 is described in the source. [4]"
    )


def test_context_uses_source_tokens_distinct_from_document_question_numbers():
    from app.services.retrieval import RetrievedChunk

    chunk = RetrievedChunk(
        document_id="doc-1",
        filename="Interview_Questions.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        chunk_id=15,
        text="16. What is Amazon S3?",
        start_char=0,
        end_char=22,
        score=0.9,
    )

    context = RAGService._format_context([chunk])

    assert "Citation token: ⟦SRC:1⟧" in context
    assert "16. What is Amazon S3?" in context
    assert "Citation token: ⟦SRC:16⟧" not in context


def test_context_uses_real_line_breaks_between_source_fields():
    from app.services.retrieval import RetrievedChunk

    chunk = RetrievedChunk(
        document_id="doc-1",
        filename="Interview_Questions.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        chunk_id=3,
        text="Qdrant stores vectors.",
        start_char=0,
        end_char=22,
        score=0.9,
    )

    context = RAGService._format_context([chunk])

    assert "\nFilename: Interview_Questions.docx\n" in context
    assert "\nCitation token: ⟦SRC:1⟧\n" in context
    assert "\\nFilename:" not in context
