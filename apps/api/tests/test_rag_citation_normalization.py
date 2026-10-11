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