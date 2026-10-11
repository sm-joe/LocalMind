from app.services.rag import RAGService


def test_generation_diagnostics_do_not_log_answer_content(caplog):
    sensitive_answer = (
        "Sensitive internal document content must not appear in logs.[1]"
    )

    with caplog.at_level("INFO", logger="app.services.rag"):
        RAGService._log_generation_diagnostic(
            attempt="initial",
            answer=sensitive_answer,
            source_count=1,
        )

    assert sensitive_answer not in caplog.text
    assert "Sensitive internal document content" not in caplog.text
    assert "attempt=initial" in caplog.text
    assert "source_count=1" in caplog.text
    assert "has_citations=True" in caplog.text
    assert "citations_in_range=True" in caplog.text


def test_generation_diagnostics_handle_invalid_citation_numbers(caplog):
    answer = "This answer cites an unavailable source.[2]"

    with caplog.at_level("INFO", logger="app.services.rag"):
        RAGService._log_generation_diagnostic(
            attempt="correction",
            answer=answer,
            source_count=1,
        )

    assert answer not in caplog.text
    assert "attempt=correction" in caplog.text
    assert "citations_in_range=False" in caplog.text