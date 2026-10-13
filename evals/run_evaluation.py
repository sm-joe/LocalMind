"""Run a self-contained LocalRAG evaluation against an isolated API instance.

The benchmark corpus is declared in the dataset JSON. This runner uploads each
corpus document with a per-run unique filename/content marker, evaluates source
identity by the runtime document IDs returned by the API, and deletes only the
IDs created by this run. It intentionally uses only the Python standard library.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "localrag_baseline.json"
RESULTS_DIR = ROOT_DIR / "evals" / "results"
API_BASE_URL = os.environ.get(
    "LOCALRAG_EVAL_API_URL", "http://localhost:18000"
).rstrip("/")
TOP_K = int(os.environ.get("LOCALRAG_EVAL_TOP_K", "3"))
REQUEST_TIMEOUT_SECONDS = int(
    os.environ.get("LOCALRAG_EVAL_REQUEST_TIMEOUT", "180")
)
HEALTH_TIMEOUT_SECONDS = int(
    os.environ.get("LOCALRAG_EVAL_HEALTH_TIMEOUT", "120")
)


def load_dataset() -> dict[str, Any]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: dict[str, Any]) -> None:
    if not isinstance(dataset.get("documents"), list) or not dataset["documents"]:
        raise ValueError("Evaluation dataset must define a non-empty documents list.")
    if not isinstance(dataset.get("cases"), list) or not dataset["cases"]:
        raise ValueError("Evaluation dataset must define a non-empty cases list.")

    document_ids: set[str] = set()
    for document in dataset["documents"]:
        logical_id = document.get("id")
        filename = document.get("filename")
        content = document.get("content")
        if not isinstance(logical_id, str) or not logical_id.strip():
            raise ValueError("Every evaluation document needs a non-empty logical id.")
        if logical_id in document_ids:
            raise ValueError(f"Duplicate evaluation document id: {logical_id}")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Evaluation document {logical_id!r} needs a filename.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Evaluation document {logical_id!r} has empty content.")
        document_ids.add(logical_id)

    case_ids: set[str] = set()
    for case in dataset["cases"]:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every evaluation case needs a non-empty id.")
        if case_id in case_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        case_ids.add(case_id)
        expected_sources = case.get("expected_sources")
        if not isinstance(expected_sources, list) or not expected_sources:
            raise ValueError(f"Evaluation case {case_id!r} needs expected_sources.")
        unknown_sources = set(expected_sources) - document_ids
        if unknown_sources:
            raise ValueError(
                f"Evaluation case {case_id!r} references unknown source IDs: "
                f"{sorted(unknown_sources)}"
            )


def request_json(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    request_headers = dict(headers or {})
    request_body = body
    if payload is not None:
        request_body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=request_body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API {method} {path} returned HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach API {API_BASE_URL}: {exc.reason}") from exc

    if not raw_body:
        return {}
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API {method} {path} returned invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"API {method} {path} returned an unexpected response shape.")
    return decoded


def wait_for_api() -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            request_json("/ready", timeout=5)
            print(f"API ready: {API_BASE_URL}")
            return
        except Exception as exc:  # readiness retries intentionally include connection and HTTP errors
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(
        f"LocalRAG API did not become ready within {HEALTH_TIMEOUT_SECONDS}s: {last_error}"
    )


def upload_document(document: dict[str, str], run_id: str) -> dict[str, Any]:
    boundary = f"----LocalRAGEvaluation{uuid.uuid4().hex}"
    filename = f"eval-{run_id}-{document['id']}.md"
    # The unique marker prevents the ingestion service's content-hash deduplication
    # from ever returning an existing user document ID for benchmark content.
    content = (
        document["content"].rstrip()
        + f"\n\n<!-- localrag-evaluation-run:{run_id} -->\n"
    ).encode("utf-8")
    parts = [
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/markdown; charset=utf-8\r\n\r\n"
        ).encode(),
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    response = request_json(
        "/documents/upload",
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        body=b"".join(parts),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    document_id = response.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise RuntimeError(f"Upload did not return a document_id for {filename!r}.")
    if response.get("chunks", 0) <= 0:
        raise RuntimeError(f"Evaluation document {filename!r} produced no chunks.")
    if response.get("status") not in {"indexed", "already_indexed"}:
        raise RuntimeError(
            f"Unexpected upload status for {filename!r}: {response.get('status')!r}"
        )
    return {
        "logical_id": document["id"],
        "document_id": document_id,
        "filename": response.get("filename", filename),
        "chunks": response["chunks"],
    }


def delete_document(document_id: str) -> None:
    encoded_id = urllib.parse.quote(document_id, safe="")
    request_json(
        f"/documents/{encoded_id}",
        method="DELETE",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def call_chat_api(question: str) -> dict[str, Any]:
    return request_json(
        "/chat",
        method="POST",
        payload={"question": question, "top_k": TOP_K},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def evaluate_concepts(
    answer: str,
    expected_concepts: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    matched: list[str] = []
    missing: list[str] = []
    details: list[dict[str, Any]] = []

    for concept in expected_concepts:
        name = str(concept["name"])
        variants = [str(value) for value in concept.get("variants", [])]
        matched_variant = next(
            (
                variant
                for variant in variants
                if normalize_text(variant)
                and normalize_text(variant) in normalized_answer
            ),
            None,
        )
        did_match = matched_variant is not None
        (matched if did_match else missing).append(name)
        details.append(
            {
                "name": name,
                "matched": did_match,
                "matched_variant": matched_variant,
            }
        )

    total = len(expected_concepts)
    score = len(matched) / total if total else 1.0
    return {
        "matched": matched,
        "missing": missing,
        "score": round(score, 3),
        "passed": not missing,
        "details": details,
    }


def extract_citations(answer: str) -> list[int]:
    citations: list[int] = []
    for match in re.findall(r"\[(\d+)\]", answer):
        citation = int(match)
        if citation not in citations:
            citations.append(citation)
    return citations


def evaluate_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    citations = extract_citations(answer)
    source_count = len(sources)
    valid = [value for value in citations if 1 <= value <= source_count]
    invalid = [value for value in citations if value < 1 or value > source_count]
    return {
        "citations": citations,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "source_count": source_count,
        "passed": bool(citations) and not invalid,
    }


def evaluate_source(
    response: dict[str, Any],
    expected_logical_source_ids: list[str],
    logical_ids_by_document_id: dict[str, str],
) -> dict[str, Any]:
    sources = response.get("sources", [])
    expected_set = set(expected_logical_source_ids)
    matches: list[dict[str, Any]] = []
    for rank, source in enumerate(sources, start=1):
        document_id = str(source.get("document_id", ""))
        logical_id = logical_ids_by_document_id.get(document_id)
        if logical_id in expected_set:
            matches.append(
                {
                    "logical_source_id": logical_id,
                    "document_id": document_id,
                    "rank": rank,
                    "filename": source.get("filename"),
                }
            )
    return {
        "expected_source_ids": expected_logical_source_ids,
        "matching_sources": matches,
        "matching_source_count": len(matches),
        "rank": min((item["rank"] for item in matches), default=None),
        "passed": bool(matches),
    }


def evaluate_case(
    case: dict[str, Any],
    source_ids_by_logical_id: dict[str, str],
    logical_ids_by_document_id: dict[str, str],
) -> dict[str, Any]:
    response = call_chat_api(case["question"])
    answer = str(response.get("answer", ""))
    sources = response.get("sources", [])
    missing_provisioned_sources = [
        logical_id for logical_id in case["expected_sources"]
        if logical_id not in source_ids_by_logical_id
    ]
    if missing_provisioned_sources:
        raise RuntimeError(
            f"Expected sources were not provisioned: {missing_provisioned_sources}"
        )
    source_evaluation = evaluate_source(
        response, case["expected_sources"], logical_ids_by_document_id
    )
    retrieval = response.get("retrieval", {})
    concept_evaluation = evaluate_concepts(answer, case.get("expected_concepts", []))
    citation_evaluation = evaluate_citations(answer, sources)
    retrieval_passed = source_evaluation["passed"] and bool(
        retrieval.get("has_context", False)
    )
    grounding_passed = retrieval_passed and citation_evaluation["passed"]
    answer_passed = concept_evaluation["passed"]
    passed = retrieval_passed and answer_passed and grounding_passed

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_answer": case.get("expected_answer", ""),
        "actual_answer": answer,
        "retrieval": {
            "expected_source_retrieved": source_evaluation["passed"],
            "expected_source_rank": source_evaluation["rank"],
            "candidate_count": retrieval.get("candidate_count", 0),
            "returned_count": retrieval.get("returned_count", 0),
            "filtered_count": retrieval.get("filtered_count", 0),
            "top_candidate_score": retrieval.get("top_candidate_score"),
            "top_returned_score": retrieval.get("top_score"),
            "has_context": retrieval.get("has_context", False),
            "source_count": len(sources),
        },
        "source_evaluation": source_evaluation,
        "concept_evaluation": concept_evaluation,
        "citation_evaluation": citation_evaluation,
        "grounding_passed": grounding_passed,
        "answer_passed": answer_passed,
        "passed": passed,
        "performance": response.get("performance", {}),
    }


def calculate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(bool(result.get("passed")) for result in results)
    retrieval_passed = sum(
        bool(result.get("retrieval", {}).get("expected_source_retrieved"))
        for result in results
    )
    context_passed = sum(
        bool(result.get("retrieval", {}).get("has_context")) for result in results
    )
    concept_passed = sum(bool(result.get("answer_passed")) for result in results)
    grounding_passed = sum(bool(result.get("grounding_passed")) for result in results)
    citation_passed = sum(
        bool(result.get("citation_evaluation", {}).get("passed")) for result in results
    )
    concept_scores = [
        result["concept_evaluation"]["score"]
        for result in results
        if "concept_evaluation" in result
    ]

    def rate(count: int) -> float:
        return round(count / total, 3) if total else 0.0

    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": rate(passed),
        "expected_source_recall": rate(retrieval_passed),
        "context_retrieval_rate": rate(context_passed),
        "concept_answer_pass_rate": rate(concept_passed),
        "citation_coverage": rate(citation_passed),
        "grounded_answer_rate": rate(grounding_passed),
        "average_concept_score": round(sum(concept_scores) / len(concept_scores), 3)
        if concept_scores
        else 0.0,
    }


def print_case_result(result: dict[str, Any]) -> None:
    retrieval = result.get("retrieval", {})
    concepts = result.get("concept_evaluation", {})
    citations = result.get("citation_evaluation", {})
    print(f"  Retrieval: {'PASS' if retrieval.get('expected_source_retrieved') else 'FAIL'}")
    print(f"  Concepts:  {'PASS' if concepts.get('passed') else 'FAIL'} ({concepts.get('score', 0):.0%})")
    print(f"  Citation:  {'PASS' if citations.get('passed') else 'FAIL'}")
    print(f"  Grounding: {'PASS' if result.get('grounding_passed') else 'FAIL'}")
    print(f"  Overall:   {'PASS' if result.get('passed') else 'FAIL'}")
    if result.get("error"):
        print(f"  Error:     {result['error']}")


def main() -> int:
    dataset = load_dataset()
    run_id = uuid.uuid4().hex[:12]
    created_documents: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    setup_error: str | None = None
    cleanup_errors: list[str] = []

    print("\n========================================")
    print("LocalRAG Evaluation")
    print("========================================")
    print(f"Dataset: {dataset['name']}")
    print(f"Dataset version: {dataset['version']}")
    print(f"Corpus documents: {len(dataset['documents'])}")
    print(f"Cases: {len(dataset['cases'])}")
    print(f"API: {API_BASE_URL}")

    try:
        wait_for_api()
        for document in dataset["documents"]:
            indexed = upload_document(document, run_id)
            created_documents.append(indexed)
            print(
                f"Prepared {document['id']}: {indexed['chunks']} chunks "
                f"(runtime ID {indexed['document_id']})"
            )

        source_ids_by_logical_id = {
            item["logical_id"]: item["document_id"] for item in created_documents
        }
        logical_ids_by_document_id = {
            item["document_id"]: item["logical_id"] for item in created_documents
        }

        for case in dataset["cases"]:
            print(f"\nRunning {case['id']}: {case['question']}")
            try:
                result = evaluate_case(
                    case,
                    source_ids_by_logical_id,
                    logical_ids_by_document_id,
                )
            except Exception as exc:
                result = {
                    "id": case["id"],
                    "question": case["question"],
                    "passed": False,
                    "error": str(exc),
                }
            print_case_result(result)
            results.append(result)
    except Exception as exc:
        setup_error = str(exc)
        print(f"\nERROR preparing evaluation: {setup_error}", file=sys.stderr)
    finally:
        # Delete only IDs returned by uploads in this run. Never delete by filename,
        # content query, collection-wide filter, or user-provided identifier.
        for item in reversed(created_documents):
            try:
                delete_document(item["document_id"])
                print(f"Cleaned evaluation document: {item['logical_id']}")
            except Exception as exc:
                message = f"{item['logical_id']} ({item['document_id']}): {exc}"
                cleanup_errors.append(message)
                print(f"ERROR cleaning evaluation document {message}", file=sys.stderr)

    summary = calculate_summary(results)
    output = {
        "dataset": dataset["name"],
        "dataset_version": dataset["version"],
        "run_id": run_id,
        "api_base_url": API_BASE_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "setup_error": setup_error,
        "cleanup_errors": cleanup_errors,
        "summary": summary,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "localrag-evaluation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n========================================")
    print("Evaluation Summary")
    print("========================================")
    print(f"Passed: {summary['passed_cases']}/{summary['total_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Expected source recall: {summary['expected_source_recall']:.1%}")
    print(f"Context retrieval rate: {summary['context_retrieval_rate']:.1%}")
    print(f"Concept answer pass rate: {summary['concept_answer_pass_rate']:.1%}")
    print(f"Citation coverage: {summary['citation_coverage']:.1%}")
    print(f"Grounded answer rate: {summary['grounded_answer_rate']:.1%}")
    print(f"Average concept score: {summary['average_concept_score']:.1%}")
    print(f"Cleanup errors: {len(cleanup_errors)}")
    print(f"Results written to: {output_path}")

    if setup_error or cleanup_errors or summary["failed_cases"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
