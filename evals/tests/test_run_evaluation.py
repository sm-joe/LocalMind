import sys
import unittest
from pathlib import Path
from unittest.mock import patch

EVALS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS_DIR))

import run_evaluation as evaluation  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_dataset_is_self_contained_and_has_multiple_documents(self):
        dataset = evaluation.load_dataset()
        self.assertGreaterEqual(len(dataset["documents"]), 4)
        self.assertGreaterEqual(len(dataset["cases"]), 5)
        self.assertTrue(all("content" in item for item in dataset["documents"]))
        self.assertTrue(
            all("expected_sources" in case for case in dataset["cases"])
        )
        self.assertTrue(
            all("expected_filename" not in case for case in dataset["cases"])
        )

    def test_dataset_rejects_unknown_logical_source(self):
        dataset = {
            "documents": [{"id": "doc-a", "filename": "a.md", "content": "text"}],
            "cases": [{"id": "case-a", "expected_sources": ["missing"]}],
        }
        with self.assertRaisesRegex(ValueError, "unknown source IDs"):
            evaluation.validate_dataset(dataset)

    def test_dataset_rejects_duplicate_document_ids(self):
        dataset = {
            "documents": [
                {"id": "same", "filename": "a.md", "content": "one"},
                {"id": "same", "filename": "b.md", "content": "two"},
            ],
            "cases": [{"id": "case-a", "expected_sources": ["same"]}],
        }
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation document id"):
            evaluation.validate_dataset(dataset)


class ScoringTests(unittest.TestCase):
    def test_source_matching_uses_runtime_document_id_not_filename(self):
        response = {
            "sources": [
                {"document_id": "runtime-123", "filename": "random-run-name.md"}
            ]
        }
        result = evaluation.evaluate_source(
            response,
            ["architecture"],
            {"runtime-123": "architecture"},
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["matching_sources"][0]["logical_source_id"], "architecture")

    def test_unexpected_document_id_does_not_satisfy_expected_source(self):
        response = {
            "sources": [
                {"document_id": "user-document-id", "filename": "architecture.md"}
            ]
        }
        result = evaluation.evaluate_source(
            response,
            ["architecture"],
            {"other-id": "architecture"},
        )
        self.assertFalse(result["passed"])

    def test_citation_indices_must_point_to_returned_sources(self):
        sources = [{"document_id": "doc-1"}]
        self.assertTrue(evaluation.evaluate_citations("Answer [1].", sources)["passed"])
        self.assertFalse(evaluation.evaluate_citations("Answer [2].", sources)["passed"])

    def test_concept_score_is_fraction_of_matched_concepts(self):
        result = evaluation.evaluate_concepts(
            "Qdrant is the vector database.",
            [
                {"name": "Qdrant", "variants": ["Qdrant"]},
                {"name": "vector database", "variants": ["vector database"]},
                {"name": "Ollama", "variants": ["Ollama"]},
            ],
        )
        self.assertEqual(result["score"], 0.667)
        self.assertFalse(result["passed"])


class ProvisioningTests(unittest.TestCase):
    @patch.object(evaluation, "request_json")
    def test_upload_uses_unique_runtime_filename_and_run_marker(self, request_json):
        request_json.return_value = {
            "document_id": "runtime-doc-1",
            "filename": "eval-run123-architecture.md",
            "chunks": 2,
            "status": "indexed",
        }
        result = evaluation.upload_document(
            {
                "id": "architecture",
                "filename": "architecture.md",
                "content": "Qdrant is the vector database.",
            },
            "run123",
        )
        self.assertEqual(result["document_id"], "runtime-doc-1")
        kwargs = request_json.call_args.kwargs
        body = kwargs["body"]
        self.assertIn(b"eval-run123-architecture.md", body)
        self.assertIn(b"localrag-evaluation-run:run123", body)

    @patch.object(evaluation, "request_json")
    def test_cleanup_targets_only_exact_document_id(self, request_json):
        request_json.return_value = {"document_id": "runtime-doc-1", "status": "deleted"}
        evaluation.delete_document("runtime-doc-1")
        request_json.assert_called_once_with(
            "/documents/runtime-doc-1",
            method="DELETE",
            timeout=evaluation.REQUEST_TIMEOUT_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
