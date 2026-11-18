import unittest

from evals.run_evaluation import evaluate_citations, evaluate_concepts


class EvaluationMetricTests(unittest.TestCase):
    def test_fails_unverifiable_citation(self):
        result = evaluate_citations(
            "The answer refers to nonexistent source [4].",
            [{"citation": "[1]"}],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_citations"], [4])

    def test_passes_supported_citation(self):
        result = evaluate_citations(
            "Answer cites the correct source [1].",
            [{"citation": "[1]"}],
        )
        self.assertTrue(result["passed"])

    def test_fails_missing_expected_concept(self):
        result = evaluate_concepts(
            "Only Qdrant is mentioned.",
            [{"name": "Ollama", "variants": ["Ollama"]}],
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
