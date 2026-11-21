import unittest

from tests.boundary_corpus import buildBoundaryCorpus, compareBoundaryStrategies, evaluateBoundaryRecovery


class BoundaryAccuracyTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = buildBoundaryCorpus()

    def test_corpus_has_one_hundred_samples_and_expected_truth(self):
        self.assertEqual(len(self.samples), 100)
        self.assertEqual(sum(len(sample.truth) for sample in self.samples), 611)
        self.assertTrue(all(sample.tailcall_target in sample.truth for sample in self.samples))

    def test_baseline_misses_the_tailcalled_boundary(self):
        metrics = evaluateBoundaryRecovery(self.samples, "baseline")
        self.assertEqual(metrics["true_positives"], 511)
        self.assertEqual(metrics["false_negatives"], 100)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertAlmostEqual(metrics["recall"], 0.8363, places=4)

    def test_evidence_strategy_recovers_all_boundaries_without_false_positives(self):
        metrics = evaluateBoundaryRecovery(self.samples, "evidence")
        self.assertEqual(metrics["true_positives"], 611)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_comparison_reports_both_strategies(self):
        comparison = compareBoundaryStrategies(self.samples)
        self.assertEqual(comparison["metric"], "exact function-start match")
        self.assertEqual(comparison["baseline_default"]["f1"], 0.9109)
        self.assertEqual(comparison["evidence"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
