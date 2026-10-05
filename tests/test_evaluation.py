"""Unit tests for evaluation metrics."""

import unittest

import numpy as np

from training.evaluation import (
    average_precision,
    confusion_at_threshold,
    roc_auc,
    select_threshold,
)


class TestEvaluation(unittest.TestCase):
    """Test evaluation metrics and threshold selection."""

    def test_perfect_classifier_pr_auc_1(self):
        labels = np.array([0, 1, 0, 1, 0])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        pr_auc = average_precision(labels, scores)
        self.assertAlmostEqual(pr_auc, 1.0)

    def test_random_classifier_pr_auc_near_base_rate(self):
        # 10% base rate
        np.random.seed(42)
        labels = np.zeros(10000)
        labels[:1000] = 1
        scores = np.random.uniform(0, 1, 10000)
        
        pr_auc = average_precision(labels, scores)
        # Should be roughly 0.1
        self.assertTrue(0.08 <= pr_auc <= 0.12)

    def test_roc_auc_perfect_1(self):
        labels = np.array([0, 1, 0, 1, 0])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        roc = roc_auc(labels, scores)
        self.assertAlmostEqual(roc, 1.0)

    def test_roc_auc_inverted_0(self):
        labels = np.array([0, 1, 0, 1, 0])
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        roc = roc_auc(labels, scores)
        self.assertAlmostEqual(roc, 0.0)

    def test_roc_auc_random_approx_half(self):
        np.random.seed(42)
        labels = np.zeros(1000)
        labels[:100] = 1
        scores = np.random.uniform(0, 1, 1000)
        roc = roc_auc(labels, scores)
        self.assertTrue(0.45 <= roc <= 0.55)

    def test_select_threshold_meets_recall_floor(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        # To get 75% recall, we need 3 out of 4 true positives.
        # Threshold at 0.7 gives exactly 3 true positives and 0 false positives.
        choice = select_threshold(labels, scores, min_recall=0.75)
        self.assertEqual(choice.threshold, 0.7)
        self.assertEqual(choice.recall, 0.75)
        self.assertEqual(choice.precision, 1.0)

    def test_select_threshold_fallback_when_floor_unreachable(self):
        labels = np.array([1, 1, 0, 0])
        # One positive is scored lower than everything else
        scores = np.array([0.9, 0.1, 0.8, 0.7])
        # A floor of 1.0 requires catching the 0.1 score.
        # Catching 0.1 means threshold 0.1, which flags everything.
        choice = select_threshold(labels, scores, min_recall=1.0)
        self.assertEqual(choice.threshold, 0.1)
        self.assertEqual(choice.recall, 1.0)
        self.assertEqual(choice.precision, 0.5)

    def test_confusion_at_threshold_counts(self):
        labels = np.array([1, 1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.4, 0.7, 0.6, 0.5, 0.2])
        # Threshold 0.7 flags: 0.9(TP), 0.8(TP), 0.7(FP)
        choice = confusion_at_threshold(labels, scores, 0.7)
        self.assertEqual(choice.true_positives, 2)
        self.assertEqual(choice.false_positives, 1)
        self.assertEqual(choice.false_negatives, 1) # 0.4(FN)
        self.assertEqual(choice.true_negatives, 3) # 0.6, 0.5, 0.2 (TN)
        
        self.assertEqual(choice.precision, 2 / 3)
        self.assertEqual(choice.recall, 2 / 3)

if __name__ == "__main__":
    unittest.main()
