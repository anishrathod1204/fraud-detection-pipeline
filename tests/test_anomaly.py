"""Unit tests for anomaly models."""

import unittest

import numpy as np

from training.anomaly import Autoencoder, IsolationForest


class TestIsolationForest(unittest.TestCase):
    """Test the IsolationForest implementation."""

    def test_isolation_forest_scores_in_range(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        forest = IsolationForest(n_estimators=10, max_samples=32)
        forest.fit(X)
        scores = forest.score_samples(X)
        
        self.assertTrue(np.all(scores > 0.0))
        self.assertTrue(np.all(scores <= 1.0))

    def test_anomaly_scores_higher_for_outliers(self):
        np.random.seed(42)
        inliers = np.random.randn(100, 5)
        outliers = np.random.uniform(low=-10, high=10, size=(10, 5))
        
        X = np.vstack([inliers, outliers])
        forest = IsolationForest(n_estimators=50, max_samples=64)
        forest.fit(X)
        scores = forest.score_samples(X)
        
        inlier_scores = scores[:100]
        outlier_scores = scores[100:]
        
        self.assertGreater(outlier_scores.mean(), inlier_scores.mean())

    def test_isolation_forest_roundtrip(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        forest = IsolationForest(n_estimators=5, max_samples=16)
        forest.fit(X)
        
        scores_before = forest.score_samples(X)
        
        payload = forest.to_dict()
        forest2 = IsolationForest.from_dict(payload)
        
        scores_after = forest2.score_samples(X)
        np.testing.assert_allclose(scores_before, scores_after)


class TestAutoencoder(unittest.TestCase):
    """Test the Autoencoder implementation."""

    def test_autoencoder_reconstruction_error_shape(self):
        np.random.seed(42)
        X = np.random.randn(50, 4)
        ae = Autoencoder(n_features=4, hidden=8, latent=2)
        ae.fit(X, epochs=2)
        
        errors = ae.reconstruction_error(X)
        self.assertEqual(errors.shape, (50,))

    def test_autoencoder_error_higher_for_outliers(self):
        np.random.seed(42)
        inliers = np.random.randn(100, 4)
        # Shift outliers far from the origin
        outliers = np.random.randn(10, 4) + 10.0
        
        X = np.vstack([inliers, outliers])
        ae = Autoencoder(n_features=4, hidden=8, latent=2)
        # Train mostly on inliers (which is the case in practice)
        ae.fit(inliers, epochs=10, learning_rate=0.01)
        
        errors = ae.reconstruction_error(X)
        inlier_errors = errors[:100]
        outlier_errors = errors[100:]
        
        self.assertGreater(outlier_errors.mean(), inlier_errors.mean())

    def test_autoencoder_roundtrip(self):
        np.random.seed(42)
        X = np.random.randn(20, 3)
        ae = Autoencoder(n_features=3, hidden=4, latent=2)
        ae.fit(X, epochs=2)
        
        errors_before = ae.reconstruction_error(X)
        
        payload = ae.to_dict()
        ae2 = Autoencoder.from_dict(payload)
        
        errors_after = ae2.reconstruction_error(X)
        np.testing.assert_allclose(errors_before, errors_after)


if __name__ == "__main__":
    unittest.main()
