"""Unit tests for model bundle serialisation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.features import FeatureScaler
from training.anomaly import Autoencoder, IsolationForest
from training.bundle import BundleError, ModelBundle, load_bundle, save_bundle


class TestModelBundle(unittest.TestCase):
    """Test model bundle saving and loading."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_bundle(self):
        np.random.seed(42)
        X = np.random.randn(20, 3)
        feature_names = ["f1", "f2", "f3"]
        
        scaler = FeatureScaler.fit(X, feature_names)
        scaled_X = scaler.transform(X)
        
        forest = IsolationForest(n_estimators=5, max_samples=16).fit(scaled_X)
        ae = Autoencoder(n_features=3, hidden=4, latent=2).fit(scaled_X, epochs=2)
        
        bundle = ModelBundle(
            scaler=scaler,
            forest=forest,
            autoencoder=ae,
            feature_names=tuple(feature_names),
            velocity_window=10,
            threshold=0.8,
            metadata={"test": "data"},
        )
        
        save_bundle(bundle, self.artifact_dir)
        loaded = load_bundle(self.artifact_dir)
        
        self.assertEqual(loaded.feature_names, tuple(feature_names))
        self.assertEqual(loaded.velocity_window, 10)
        self.assertEqual(loaded.threshold, 0.8)
        self.assertEqual(loaded.metadata["test"], "data")
        
        if_scores, ae_scores = loaded.score(X)
        self.assertEqual(if_scores.shape, (20,))
        self.assertEqual(ae_scores.shape, (20,))

    def test_load_bundle_wrong_version(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.artifact_dir / "manifest.json"
        
        manifest = {
            "schema_version": 99,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        
        with self.assertRaises(BundleError) as ctx:
            load_bundle(self.artifact_dir)
        self.assertIn("not supported", str(ctx.exception))

    def test_load_bundle_missing_manifest(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(BundleError) as ctx:
            load_bundle(self.artifact_dir)
        self.assertIn("no manifest.json", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
