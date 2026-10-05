"""Unit tests for feature engineering.

Tests the batch implementation and the streaming single-row implementation,
and asserts that they agree exactly.
"""

import unittest

import numpy as np
import pandas as pd

from common.features import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    VelocityTracker,
    build_batch_features,
    build_feature_vector,
)


class TestBaseFeatures(unittest.TestCase):
    """Test the pure-row base feature extraction."""

    def test_base_features_amount(self):
        row = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 1000.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4000.0,
            "nameDest": "C2",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        
        # log_amount = log1p(amount)
        self.assertAlmostEqual(features["log_amount"].iloc[0], np.log1p(1000.0))
        # Label cols are dropped
        self.assertNotIn("isFraud", features.columns)
        self.assertNotIn("isFlaggedFraud", features.columns)

    def test_balance_delta_signs(self):
        row = {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 500.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 500.0,
            "nameDest": "C2",
            "oldbalanceDest": 200.0,
            "newbalanceDest": 700.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        
        self.assertEqual(features["orig_balance_delta"].iloc[0], -500.0)
        self.assertEqual(features["dest_balance_delta"].iloc[0], 500.0)

    def test_orig_balance_error_zero(self):
        # A perfectly balanced transaction
        row = {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 500.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 500.0,
            "nameDest": "C2",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 500.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        
        self.assertAlmostEqual(features["orig_balance_error"].iloc[0], 0.0)

    def test_dest_balance_error_nonzero(self):
        # PAYMENT dest is untouched, but amount implies movement
        row = {
            "step": 1,
            "type": "PAYMENT",
            "amount": 500.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 500.0,
            "nameDest": "M2",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        
        self.assertAlmostEqual(features["dest_balance_error"].iloc[0], 0.0) # wait, amount implies 0 movement for payment, so error should be 0.
        # Wait, the spec says "dest stays flat for PAYMENT type", and "dest_balance_error_nonzero" implies an error.
        # Actually dest_expected = 0 for PAYMENT. If dest stays flat, delta=0, expected=0, error=0.
        # Let's test a case where dest moves unexpectedly, or just that dest stays flat.
        
        # Let's check common.features logic:
        # dest_type = np.isin(tx_type, ("CASH_IN", "TRANSFER", "CASH_OUT"))
        # dest_expected = np.where(dest_type, amount, 0.0)
        # So for PAYMENT, dest_expected = 0.0.
        # If newbalanceDest = 500.0 (wrong for payment), error would be 500.0.
        row["newbalanceDest"] = 500.0
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        self.assertAlmostEqual(features["dest_balance_error"].iloc[0], 500.0)

    def test_one_hot_exclusive(self):
        row = {
            "step": 1,
            "type": "CASH_IN",
            "amount": 100.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 0.0,
            "newbalanceOrig": 100.0,
            "nameDest": "C2",
            "oldbalanceDest": 100.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        
        type_cols = [c for c in features.columns if c.startswith("type_")]
        sums = features[type_cols].sum(axis=1)
        self.assertEqual(sums.iloc[0], 1.0)
        self.assertEqual(features["type_CASH_IN"].iloc[0], 1.0)

    def test_orig_zeroed_flag(self):
        row = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 100.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 0.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C2",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        self.assertEqual(features["orig_zeroed"].iloc[0], 1.0)

    def test_orig_emptied_flag(self):
        row = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 100.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 100.0, # started with money
            "newbalanceOrig": 0.0,   # ended empty
            "nameDest": "C2",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
        df = pd.DataFrame([row])
        features = build_batch_features(df, with_velocity=False)
        self.assertEqual(features["orig_emptied"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()

class TestVelocityFeatures(unittest.TestCase):
    """Test the stateful velocity tracker and batch agreement."""

    def test_velocity_zero_on_first_tx(self):
        tracker = VelocityTracker(window=10, cache_max_accounts=10)
        row = {
            "step": 5, "type": "TRANSFER", "amount": 100.0,
            "nameOrig": "C1", "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
            "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        features = tracker.process(row)
        
        # We need to know indices of velocity features. 
        # FEATURE_NAMES has them at the end.
        idx_count = FEATURE_NAMES.index("orig_velocity_count")
        idx_amount = FEATURE_NAMES.index("orig_velocity_amount")
        
        self.assertEqual(features[idx_count], 0.0)
        self.assertEqual(features[idx_amount], 0.0)

    def test_velocity_accumulates(self):
        tracker = VelocityTracker(window=10, cache_max_accounts=10)
        row1 = {
            "step": 5, "type": "TRANSFER", "amount": 100.0,
            "nameOrig": "C1", "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
            "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        row2 = {
            "step": 7, "type": "TRANSFER", "amount": 200.0,
            "nameOrig": "C1", "oldbalanceOrg": 200.0, "newbalanceOrig": 0.0,
            "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        tracker.process(row1)
        features = tracker.process(row2)
        
        idx_count = FEATURE_NAMES.index("orig_velocity_count")
        idx_amount = FEATURE_NAMES.index("orig_velocity_amount")
        
        self.assertEqual(features[idx_count], 1.0)
        self.assertEqual(features[idx_amount], 100.0)

    def test_velocity_window_evicts(self):
        tracker = VelocityTracker(window=10, cache_max_accounts=10)
        row1 = {
            "step": 5, "type": "TRANSFER", "amount": 100.0,
            "nameOrig": "C1", "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
            "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        row2 = {
            "step": 16, "type": "TRANSFER", "amount": 200.0,
            "nameOrig": "C1", "oldbalanceOrg": 200.0, "newbalanceOrig": 0.0,
            "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        tracker.process(row1)
        features = tracker.process(row2)
        
        idx_count = FEATURE_NAMES.index("orig_velocity_count")
        idx_amount = FEATURE_NAMES.index("orig_velocity_amount")
        
        # row1 is at step 5, row2 at step 16. Window is 10.
        # lower bound for step 16 is 16 - 10 = 6.
        # Step 5 < 6, so it should be evicted.
        self.assertEqual(features[idx_count], 0.0)
        self.assertEqual(features[idx_amount], 0.0)

    def test_batch_stream_velocity_agreement(self):
        # drive _batch_velocity and VelocityTracker over the same 20-row frame
        np.random.seed(42)
        steps = np.sort(np.random.randint(1, 100, 20))
        accounts = np.random.choice(["C1", "C2", "C3"], 20)
        amounts = np.random.uniform(10, 1000, 20)
        
        rows = []
        for i in range(20):
            rows.append({
                "step": steps[i],
                "type": "TRANSFER",
                "amount": amounts[i],
                "nameOrig": accounts[i],
                "oldbalanceOrg": amounts[i],
                "newbalanceOrig": 0.0,
                "nameDest": "D1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": amounts[i],
                "isFraud": 0,
                "isFlaggedFraud": 0,
            })
            
        df = pd.DataFrame(rows)
        batch_features = build_batch_features(df, velocity_window=10)
        batch_count = batch_features["orig_velocity_count"].values
        batch_amount = batch_features["orig_velocity_amount"].values
        
        tracker = VelocityTracker(window=10, cache_max_accounts=100)
        stream_count = []
        stream_amount = []
        
        idx_count = FEATURE_NAMES.index("orig_velocity_count")
        idx_amount = FEATURE_NAMES.index("orig_velocity_amount")
        
        for row in rows:
            feat = tracker.process(row)
            stream_count.append(feat[idx_count])
            stream_amount.append(feat[idx_amount])
            
        np.testing.assert_allclose(batch_count, stream_count, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(batch_amount, stream_amount, rtol=1e-9, atol=1e-9)
