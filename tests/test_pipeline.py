"""Unit tests for everything that does not need Kafka or Cassandra."""
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.cassandra_store import COLUMNS, hour_bucket  # noqa: E402
from common.features import FEATURE_COLUMNS, build_features  # noqa: E402
from producer.producer import make_message  # noqa: E402
from scripts.generate_data import generate  # noqa: E402
from streaming.scorer import score_messages, to_rows  # noqa: E402
from training import train as train_mod  # noqa: E402


def _tmp(tmp_path, name):
    return str(tmp_path / name)


def test_generator_shape_and_rate():
    df = generate(5000, 0.02, seed=1)
    assert len(df) == 5000
    assert df["isFraud"].sum() == 100
    assert set(df["type"]) <= {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    assert (df.loc[df.isFraud == 1, "type"].isin(["TRANSFER", "CASH_OUT"])).all()


def test_features_columns_and_no_nan():
    df = generate(2000, 0.02, seed=2)
    X = build_features(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()


def test_features_reject_missing_columns():
    try:
        build_features(pd.DataFrame({"amount": [1.0]}))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fraud_features_differ_from_legit():
    df = generate(20000, 0.02, seed=3)
    X = build_features(df)
    assert X.loc[df.isFraud == 1, "orig_drained"].mean() > 0.95
    assert X.loc[df.isFraud == 0, "orig_drained"].mean() < 0.05


def test_end_to_end_training_and_scoring(tmp_path):
    data = _tmp(tmp_path, "d.csv")
    generate(30000, 0.02, seed=4).to_csv(data, index=False)
    model_path = _tmp(tmp_path, "m.joblib")
    report = _tmp(tmp_path, "r.md")
    metrics = train_mod.main(data, model_path, report)
    assert metrics["roc_auc"] > 0.95
    assert metrics["recall"] > 0.5
    assert os.path.exists(report)

    import joblib
    bundle = joblib.load(model_path)
    df = pd.read_csv(data).sample(300, random_state=0)
    msgs = [make_message(r) for r in df.to_dict("records")]
    scored = score_messages(bundle, msgs)
    assert len(scored) == 300
    assert scored["predicted_fraud"].dtype == bool
    rows, alerts = to_rows(scored)
    assert len(rows) == 300
    assert all(len(r) == len(COLUMNS) for r in rows)
    assert len(alerts) == int(scored["predicted_fraud"].sum())


def test_scoring_drops_malformed_messages(tmp_path):
    data = _tmp(tmp_path, "d.csv")
    generate(10000, 0.02, seed=5).to_csv(data, index=False)
    model_path = _tmp(tmp_path, "m.joblib")
    train_mod.main(data, model_path, _tmp(tmp_path, "r.md"))
    import joblib
    bundle = joblib.load(model_path)
    good = make_message(pd.read_csv(data).iloc[0].to_dict())
    bad = {"txn_id": "x", "amount": 5}
    out = score_messages(bundle, [good, bad])
    assert len(out) == 1 and out.attrs["dropped"] == 1
    empty = score_messages(bundle, [bad])
    assert len(empty) == 0 and empty.attrs["dropped"] == 1


def test_hour_bucket_utc():
    ts = datetime(2026, 10, 7, 13, 45, tzinfo=timezone.utc)
    assert hour_bucket(ts) == "2026-10-07-13"


def test_make_message_fields():
    row = generate(10, 0.1, seed=6).iloc[0].to_dict()
    m = make_message(row)
    for k in ["txn_id", "event_time", "step", "type", "amount", "oldbalanceOrg", "label"]:
        assert k in m
    datetime.fromisoformat(m["event_time"])
