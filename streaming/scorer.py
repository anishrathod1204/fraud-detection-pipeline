"""Streaming scorer: Kafka -> Isolation Forest -> Cassandra, with Prometheus metrics.

Micro-batches messages, scores them in one vectorised call, writes results with idempotent
upserts keyed by txn_id, then commits Kafka offsets (at-least-once delivery + idempotent sink
= effectively exactly-once results).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config  # noqa: E402
from common.cassandra_store import COLUMNS, hour_bucket  # noqa: E402
from common.features import build_features  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402
from common.schema import REQUIRED_FOR_SCORING  # noqa: E402

log = get_logger("scorer")


def load_model(path: str) -> dict:
    return joblib.load(path)


def score_messages(bundle: dict, messages: list) -> pd.DataFrame:
    """Score a list of transaction dicts. Returns a DataFrame with anomaly_score and predicted_fraud.

    Messages missing required fields are dropped (and reported via df.attrs['dropped']).
    """
    valid = [m for m in messages if all(m.get(c) is not None for c in REQUIRED_FOR_SCORING)]
    dropped = len(messages) - len(valid)
    if not valid:
        out = pd.DataFrame()
        out.attrs["dropped"] = dropped
        return out
    df = pd.DataFrame(valid)
    X = build_features(df)
    df["anomaly_score"] = -bundle["model"].score_samples(X.to_numpy())
    df["predicted_fraud"] = df["anomaly_score"] >= bundle["threshold"]
    df.attrs["dropped"] = dropped
    return df


def to_rows(df: pd.DataFrame):
    """Convert a scored DataFrame into Cassandra tuples (all rows, alert rows)."""
    now = datetime.now(timezone.utc)
    rows, alerts = [], []
    for r in df.to_dict("records"):
        try:
            ts = datetime.fromisoformat(str(r.get("event_time")))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            ts = now
        row = (
            hour_bucket(ts), ts, str(r.get("txn_id") or ""), int(r["step"]), str(r["type"]), float(r["amount"]),
            str(r.get("nameOrig") or ""), str(r.get("nameDest") or ""),
            float(r["oldbalanceOrg"]), float(r["newbalanceOrig"]),
            float(r["oldbalanceDest"]), float(r["newbalanceDest"]),
            float(r["anomaly_score"]), bool(r["predicted_fraud"]),
            int(r.get("label") or 0), now,
        )
        assert len(row) == len(COLUMNS)
        rows.append(row)
        if row[13]:
            alerts.append(row)
    return rows, alerts


def main() -> None:
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    from common import cassandra_store
    from common.kafka_utils import ensure_topic

    if not os.path.exists(config.MODEL_PATH):
        log.error("Model not found at %s. Train it first:  .\\run.ps1 train", config.MODEL_PATH)
        sys.exit(1)
    bundle = load_model(config.MODEL_PATH)
    log.info("Model loaded, alert threshold %.4f", bundle["threshold"])

    processed = Counter("fraud_transactions_processed_total", "Transactions scored")
    alerts_c = Counter("fraud_alerts_total", "Transactions flagged as fraud")
    flagged_amount = Counter("fraud_flagged_amount_total", "Sum of flagged transaction amounts")
    errors = Counter("fraud_scoring_errors_total", "Malformed messages dropped")
    batch_hist = Histogram("fraud_batch_seconds", "Time to score + persist one batch",
                           buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5))
    e2e_hist = Histogram("fraud_end_to_end_latency_seconds", "Event time to persisted",
                         buckets=(.1, .25, .5, 1, 2, 5, 10, 30, 60))
    score_hist = Histogram("fraud_anomaly_score", "Anomaly score distribution",
                           buckets=(.3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .9, 1.0))
    batch_size = Gauge("fraud_last_batch_size", "Messages in the last batch")
    Gauge("fraud_model_threshold", "Alert threshold").set(bundle["threshold"])
    start_http_server(config.METRICS_PORT)
    log.info("Prometheus metrics on :%d", config.METRICS_PORT)

    _cluster, session = cassandra_store.connect()
    writer = cassandra_store.Writer(session)
    ensure_topic(config.KAFKA_BOOTSTRAP, config.KAFKA_TOPIC)

    consumer = None
    while consumer is None:
        try:
            consumer = KafkaConsumer(
                config.KAFKA_TOPIC, bootstrap_servers=config.KAFKA_BOOTSTRAP, group_id=config.KAFKA_GROUP,
                enable_auto_commit=False, auto_offset_reset="earliest",
                value_deserializer=lambda b: b, max_poll_records=500,
            )
        except NoBrokersAvailable:
            log.warning("Waiting for Kafka...")
            time.sleep(3)
    log.info("Consuming topic %s", config.KAFKA_TOPIC)

    while True:
        polled = consumer.poll(timeout_ms=1000, max_records=500)
        raw = [rec.value for recs in polled.values() for rec in recs]
        if not raw:
            continue
        t0 = time.time()
        messages = []
        for b in raw:
            try:
                messages.append(json.loads(b))
            except (ValueError, UnicodeDecodeError):
                errors.inc()
        df = score_messages(bundle, messages)
        if df.attrs.get("dropped"):
            errors.inc(df.attrs["dropped"])
        if len(df):
            rows, alerts = to_rows(df)
            for attempt in range(1, 6):
                try:
                    writer.write(rows, alerts)
                    break
                except Exception as exc:
                    log.warning("Cassandra write failed (%d/5): %s", attempt, str(exc)[:150])
                    time.sleep(2 * attempt)
            else:
                raise RuntimeError("Cassandra unavailable; exiting without committing offsets")
            done = datetime.now(timezone.utc)
            for r in rows:
                e2e_hist.observe(max((done - r[1]).total_seconds(), 0))
                score_hist.observe(r[12])
            processed.inc(len(rows))
            alerts_c.inc(len(alerts))
            flagged_amount.inc(sum(a[5] for a in alerts))
        consumer.commit()
        batch_size.set(len(raw))
        batch_hist.observe(time.time() - t0)


if __name__ == "__main__":
    main()
