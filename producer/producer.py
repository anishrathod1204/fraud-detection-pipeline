"""Replay a PaySim-style CSV into Kafka at a controlled rate."""
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402

log = get_logger("producer")


def make_message(row: dict) -> dict:
    """Build the JSON event. The ground-truth label is carried only for later evaluation."""
    return {
        "txn_id": str(uuid.uuid4()),
        "event_time": datetime.now(timezone.utc).isoformat(),
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "nameOrig": row["nameOrig"],
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "nameDest": row["nameDest"],
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "label": int(row.get("isFraud", 0)),
    }


def main() -> None:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable

    from common.kafka_utils import ensure_topic

    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=config.PRODUCER_RATE, help="messages per second")
    ap.add_argument("--limit", type=int, default=0, help="stop after N messages (0 = run forever)")
    ap.add_argument("--data", default=config.DATA_FILE)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        log.error("Data file %s not found. Run the data step first.", a.data)
        sys.exit(1)

    ensure_topic(config.KAFKA_BOOTSTRAP, config.KAFKA_TOPIC)
    producer = None
    while producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=config.KAFKA_BOOTSTRAP, acks="all", linger_ms=20, retries=5,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
        except NoBrokersAvailable:
            log.warning("Waiting for Kafka...")
            time.sleep(3)

    df = pd.read_csv(a.data)
    log.info("Streaming %s rows from %s at %d msg/s", f"{len(df):,}", a.data, a.rate)
    sent, started, last_report = 0, time.time(), time.time()
    while True:
        for row in df.sample(frac=1.0).to_dict("records"):
            msg = make_message(row)
            producer.send(config.KAFKA_TOPIC, key=msg["nameOrig"], value=msg)
            sent += 1
            if a.limit and sent >= a.limit:
                producer.flush()
                log.info("Sent %d messages, done.", sent)
                return
            # pace: sleep until the next send is due
            due = started + sent / max(a.rate, 1)
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            if time.time() - last_report > 10:
                log.info("Sent %d messages (%.1f msg/s)", sent, sent / (time.time() - started))
                last_report = time.time()


if __name__ == "__main__":
    main()
