"""Prometheus metrics exported by the producer.

Defined in one module so that the metric names, labels and help text are
reviewable together, rather than scattered across the code that happens to
increment them. Every name is prefixed ``fraud_producer_`` and every counter
ends ``_total``, per Prometheus naming conventions, so the Grafana queries in
``dashboard/fraud_dashboard.json`` can rely on the pattern.

Counters versus gauges
----------------------
Throughput is exported as a *counter* of messages, not a gauge of
messages-per-second. Prometheus computes rates from counters with ``rate()``,
and a counter survives a missed scrape without losing the interval's activity,
whereas a pre-averaged gauge silently drops it. The one gauge here,
:data:`TARGET_TPS`, is a configured setting rather than a measurement.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import Counter, Gauge, Histogram

from common.metrics import LATENCY_BUCKETS

__all__ = [
    "BYTES_PUBLISHED",
    "CONNECT_ATTEMPTS",
    "FRAUD_RECORDS_READ",
    "KAFKA_CONNECTED",
    "MESSAGES_PUBLISHED",
    "PUBLISH_FAILURES",
    "PUBLISH_LATENCY",
    "RECORDS_READ",
    "TARGET_TPS",
    "THROTTLE_SLEEP",
]

#: Rows pulled off the CSV, before any filtering. Diverges from
#: :data:`MESSAGES_PUBLISHED` when records are dropped as malformed.
RECORDS_READ: Final[Counter] = Counter(
    "fraud_producer_records_read_total",
    "PaySim rows read from the source dataset.",
)

#: Ground-truth fraud rows emitted. Not a detection metric - it is the
#: denominator the dashboard needs to show detected-versus-actual, which is the
#: only way to see false negatives on a live chart.
FRAUD_RECORDS_READ: Final[Counter] = Counter(
    "fraud_producer_fraud_records_total",
    "Rows read whose ground-truth isFraud label is 1.",
)

#: Successful publishes, acknowledged by the broker.
MESSAGES_PUBLISHED: Final[Counter] = Counter(
    "fraud_producer_messages_published_total",
    "Transactions acknowledged by Kafka.",
    labelnames=("topic",),
)

#: Serialised payload bytes acknowledged, for bandwidth panels. Measured
#: pre-compression, since that is the figure that bounds producer-side cost.
BYTES_PUBLISHED: Final[Counter] = Counter(
    "fraud_producer_bytes_published_total",
    "Uncompressed payload bytes acknowledged by Kafka.",
    labelnames=("topic",),
)

#: Failed publishes, labelled by exception class. The label is deliberately the
#: exception type and not the message: messages embed broker addresses and
#: offsets, which would give this counter unbounded cardinality.
PUBLISH_FAILURES: Final[Counter] = Counter(
    "fraud_producer_publish_failures_total",
    "Transactions that could not be published, by error class.",
    labelnames=("topic", "reason"),
)

#: Time from handing a record to the client until the broker acknowledges it.
PUBLISH_LATENCY: Final[Histogram] = Histogram(
    "fraud_producer_publish_latency_seconds",
    "Send-to-acknowledgement latency for a batch flush.",
    buckets=LATENCY_BUCKETS,
)

#: Configured target rate, 0 when unthrottled. Exported so a dashboard can draw
#: the ceiling alongside the achieved rate and make saturation obvious.
TARGET_TPS: Final[Gauge] = Gauge(
    "fraud_producer_target_tps",
    "Configured target transactions per second; 0 means unthrottled.",
)

#: Cumulative time spent sleeping to hold the target rate. Near-zero while the
#: target rate is being met means the producer is at its ceiling, which is the
#: signal that a load test has found the real limit.
THROTTLE_SLEEP: Final[Counter] = Counter(
    "fraud_producer_throttle_sleep_seconds_total",
    "Cumulative seconds slept by the rate limiter.",
)

#: Broker connection attempts, including retries.
CONNECT_ATTEMPTS: Final[Counter] = Counter(
    "fraud_producer_connect_attempts_total",
    "Attempts to establish a Kafka producer connection.",
)

#: 1 while a broker connection is established, 0 otherwise.
KAFKA_CONNECTED: Final[Gauge] = Gauge(
    "fraud_producer_kafka_connected",
    "1 when the producer holds a live Kafka connection, 0 otherwise.",
)
