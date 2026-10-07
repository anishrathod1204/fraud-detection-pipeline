# Architecture

```
 PaySim-style CSV ──► producer ──► Kafka (topic: transactions, 3 partitions)
                                      │
                                      ▼
                                   scorer  (micro-batch ≤500 msgs)
                       features ─► Isolation Forest ─► anomaly score ≥ threshold ?
                                      │                        │
                          upsert (idempotent)           Prometheus metrics :8000
                                      ▼                        │
                              Cassandra                        ▼
                      scored_transactions                 Prometheus ─► Grafana
                      fraud_alerts ─► Streamlit dashboard        ▲
                                                       kafka-exporter (consumer lag)
```

## Design decisions
- **Unsupervised model.** Real fraud labels are rare and delayed. Isolation Forest never sees labels; they only
  pick the alert threshold and produce the evaluation report.
- **One feature function** (`common/features.py`) used by training and the live scorer, so there is no train/serve skew.
- **At-least-once + idempotent sink.** The scorer commits Kafka offsets only after Cassandra accepted the batch.
  Rows are keyed by `txn_id`, so a replay overwrites rather than duplicates.
- **Time-bucketed partitions.** Cassandra partitions are `hour_bucket`, clustering by `event_time DESC`, which keeps
  partitions bounded and makes "latest alerts" a cheap query.
- **Backpressure visible.** Consumer lag is exported by kafka-exporter and charted in Grafana.
- **Python scorer instead of Spark.** Spark Structured Streaming adds Java, Maven downloads and Hadoop shims (painful
  on Windows) for a few hundred events/second that a vectorised Python consumer handles easily. The scoring logic
  (`score_messages`) is a pure function and could be wrapped in a Spark `foreachBatch` if you need to scale out.
