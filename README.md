# Real-time Fraud Detection Pipeline

Kafka → Isolation Forest scorer → Cassandra, with live Grafana metrics and a Streamlit alert feed.
Everything runs in Docker, so **Windows, macOS and Linux behave the same** (no Java, Spark or Python setup on your machine).

```
CSV ─► producer ─► Kafka ─► scorer (features + Isolation Forest) ─► Cassandra ─► Streamlit alerts
                              └──────────────► Prometheus ─► Grafana (throughput, latency, lag)
```
More detail in [docs/architecture.md](docs/architecture.md).

