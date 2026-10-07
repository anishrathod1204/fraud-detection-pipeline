# Real-time Fraud Detection Pipeline

Kafka → Isolation Forest scorer → Cassandra, with live Grafana metrics and a Streamlit alert feed.
Everything runs in Docker, so **Windows, macOS and Linux behave the same** (no Java, Spark or Python setup on your machine).

```
CSV ─► producer ─► Kafka ─► scorer (features + Isolation Forest) ─► Cassandra ─► Streamlit alerts
                              └──────────────► Prometheus ─► Grafana (throughput, latency, lag)
```
More detail in [docs/architecture.md](docs/architecture.md).

## Requirements
- Docker Desktop (running), ~6 GB free RAM, ~5 GB disk
- Windows: PowerShell. macOS/Linux: `make`

## Quick start

**Windows (PowerShell, from this folder)**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1 all
```
**macOS / Linux / WSL**
```bash
make all
```
`all` does: start infrastructure → generate data → train model → start producer, scorer and dashboard.
First run downloads images and takes ~5–10 minutes.

Then open:

| What | URL |
|---|---|
| Live fraud alerts | http://localhost:8501 |
| Grafana dashboard | http://localhost:3000 (admin / admin) |
| Kafka UI | http://localhost:8080 |
| Prometheus | http://localhost:9090 |

Check everything: `.\run.ps1 health` (or `make health`). Within ~1 minute the dashboard shows alerts and
Grafana shows throughput of about 40 tx/s.

