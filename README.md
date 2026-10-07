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

## Commands

| Windows | Linux/macOS | Does |
|---|---|---|
| `.\run.ps1 up` | `make up` | Kafka, Cassandra, Prometheus, Grafana, Kafka UI |
| `.\run.ps1 data` | `make data` | Generate PaySim-style data (skipped if `data/paysim.csv` exists) |
| `.\run.ps1 train` | `make train` | Train model, write `docs/model_evaluation.md` |
| `.\run.ps1 start` | `make start` | Producer + scorer + dashboard |
| `.\run.ps1 logs scorer` | `make logs S=scorer` | Tail logs |
| `.\run.ps1 test` | `make test` | Unit tests |
| `.\run.ps1 stop` | `make stop` | Stop app only |
| `.\run.ps1 down` | `make down` | Stop all, keep data |
| `.\run.ps1 clean` | `make clean` | Stop all, delete data |

## Using the real PaySim dataset
Download `PS_20174392719_1491204439457_log.csv` from Kaggle ("PaySim synthetic financial datasets"),
save it as `data/paysim.csv`, then `.\run.ps1 train`. Real fraud is only ~0.13% of rows, so expect
alerts to be much rarer and the precision lower than on the generated data.

## Tuning
Edit `.env`: `PRODUCER_RATE` (tx/s). After changing it: `docker compose --profile app up -d producer`.
Code is bind-mounted into the containers: edit, then `docker compose --profile app restart scorer`.

## Project layout
```
common/      config, feature engineering (shared by training + serving), Cassandra + Kafka helpers
scripts/     synthetic data generator
training/    Isolation Forest training + evaluation report
producer/    CSV → Kafka replayer
streaming/   Kafka → score → Cassandra, Prometheus metrics
dashboard/   Streamlit alert feed
infra/       Prometheus + Grafana provisioning and dashboard
tests/       unit tests (data, features, training, scoring)
```

