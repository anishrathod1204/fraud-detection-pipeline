# Real-Time Financial Fraud Detection Pipeline

A streaming fraud detection system that ingests mobile-money transactions from
Kafka, scores them in near real time with an unsupervised anomaly detection
model running inside Spark Structured Streaming, persists results to Cassandra,
and surfaces live fraud signals through Grafana and a Streamlit alert feed.

Built on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) synthetic
mobile money dataset (~6.3M transactions, 0.13% fraud).

## Status

| Phase | Component | State |
| ----- | --------- | ----- |
| 1 | Infrastructure (Docker Compose) | done |
| 2 | Kafka producer / stream simulation | planned |
| 3 | Model training and evaluation | planned |
| 4 | Spark Structured Streaming scoring job | planned |
| 5 | Grafana + Streamlit dashboards | planned |
| 6 | Architecture docs, tuning, load testing | planned |

## Getting started

### Prerequisites

| Requirement | Version | Notes |
| ----------- | ------- | ----- |
| Docker Engine | 24+ | with the Compose v2 plugin (`docker compose`, not `docker-compose`) |
| Python | 3.10+ | for the producer, training and streaming jobs |
| Java | 11 or 17 | required by Spark; 17 is the sweet spot for Spark 3.5 |
| RAM | 8 GB free | Cassandra takes 1 GB heap, Kafka ~1 GB, Spark 2 GB |
| Disk | 5 GB free | Kafka log segments plus Cassandra SSTables |

### Bring up the stack

```bash
make env     # creates .env from .env.example
make up      # starts every service and blocks until all are healthy
```

`make up` is not a bare `docker compose up -d`. It verifies the Docker daemon is
reachable, validates the compose file, then polls until Kafka, Cassandra,
Prometheus and Grafana all report `healthy` and both one-shot initialisers
(`kafka-init`, `cassandra-init`) have exited 0. If anything fails it tells you
which service and the exact command to inspect it.

### Verify

```bash
make health
```

This probes each component through the interface the pipeline really uses -
broker API versions, topic existence, CQL plus table existence, the Prometheus
and Grafana health endpoints, and whether the Grafana datasource was actually
provisioned. A container can be `Up` while the service inside it refuses
connections, and that gap is the usual cause of a silently idle pipeline.

Expected output on a clean start is `All required checks passed.` The two
application-process checks report `SKIP` until you start the producer and the
streaming job.

### Service endpoints

| Service | URL / address | Purpose |
| ------- | ------------- | ------- |
| Grafana | http://localhost:3000 | dashboards; anonymous viewer, `admin`/`admin` to edit |
| Prometheus | http://localhost:9090 | metric store, target status at `/targets` |
| Kafka UI | http://localhost:8080 | browse topics, payloads, consumer lag |
| Kafka exporter | http://localhost:9308/metrics | raw broker and lag metrics |
| Kafka broker | `localhost:9092` | host clients |
| Cassandra | `localhost:9042` | CQL native transport |

`make urls` prints this list. Note the dual Kafka listener: host processes use
`localhost:9092`, containers inside the stack use `kafka:29092`.

### Shut down

```bash
make down     # stop containers, keep Kafka offsets and Cassandra data
make clean    # stop and delete volumes, also clears Spark checkpoints
```

`make clean` removes checkpoints deliberately: they reference Kafka offsets that
no longer exist once topic data is wiped, and a stale checkpoint makes the next
streaming run fail to start.

## Stack

| Concern | Technology | Rationale |
| ------- | ---------- | --------- |
| Ingestion | Apache Kafka (KRaft) | Durable, replayable, partitioned log |
| Processing | Spark Structured Streaming | Micro-batch scoring with exactly-once sinks |
| Model | Isolation Forest (+ autoencoder baseline) | Unsupervised, no reliance on scarce labels |
| Storage | Apache Cassandra | High write throughput, time-series friendly |
| Metrics | Prometheus | Pull-based scraping of component exporters |
| Visualisation | Grafana, Streamlit | Operational dashboards and demo feed |
| Orchestration | Docker Compose | One-command local environment |

## Repository layout

```
fraud-detection-pipeline/
├── docker-compose.yml     # full local stack
├── Makefile               # one-command lifecycle targets
├── common/                # shared config, logging, schema, features
├── producer/              # PaySim -> Kafka stream simulator
├── training/              # feature engineering + model training
├── streaming/             # Spark Structured Streaming scoring job
├── dashboard/             # Grafana dashboard JSON + Streamlit app
├── infra/                 # Cassandra schema, Prometheus, Grafana provisioning
├── scripts/               # dataset generation, load testing, helpers
├── tests/                 # unit and integration tests
└── docs/                  # architecture and evaluation reports
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) - system diagram, data flow,
  and design tradeoffs.
- `docs/model_evaluation.md` - generated by `make train`.
- `docs/load_test_results.md` - generated by `make load-test`.

## Licence

MIT - see [`LICENSE`](LICENSE).
