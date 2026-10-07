# Linux / macOS / WSL / Git Bash.  On plain Windows PowerShell use .\run.ps1 instead.
.PHONY: stale env all up data train start health urls logs test stop down clean

env:
	@[ -f .env ] || cp .env.example .env


stale:
	@for n in $$(docker ps -a --filter name=fraud- --format '{{.Names}}'); do \
	  p=$$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' $$n); \
	  [ "$$p" = "fraud-pipeline" ] || { echo "Removing leftover container $$n"; docker rm -f $$n >/dev/null; }; \
	done
	@docker network inspect fraud-net -f '{{index .Labels "com.docker.compose.project"}}' 2>/dev/null | grep -qx fraud-pipeline || docker network rm fraud-net >/dev/null 2>&1 || true
	@for v in fraud-cassandra-data fraud-grafana-data fraud-prometheus-data fraud-kafka-data; do \
	  docker volume inspect $$v -f '{{index .Labels "com.docker.compose.project"}}' 2>/dev/null | grep -qx fraud-pipeline || docker volume rm $$v >/dev/null 2>&1 || true; \
	done

up: env stale
	docker compose up -d --remove-orphans kafka cassandra prometheus grafana kafka-ui kafka-exporter
	@echo "Waiting for health (can take ~2 min on first run)..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' fraud-cassandra 2>/dev/null)" = "healthy" ] && \
	       [ "$$(docker inspect -f '{{.State.Health.Status}}' fraud-kafka 2>/dev/null)" = "healthy" ]; do sleep 5; done
	@echo "Infrastructure healthy."

data: env
	docker compose run --rm datagen

train: env
	docker compose run --rm trainer

start: env
	@[ -f models/isolation_forest.joblib ] || (echo "No model: run make data train first"; exit 1)
	docker compose --profile app up -d --build scorer producer dashboard
	@$(MAKE) urls

all: up data train start

health:
	@docker compose --profile app ps
	@curl -s localhost:8000/metrics | grep '^fraud_transactions_processed_total ' || echo "scorer metrics not up yet"

urls:
	@echo "Dashboard  http://localhost:8501"; echo "Grafana    http://localhost:3000 (admin/admin)"; \
	 echo "Kafka UI   http://localhost:8080"; echo "Prometheus http://localhost:9090"

logs:
	docker compose --profile app logs -f --tail 50 $(S)

test: env
	docker compose run --rm tests

stop:
	docker compose --profile app stop scorer producer dashboard

down:
	docker compose --profile app down --remove-orphans

clean:
	docker compose --profile app down -v --remove-orphans
