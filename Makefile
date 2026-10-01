# =============================================================================
# Real-Time Fraud Detection Pipeline - developer entry points
# =============================================================================
# `make help` lists every target. Targets are grouped: stack lifecycle, data and
# model, pipeline runtime, quality, and observability.
#
# Every target is a thin wrapper around a documented command, so nothing here is
# load-bearing magic - you can always run the underlying command directly.
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Overridable: `make up COMPOSE="docker-compose"` for the legacy v1 binary.
COMPOSE ?= docker compose
PYTHON  ?= python3

# Loaded by the application processes themselves (common/config.py); exported
# here so `make` targets honour local overrides too.
ENV_FILE ?= .env

# Tunable without editing this file: `make produce TPS=500`
TPS            ?= 50
ROWS           ?= 2000000
LOAD_TEST_TPS  ?= 5000
LOAD_TEST_SECS ?= 120

.PHONY: help up down restart ps logs health clean nuke \
        env dataset train produce produce-fraud stream dashboard \
        test lint format typecheck check load-test urls

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help: ## Show this help
	@echo ""
	@echo "Real-Time Fraud Detection Pipeline"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Typical first run:"
	@echo "    make env && make up && make dataset && make train && make produce"
	@echo ""

# -----------------------------------------------------------------------------
# Stack lifecycle
# -----------------------------------------------------------------------------
env: ## Create .env from .env.example if absent
	@if [[ -f $(ENV_FILE) ]]; then \
		echo "$(ENV_FILE) already exists, leaving it alone"; \
	else \
		cp .env.example $(ENV_FILE) && echo "created $(ENV_FILE) from .env.example"; \
	fi

up: ## Start the full stack and wait for health
	@./scripts/up.sh

down: ## Stop the stack, keep volumes
	@./scripts/down.sh

restart: down up ## Stop then start the stack

ps: ## Show container status
	@$(COMPOSE) ps

logs: ## Tail logs from all services (S=kafka to filter)
	@$(COMPOSE) logs -f --tail=100 $(S)

health: ## Probe every service endpoint and report
	@./scripts/health_check.sh

urls: ## Print the stack's web endpoints
	@echo "Grafana     http://localhost:3000  (anonymous viewer, admin/admin to edit)"
	@echo "Prometheus  http://localhost:9090"
	@echo "Kafka UI    http://localhost:8080"
	@echo "Streamlit   http://localhost:8501  (make dashboard)"

clean: ## Stop the stack and delete volumes (destroys topic and table data)
	@./scripts/down.sh --volumes

nuke: clean ## clean, plus local checkpoints and model artifacts
	@rm -rf checkpoints/ training/artifacts/*.joblib training/artifacts/*.pt \
		training/artifacts/*.json
	@echo "removed checkpoints and model artifacts"

# -----------------------------------------------------------------------------
# Data and model
# -----------------------------------------------------------------------------
dataset: ## Generate a PaySim-schema dataset (ROWS=2000000)
	@$(PYTHON) -m scripts.generate_dataset --rows $(ROWS)

train: ## Train Isolation Forest + autoencoder, write artifacts and eval report
	@$(PYTHON) -m training.train_model

# -----------------------------------------------------------------------------
# Pipeline runtime
# -----------------------------------------------------------------------------
produce: ## Stream transactions to Kafka (TPS=50)
	@$(PYTHON) -m producer.simulate_stream --tps $(TPS)

produce-fraud: ## Stream only fraud-labelled rows, for demos
	@$(PYTHON) -m producer.simulate_stream --tps $(TPS) --replay-fraud-only

stream: ## Run the Spark Structured Streaming scoring job
	@./scripts/run_streaming.sh

dashboard: ## Run the Streamlit live alert feed
	@$(PYTHON) -m streamlit run dashboard/live_feed.py

# -----------------------------------------------------------------------------
# Quality
# -----------------------------------------------------------------------------
test: ## Run the test suite
	@$(PYTHON) -m pytest -q

lint: ## Lint with ruff
	@$(PYTHON) -m ruff check .

format: ## Auto-format with ruff
	@$(PYTHON) -m ruff format .
	@$(PYTHON) -m ruff check --fix .

typecheck: ## Static type check with mypy
	@$(PYTHON) -m mypy common producer training streaming dashboard scripts

check: lint typecheck test ## Lint, typecheck and test

# -----------------------------------------------------------------------------
# Observability
# -----------------------------------------------------------------------------
load-test: ## Saturate the pipeline and record results (LOAD_TEST_TPS=5000)
	@./scripts/load_test.sh --tps $(LOAD_TEST_TPS) --duration $(LOAD_TEST_SECS)
