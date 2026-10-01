#!/usr/bin/env bash
# =============================================================================
# Stack health check
# =============================================================================
# Probes every component through the interface the pipeline actually uses, not
# just "is the container running". A container can be up while the thing inside
# it rejects connections, and that distinction is the usual cause of a
# mysteriously silent pipeline.
#
# Exit code 0 means every REQUIRED check passed. Optional checks (the producer
# and streaming metrics endpoints) are reported but never fail the run, because
# those processes are started on demand.
#
# Usage: ./scripts/health_check.sh
# =============================================================================
set -uo pipefail   # not -e: every check must run even if an earlier one fails

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE="${COMPOSE:-docker compose}"
KEYSPACE="${CASSANDRA_KEYSPACE:-fraud_detection}"
PRODUCER_METRICS_PORT="${PRODUCER_METRICS_PORT:-8001}"
STREAMING_METRICS_PORT="${STREAMING_METRICS_PORT:-8002}"

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  BOLD=''; GREEN=''; RED=''; DIM=''; RESET=''
fi

FAILURES=0

pass() { printf '  %sPASS%s  %s\n' "$GREEN" "$RESET" "$*"; }
fail() { printf '  %sFAIL%s  %s\n' "$RED" "$RESET" "$*"; FAILURES=$(( FAILURES + 1 )); }
skip() { printf '  %sSKIP%s  %s\n' "$DIM" "$RESET" "$*"; }
section() { printf '\n%s%s%s\n' "$BOLD" "$*" "$RESET"; }

# check <description> <command...>  -> required check
check() {
  local description="$1"; shift
  if "$@" >/dev/null 2>&1; then pass "$description"; else fail "$description"; fi
}

# optional_check <description> <command...> -> reported, never fatal
optional_check() {
  local description="$1"; shift
  if "$@" >/dev/null 2>&1; then pass "$description"; else skip "$description (not running)"; fi
}

in_kafka()     { $COMPOSE exec -T kafka "$@"; }
in_cassandra() { $COMPOSE exec -T cassandra "$@"; }

http_ok() { curl --silent --show-error --fail --max-time 5 "$1" >/dev/null; }

# ---------------------------------------------------------------------------
check_kafka() {
  section "Kafka"
  check "broker accepts API requests" \
    in_kafka kafka-broker-api-versions --bootstrap-server kafka:29092

  local topics
  topics="$(in_kafka kafka-topics --bootstrap-server kafka:29092 --list 2>/dev/null)"
  for topic in transactions fraud-alerts transactions-dlq; do
    if grep -qx "$topic" <<<"$topics"; then
      pass "topic exists: ${topic}"
    else
      fail "topic missing: ${topic}"
    fi
  done
}

check_cassandra() {
  section "Cassandra"
  check "native transport accepts CQL" in_cassandra cqlsh -e "DESCRIBE KEYSPACES"

  local tables
  tables="$(in_cassandra cqlsh -e \
    "SELECT table_name FROM system_schema.tables WHERE keyspace_name='${KEYSPACE}';" 2>/dev/null)"
  for table in transactions_raw fraud_alerts fraud_alerts_by_account account_alert_counters; do
    if grep -qw "$table" <<<"$tables"; then
      pass "table exists: ${KEYSPACE}.${table}"
    else
      fail "table missing: ${KEYSPACE}.${table}"
    fi
  done
}

check_observability() {
  section "Observability"
  check "Prometheus is healthy"          http_ok "http://localhost:9090/-/healthy"
  check "Grafana API is healthy"         http_ok "http://localhost:3000/api/health"
  check "Kafka exporter serves metrics"  http_ok "http://localhost:9308/metrics"
  check "Kafka UI responds"              http_ok "http://localhost:8080/actuator/health"

  # Confirms the datasource provisioning actually applied.
  if curl --silent --fail --max-time 5 \
      "http://localhost:3000/api/datasources/uid/fraud-prometheus" >/dev/null 2>&1; then
    pass "Grafana datasource 'fraud-prometheus' provisioned"
  else
    fail "Grafana datasource 'fraud-prometheus' not provisioned"
  fi
}

check_application() {
  section "Application processes (started on demand)"
  optional_check "producer metrics on :${PRODUCER_METRICS_PORT}" \
    http_ok "http://localhost:${PRODUCER_METRICS_PORT}/metrics"
  optional_check "streaming metrics on :${STREAMING_METRICS_PORT}" \
    http_ok "http://localhost:${STREAMING_METRICS_PORT}/metrics"
}

main() {
  command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 2; }
  docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable" >&2; exit 2; }

  printf '%sFraud detection stack health check%s\n' "$BOLD" "$RESET"

  check_kafka
  check_cassandra
  check_observability
  check_application

  printf '\n'
  if (( FAILURES == 0 )); then
    printf '%s%sAll required checks passed.%s\n' "$BOLD" "$GREEN" "$RESET"
    exit 0
  fi
  printf '%s%s%d required check(s) failed.%s\n' "$BOLD" "$RED" "$FAILURES" "$RESET"
  printf 'Inspect a service with: %s logs <service>\n' "$COMPOSE"
  exit 1
}

main "$@"
