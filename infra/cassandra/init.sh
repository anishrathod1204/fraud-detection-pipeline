#!/usr/bin/env bash
# =============================================================================
# Cassandra schema initialiser
# =============================================================================
# Runs once, inside the cassandra image, after the node reports healthy. The
# compose healthcheck already waits for the native transport, but this script
# retries anyway: a node can accept CQL and still reject DDL for a few seconds
# while the schema agreement protocol settles.
#
# Idempotent - schema.cql is entirely IF NOT EXISTS.
# =============================================================================
set -euo pipefail

HOST="${CASSANDRA_HOST:-cassandra}"
PORT="${CASSANDRA_PORT:-9042}"
KEYSPACE="${CASSANDRA_KEYSPACE:-fraud_detection}"
SCHEMA_FILE="${CASSANDRA_SCHEMA_FILE:-/opt/init/schema.cql}"
MAX_ATTEMPTS="${CASSANDRA_INIT_MAX_ATTEMPTS:-40}"
SLEEP_SECONDS="${CASSANDRA_INIT_SLEEP_SECONDS:-5}"

log() { printf '[cassandra-init] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

cql() { cqlsh "$HOST" "$PORT" "$@"; }

wait_for_cql() {
  local attempt=1
  until cql -e "DESCRIBE KEYSPACES" >/dev/null 2>&1; do
    if (( attempt >= MAX_ATTEMPTS )); then
      log "ERROR ${HOST}:${PORT} did not accept CQL after ${MAX_ATTEMPTS} attempts"
      return 1
    fi
    log "waiting for CQL on ${HOST}:${PORT} (attempt ${attempt}/${MAX_ATTEMPTS})"
    sleep "$SLEEP_SECONDS"
    (( attempt++ ))
  done
  log "CQL available on ${HOST}:${PORT}"
}

apply_schema() {
  log "applying ${SCHEMA_FILE}"
  # Retry the whole file: a transient schema-agreement failure is retryable and
  # the DDL is idempotent, so re-application is harmless.
  local attempt=1
  until cql -f "$SCHEMA_FILE"; do
    if (( attempt >= 5 )); then
      log "ERROR failed to apply schema after ${attempt} attempts"
      return 1
    fi
    log "schema application failed, retrying in ${SLEEP_SECONDS}s (attempt ${attempt}/5)"
    sleep "$SLEEP_SECONDS"
    (( attempt++ ))
  done
}

verify_schema() {
  local expected="transactions_raw fraud_alerts fraud_alerts_by_account account_alert_counters"
  local existing
  existing=$(cql -e "SELECT table_name FROM system_schema.tables WHERE keyspace_name='${KEYSPACE}';")

  local missing=0
  for table in $expected; do
    if ! grep -qw "$table" <<<"$existing"; then
      log "ERROR expected table ${KEYSPACE}.${table} is missing"
      missing=1
    else
      log "verified ${KEYSPACE}.${table}"
    fi
  done
  return "$missing"
}

main() {
  wait_for_cql
  apply_schema
  verify_schema
  log "schema initialisation complete"
}

main "$@"
