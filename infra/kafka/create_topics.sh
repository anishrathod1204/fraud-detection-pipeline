#!/usr/bin/env bash
# =============================================================================
# Kafka topic bootstrap
# =============================================================================
# Runs once, inside the cp-kafka image, after the broker reports healthy.
# Broker-side auto-creation is disabled so that partition counts and retention
# are explicit and reproducible rather than inherited from broker defaults.
#
# Idempotent: `--if-not-exists` makes re-running the stack safe, and an existing
# topic with too few partitions is expanded in place (Kafka permits increasing
# partition count, never decreasing it).
# =============================================================================
set -euo pipefail

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}"

TRANSACTIONS_TOPIC="${KAFKA_TRANSACTIONS_TOPIC:-transactions}"
ALERTS_TOPIC="${KAFKA_ALERTS_TOPIC:-fraud-alerts}"
DLQ_TOPIC="${KAFKA_DLQ_TOPIC:-transactions-dlq}"

TRANSACTIONS_PARTITIONS="${KAFKA_TRANSACTIONS_PARTITIONS:-6}"
ALERTS_PARTITIONS="${KAFKA_ALERTS_PARTITIONS:-3}"
DLQ_PARTITIONS="${KAFKA_DLQ_PARTITIONS:-1}"
REPLICATION_FACTOR="${KAFKA_REPLICATION_FACTOR:-1}"

# Retention: raw traffic is replayable for a day; alerts are the audited
# artefact and are kept a week so a restarted consumer cannot lose them.
TRANSACTIONS_RETENTION_MS="${KAFKA_TRANSACTIONS_RETENTION_MS:-86400000}"
ALERTS_RETENTION_MS="${KAFKA_ALERTS_RETENTION_MS:-604800000}"

log() { printf '[kafka-init] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

wait_for_broker() {
  local attempt=1 max_attempts=30
  until kafka-broker-api-versions --bootstrap-server "$BOOTSTRAP" >/dev/null 2>&1; do
    if (( attempt >= max_attempts )); then
      log "ERROR broker at ${BOOTSTRAP} unreachable after ${max_attempts} attempts"
      return 1
    fi
    log "broker not ready (attempt ${attempt}/${max_attempts}), retrying in 2s"
    sleep 2
    (( attempt++ ))
  done
  log "broker at ${BOOTSTRAP} is reachable"
}

# create_topic <name> <partitions> <retention_ms>
create_topic() {
  local name="$1" partitions="$2" retention_ms="$3"

  kafka-topics --bootstrap-server "$BOOTSTRAP" \
    --create --if-not-exists \
    --topic "$name" \
    --partitions "$partitions" \
    --replication-factor "$REPLICATION_FACTOR" \
    --config "retention.ms=${retention_ms}" \
    --config "compression.type=producer" >/dev/null

  # Expand an already-existing topic that was created with fewer partitions.
  local current
  current=$(kafka-topics --bootstrap-server "$BOOTSTRAP" --describe --topic "$name" \
    | awk -F'PartitionCount: ' 'NR==1 {split($2, a, "\t"); print a[1]}' | tr -d '[:space:]')

  if [[ -n "$current" && "$current" -lt "$partitions" ]]; then
    log "expanding ${name} from ${current} to ${partitions} partitions"
    kafka-topics --bootstrap-server "$BOOTSTRAP" --alter \
      --topic "$name" --partitions "$partitions" >/dev/null
  fi

  log "topic ready: ${name} (partitions=${partitions}, retention.ms=${retention_ms})"
}

main() {
  wait_for_broker
  create_topic "$TRANSACTIONS_TOPIC" "$TRANSACTIONS_PARTITIONS" "$TRANSACTIONS_RETENTION_MS"
  create_topic "$ALERTS_TOPIC" "$ALERTS_PARTITIONS" "$ALERTS_RETENTION_MS"
  create_topic "$DLQ_TOPIC" "$DLQ_PARTITIONS" "$ALERTS_RETENTION_MS"

  log "existing topics:"
  kafka-topics --bootstrap-server "$BOOTSTRAP" --list | sed 's/^/[kafka-init]   /'
  log "bootstrap complete"
}

main "$@"
