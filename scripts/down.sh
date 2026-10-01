#!/usr/bin/env bash
# =============================================================================
# Tear down the fraud detection stack
# =============================================================================
# Default behaviour stops and removes containers but KEEPS volumes, so Kafka
# offsets and Cassandra tables survive a restart. Pass --volumes to wipe state.
#
# Usage: ./scripts/down.sh [--volumes] [extra docker compose args...]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE="${COMPOSE:-docker compose}"
REMOVE_VOLUMES=false

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=''; YELLOW=''; RESET=''
fi

log()  { printf '%s[down]%s %s\n' "$BOLD" "$RESET" "$*"; }
warn() { printf '%s[down]%s %s%s%s\n' "$BOLD" "$RESET" "$YELLOW" "$*" "$RESET"; }

parse_args() {
  EXTRA_ARGS=()
  while (( $# )); do
    case "$1" in
      --volumes|-v) REMOVE_VOLUMES=true; shift ;;
      *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
  done
}

main() {
  parse_args "$@"

  if ! docker info >/dev/null 2>&1; then
    warn "Docker daemon not reachable - nothing to stop"
    exit 0
  fi

  local args=(down --remove-orphans)

  if [[ "$REMOVE_VOLUMES" == true ]]; then
    warn "removing volumes: Kafka topic data and Cassandra tables will be lost"
    args+=(--volumes)
  fi

  (( ${#EXTRA_ARGS[@]} )) && args+=("${EXTRA_ARGS[@]}")

  log "stopping services"
  $COMPOSE "${args[@]}"

  if [[ "$REMOVE_VOLUMES" == true ]]; then
    # Spark checkpoints reference Kafka offsets that no longer exist once the
    # topic data is gone; leaving them behind makes the next run fail to start.
    if [[ -d checkpoints ]]; then
      rm -rf checkpoints
      log "removed stale Spark checkpoints"
    fi
  fi

  log "done"
}

main "$@"
