#!/usr/bin/env bash
# =============================================================================
# Bring up the fraud detection stack
# =============================================================================
# Wraps `docker compose up -d` with the parts that are easy to forget:
#   - verifies Docker is actually running before producing a confusing error
#   - creates .env from the template on first run
#   - blocks until every healthchecked service is healthy, instead of returning
#     the moment containers are *created* (which is when `up -d` returns)
#   - confirms the one-shot initialisers exited 0
#
# Usage: ./scripts/up.sh [--no-wait] [extra docker compose args...]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE="${COMPOSE:-docker compose}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-300}"
WAIT=true

# Services that declare a healthcheck and must reach "healthy".
HEALTHCHECKED_SERVICES=(kafka cassandra prometheus grafana)
# One-shot services that must reach exit code 0.
INIT_SERVICES=(kafka-init cassandra-init)

# Colour only when attached to a terminal, so piped output stays clean.
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=''; GREEN=''; RED=''; YELLOW=''; RESET=''
fi

log()  { printf '%s[up]%s %s\n' "$BOLD" "$RESET" "$*"; }
ok()   { printf '%s[up]%s %s%s%s\n' "$BOLD" "$RESET" "$GREEN" "$*" "$RESET"; }
warn() { printf '%s[up]%s %s%s%s\n' "$BOLD" "$RESET" "$YELLOW" "$*" "$RESET"; }
die()  { printf '%s[up]%s %s%s%s\n' "$BOLD" "$RESET" "$RED" "$*" "$RESET" >&2; exit 1; }

parse_args() {
  EXTRA_ARGS=()
  while (( $# )); do
    case "$1" in
      --no-wait) WAIT=false; shift ;;
      *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
  done
}

preflight() {
  command -v docker >/dev/null 2>&1 || die "docker is not on PATH"
  docker info >/dev/null 2>&1 || die "the Docker daemon is not reachable - is Docker running?"

  if [[ ! -f .env ]]; then
    cp .env.example .env
    log "created .env from .env.example"
  fi

  # Fails fast on a malformed compose file or an unresolved variable.
  $COMPOSE config --quiet || die "docker-compose.yml is invalid"
  ok "preflight checks passed"
}

container_health() {
  # Prints one of: healthy | unhealthy | starting | none | missing
  local service="$1" cid
  cid="$($COMPOSE ps -q "$service" 2>/dev/null || true)"
  [[ -z "$cid" ]] && { echo missing; return; }
  docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
    "$cid" 2>/dev/null || echo missing
}

container_exit_code() {
  local service="$1" cid
  cid="$($COMPOSE ps -aq "$service" 2>/dev/null | head -n1 || true)"
  [[ -z "$cid" ]] && { echo missing; return; }
  docker inspect --format '{{.State.Status}}:{{.State.ExitCode}}' "$cid" 2>/dev/null || echo missing
}

wait_for_health() {
  local deadline=$(( SECONDS + WAIT_TIMEOUT_SECONDS ))

  for service in "${HEALTHCHECKED_SERVICES[@]}"; do
    log "waiting for ${service}"
    while :; do
      local status
      status="$(container_health "$service")"
      case "$status" in
        healthy)   ok "${service} healthy"; break ;;
        unhealthy) die "${service} reported unhealthy - inspect with: $COMPOSE logs ${service}" ;;
        missing)   die "${service} container not found" ;;
      esac
      (( SECONDS < deadline )) || die "timed out after ${WAIT_TIMEOUT_SECONDS}s waiting for ${service}"
      sleep 3
    done
  done

  for service in "${INIT_SERVICES[@]}"; do
    log "waiting for ${service} to complete"
    while :; do
      local state
      state="$(container_exit_code "$service")"
      case "$state" in
        exited:0)  ok "${service} completed"; break ;;
        exited:*)  die "${service} failed (${state}) - inspect with: $COMPOSE logs ${service}" ;;
        missing)   die "${service} container not found" ;;
      esac
      (( SECONDS < deadline )) || die "timed out after ${WAIT_TIMEOUT_SECONDS}s waiting for ${service}"
      sleep 3
    done
  done
}

print_endpoints() {
  cat <<'EOF'

  Stack is up.

    Grafana      http://localhost:3000   dashboard "Fraud Detection - Real-Time Overview"
    Prometheus   http://localhost:9090
    Kafka UI     http://localhost:8080
    Kafka        localhost:9092          (host clients)
    Cassandra    localhost:9042

  Next:
    make dataset     generate a PaySim-schema CSV
    make train       train the models and write docs/model_evaluation.md
    make stream      start the Spark scoring job
    make produce     start streaming transactions

EOF
}

main() {
  parse_args "$@"
  preflight

  log "starting services"
  if (( ${#EXTRA_ARGS[@]} )); then
    $COMPOSE up -d --remove-orphans "${EXTRA_ARGS[@]}"
  else
    $COMPOSE up -d --remove-orphans
  fi

  if [[ "$WAIT" == true ]]; then
    wait_for_health
  else
    warn "--no-wait given, skipping health gating"
  fi

  $COMPOSE ps
  print_endpoints
}

main "$@"
