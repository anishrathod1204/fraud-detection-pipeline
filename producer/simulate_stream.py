"""Stream the PaySim dataset into Kafka as a live transaction feed.

This is the entrypoint that turns a static CSV into the event stream the rest of
the pipeline consumes. It is deliberately the *only* place that knows about the
CLI; the publisher, the rate limiter and the CSV reader are each independently
usable modules, so this file stays a thin composition layer.

How a run proceeds
------------------
1. Read configuration from the environment (``common.config``).
2. Open the Prometheus exporter, so the producer is observable from the first
   record even if it never fully connects to the broker.
3. Connect to Kafka, retrying with backoff until the broker is up.
4. Iterate the CSV in chunks, publish each row keyed by account, and pace the
   loop to the target rate.
5. Emit a periodic stats line and, on exit, a final summary.

Modes
-----
* Default: stream the whole file once, in order.
* ``--loop``: restart at EOF, for a demo that runs until interrupted.
* ``--replay-fraud-only``: stream only ``isFraud == 1`` rows. At the dataset's
  0.13% prevalence a live demo would otherwise show roughly one fraud every 770
  transactions; replaying only fraud makes the detection path easy to watch.
  This is a *demo* mode and the metrics still count honest ground truth, so the
  precision figures computed from a replay run are meaningless by construction.

Graceful shutdown
-----------------
SIGINT/SIGTERM set a flag checked between records. The in-flight batch is
flushed and the connection closed as the context manager unwinds, so Ctrl-C
does not drop acknowledged work or leave the metrics endpoint half-written.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Final, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.config import AppConfig, ConfigError, get_config
from common.dataset import DatasetError, iter_transactions
from common.logging_config import configure_logging, get_logger
from common.metrics import start_metrics_server
from producer import metrics
from producer.kafka_publisher import PublisherError, TransactionPublisher
from producer.rate_limiter import RateLimiter

__all__ = ["build_parser", "main", "run"]

_LOGGER = get_logger(__name__)

#: Exit code returned when the producer fails for an operational reason (broker
#: unreachable, dataset missing). Distinct from 0 (success) and from the
#: interpreter's own codes so a wrapper script can tell them apart.
_EXIT_FAILURE: Final[int] = 1


class _Shutdown:
    """Cooperative shutdown flag set by signal handlers.

    A plain object rather than a module global so that :func:`run` and its tests
    do not share mutable state across calls. Signal handlers may only touch
    simple, async-signal-safe state, so this stores a single boolean and nothing
    else.
    """

    __slots__ = ("requested",)

    def __init__(self) -> None:
        """Initialise with shutdown not yet requested."""
        self.requested = False

    def request(self) -> None:
        """Record that a shutdown signal was received."""
        self.requested = True


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Every flag has an environment-variable fallback through
    :func:`common.config.get_config`; the flags exist for one-off experiments
    (``--tps 5000`` for a load test) without editing ``.env``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="simulate-stream",
        description="Stream PaySim transactions to Kafka as a paced live feed.",
    )
    parser.add_argument(
        "--tps",
        type=int,
        default=None,
        help="Target transactions per second. 0 disables throttling. "
        "Overrides PRODUCER_TPS.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="PaySim CSV to stream. Overrides PAYSIM_CSV_PATH.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after this many records. Default: stream the whole file once.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart at end of file and keep streaming until interrupted.",
    )
    parser.add_argument(
        "--replay-fraud-only",
        action="store_true",
        help="Stream only isFraud==1 rows (demo mode; metrics become unrepresentative).",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=None,
        help="Seconds between progress lines. Overrides PRODUCER_STATS_INTERVAL_SECONDS.",
    )
    return parser


def _install_signal_handlers(shutdown: _Shutdown) -> None:
    """Route SIGINT/SIGTERM to the cooperative shutdown flag.

    Args:
        shutdown: Flag object to set when a signal arrives.
    """

    def _handler(signum: int, _frame: FrameType | None) -> None:
        # Keep this trivial: it runs in signal context.
        _LOGGER.warning(
            "shutdown signal received, finishing current batch",
            extra={"signal": signal.Signals(signum).name},
        )
        shutdown.request()

    signal.signal(signal.SIGINT, _handler)
    # SIGTERM is absent on some platforms; guard so import-time use is safe.
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def run(
    config: AppConfig,
    *,
    tps: int | None = None,
    csv_path: str | None = None,
    max_records: int | None = None,
    loop: bool = False,
    replay_fraud_only: bool = False,
    stats_interval: float | None = None,
) -> int:
    """Execute a streaming run and return a process exit code.

    Separated from :func:`main` so it can be driven from tests and from a
    notebook with an explicit configuration object rather than the process
    environment.

    Args:
        config: The resolved application configuration.
        tps: Target rate override; ``None`` uses the configured value.
        csv_path: Dataset path override; ``None`` uses the configured value.
        max_records: Optional cap on records emitted this run.
        loop: Restart at end of file.
        replay_fraud_only: Emit only fraud rows.
        stats_interval: Progress-line interval override.

    Returns:
        ``0`` when the stream completed, ``_EXIT_FAILURE`` on an operational
        error.
    """
    producer_settings = config.producer

    target_tps = producer_settings.tps if tps is None else max(0, tps)
    # Coerce to Path even though from_env() already resolves it: run() is a
    # public entrypoint and a caller passing a raw string (a notebook, a test)
    # should get a clear DatasetError from the iterator, not an AttributeError
    # from deep inside header validation.
    dataset_path = Path(
        producer_settings.csv_path if csv_path is None else csv_path
    ).expanduser()
    interval = (
        producer_settings.stats_interval_seconds
        if stats_interval is None
        else max(0.5, stats_interval)
    )

    metrics.TARGET_TPS.set(target_tps)

    # Export metrics before connecting: a producer stuck in a connect-retry loop
    # is exactly when an operator wants to scrape connect_attempts_total.
    if not start_metrics_server(producer_settings.metrics_port, component="producer"):
        _LOGGER.warning(
            "producer metrics endpoint unavailable; continuing without it",
            extra={"port": producer_settings.metrics_port},
        )

    limiter = RateLimiter(target_tps)
    shutdown = _Shutdown()
    _install_signal_handlers(shutdown)

    _LOGGER.info(
        "starting producer",
        extra={
            "csv_path": str(dataset_path),
            "target_tps": target_tps,
            "throttled": limiter.enabled,
            "loop": loop,
            "fraud_only": replay_fraud_only,
            "max_records": max_records,
            "transactions_topic": config.kafka.transactions_topic,
            "bootstrap_servers": config.kafka.bootstrap_servers_string,
        },
    )

    started = time.perf_counter()
    last_stats = started
    published = 0

    try:
        with TransactionPublisher(
            config.kafka,
            flush_every_records=producer_settings.flush_every_records,
            flush_interval_seconds=producer_settings.flush_interval_seconds,
        ) as publisher:
            for record in iter_transactions(
                dataset_path,
                chunk_size=producer_settings.chunk_size,
                fraud_only=replay_fraud_only,
                max_records=max_records,
                loop=loop,
            ):
                if shutdown.requested:
                    break

                publisher.publish(record)
                published += 1
                metrics.RECORDS_READ.inc()
                if record.get("isFraud"):
                    metrics.FRAUD_RECORDS_READ.inc()

                # Pace after publishing so the publish time counts towards the
                # interval rather than being added to it (see rate_limiter).
                slept = limiter.acquire()
                if slept:
                    metrics.THROTTLE_SLEEP.inc(slept)

                now = time.perf_counter()
                if now - last_stats >= interval:
                    _log_stats(limiter, published, now - started)
                    last_stats = now
    except DatasetError as exc:
        _LOGGER.error("dataset problem, aborting", extra={"error": str(exc)})
        return _EXIT_FAILURE
    except PublisherError as exc:
        _LOGGER.error("kafka problem, aborting", extra={"error": str(exc)})
        return _EXIT_FAILURE

    elapsed = time.perf_counter() - started
    _LOGGER.info(
        "producer finished",
        extra={
            "records_published": published,
            "elapsed_seconds": round(elapsed, 3),
            "achieved_tps": round(published / elapsed, 2) if elapsed > 0 else 0.0,
            "target_tps": target_tps,
            "sleep_seconds": round(limiter.total_sleep_seconds, 3),
            "interrupted": shutdown.requested,
        },
    )
    return 0


def _log_stats(limiter: RateLimiter, published: int, elapsed: float) -> None:
    """Emit a periodic progress line.

    Args:
        limiter: The rate limiter, for achieved-rate reporting.
        published: Records published so far this run.
        elapsed: Seconds since the run started.
    """
    achieved = published / elapsed if elapsed > 0 else 0.0
    _LOGGER.info(
        "progress",
        extra={
            "records_published": published,
            "elapsed_seconds": round(elapsed, 1),
            "achieved_tps": round(achieved, 1),
            "target_tps": limiter.target_tps,
            "sleep_seconds_total": round(limiter.total_sleep_seconds, 2),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, resolve configuration and run.

    Args:
        argv: Argument vector, excluding the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    args = build_parser().parse_args(argv)

    try:
        config = get_config()
    except ConfigError as exc:
        # Logging may not be configured yet, so report and exit directly.
        print(f"configuration error: {exc}", file=sys.stderr)
        return _EXIT_FAILURE

    configure_logging(config.logging)

    return run(
        config,
        tps=args.tps,
        csv_path=args.csv_path,
        max_records=args.max_records,
        loop=args.loop,
        replay_fraud_only=args.replay_fraud_only,
        stats_interval=args.stats_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
