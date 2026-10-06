"""Resilient Kafka publisher for simulated transactions.

Partitioning contract
---------------------
Every message is keyed by ``nameOrig``, the originating account. This is not
cosmetic. Kafka guarantees ordering only within a partition, and the default
partitioner maps a given key to a fixed partition, so keying by account means:

* all transactions for one account arrive in order, and
* all transactions for one account are handled by one consumer task.

The streaming job's per-account velocity features depend on the second
property: each Spark executor can keep local state for the accounts in its
partitions without coordinating with the others. Switching to round-robin
keying would silently scatter an account's history across executors and corrupt
every velocity feature, with no error to indicate it.

Throughput
----------
Sends are asynchronous. Calling ``future.get()`` after each send would make
throughput the reciprocal of the broker round trip - a few hundred messages per
second on a local broker - which would make the 5,000 tps load test impossible.
Instead records are handed to the client's batching layer and acknowledged in
groups on a configured cadence (see
:attr:`~common.config.ProducerConfig.flush_every_records`). An error callback
catches delivery failures that would otherwise be swallowed by the future.

Durability
----------
``acks=1`` by default: the leader must write the record, but replicas need not
confirm. With the single-broker development stack this is identical to ``all``.
For a real deployment it is the wrong setting and ``KAFKA_PRODUCER_ACKS=all``
with ``replication_factor >= 3`` is the correct one; this is discussed in
``docs/architecture.md``. The tradeoff is deliberate and configurable rather
than compiled in.
"""

from __future__ import annotations

import random
import time
from types import TracebackType
from typing import Any, Final, Mapping

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, KafkaTimeoutError, NoBrokersAvailable
except ImportError:  # pragma: no cover - optional dependency in local shells
    KafkaProducer = None  # type: ignore[assignment]
    KafkaError = KafkaTimeoutError = NoBrokersAvailable = RuntimeError

from common.config import KafkaConfig
from common.logging_config import get_logger
from common.metrics import timed
from common.schema import encode_transaction
from producer import metrics

__all__ = ["PublisherError", "TransactionPublisher"]

_LOGGER = get_logger(__name__)

#: Fraction of the backoff delay applied as random jitter. The producer, the
#: streaming job and the dashboard all start at once under `make up`, and
#: without jitter their retries would stay in lockstep and hammer a recovering
#: broker in synchronised waves.
_JITTER_RATIO: Final[float] = 0.25

#: Seconds to wait for in-flight records during a flush. Generous relative to a
#: local broker's round trip, because exceeding it raises and aborts the run;
#: the flush cadence, not this value, is what bounds latency.
_FLUSH_TIMEOUT_SECONDS: Final[float] = 30.0


class PublisherError(RuntimeError):
    """Raised when the broker cannot be reached after exhausting retries."""


class TransactionPublisher:
    """Publishes PaySim transactions to Kafka with retry and backoff.

    Intended to be used as a context manager so that the final flush and the
    socket close happen even when the caller is interrupted::

        with TransactionPublisher(config) as publisher:
            for record in records:
                publisher.publish(record)

    Not thread-safe, which matches its single-threaded caller.

    Attributes:
        config: Kafka connection and tuning settings.
    """

    __slots__ = (
        "_bytes_pending",
        "_errback",
        "_failed_pending",
        "_flush_every_records",
        "_flush_interval_seconds",
        "_last_flush",
        "_producer",
        "_sent_pending",
        "config",
    )

    def __init__(
        self,
        config: KafkaConfig,
        *,
        flush_every_records: int,
        flush_interval_seconds: float,
    ) -> None:
        """Store configuration without opening a connection.

        Connecting is deferred to :meth:`connect` so that construction cannot
        block and callers decide when to incur the retry loop.

        The flush cadence is injected rather than read from the global
        configuration: it is consulted once per published record, and a hidden
        global lookup on that path is both a measurable cost at 5,000 tps and
        untestable without mutating the environment.

        Args:
            config: Kafka settings, normally ``get_config().kafka``.
            flush_every_records: Records to queue before blocking for
                acknowledgement.
            flush_interval_seconds: Maximum seconds between flushes.

        Raises:
            ValueError: If either flush bound is not positive.
        """
        if flush_every_records < 1:
            raise ValueError(
                f"flush_every_records must be >= 1, got {flush_every_records}"
            )
        if flush_interval_seconds <= 0.0:
            raise ValueError(
                f"flush_interval_seconds must be > 0, got {flush_interval_seconds}"
            )

        self.config = config
        self._flush_every_records = flush_every_records
        self._flush_interval_seconds = flush_interval_seconds
        self._producer: KafkaProducer | None = None
        # Cached bound method: registered on every send, so re-binding per
        # message would allocate for nothing on the hot path.
        self._errback = self._on_delivery_error
        self._sent_pending = 0
        self._failed_pending = 0
        self._bytes_pending = 0
        self._last_flush = time.perf_counter()
        metrics.KAFKA_CONNECTED.set(0)

    # -- connection -------------------------------------------------------
    @property
    def connected(self) -> bool:
        """Whether a producer client has been created.

        Returns:
            ``True`` once :meth:`connect` has succeeded and before
            :meth:`close`.
        """
        return self._producer is not None

    def _backoff_delay(self, attempt: int) -> float:
        """Compute the delay before a given retry attempt.

        Exponential with full jitter, capped at the configured maximum.
        Doubling alone would be adequate for a single client; the jitter is what
        keeps simultaneously starting components from retrying in unison.

        Args:
            attempt: 1-based attempt number that just failed.

        Returns:
            Delay in seconds.
        """
        exponential = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
        capped = min(exponential, self.config.retry_backoff_max_seconds)
        jitter = capped * _JITTER_RATIO * random.random()
        return capped + jitter

    def connect(self) -> None:
        """Open the producer connection, retrying with exponential backoff.

        Raises:
            PublisherError: If every attempt fails. The original error is
                chained so the cause is not lost.
        """
        if KafkaProducer is None:
            raise PublisherError(
                "kafka-python is not installed in this environment. "
                "Install the project dependencies or run via Docker Compose."
            )

        if self.connected:
            return

        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_connect_retries + 1):
            metrics.CONNECT_ATTEMPTS.inc()
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=list(self.config.bootstrap_servers),
                    # Keys are account identifiers, already str; values are
                    # pre-encoded JSON bytes, so neither needs a serialiser.
                    key_serializer=lambda key: key.encode("utf-8"),
                    acks=self.config.acks if self.config.acks == "all" else int(self.config.acks),
                    linger_ms=self.config.linger_ms,
                    batch_size=self.config.batch_size,
                    compression_type=(
                        None
                        if self.config.compression_type == "none"
                        else self.config.compression_type
                    ),
                    # Client-level retries cover transient leader elections.
                    # The outer loop in this method covers "broker not up yet",
                    # which the client cannot retry because it never connected.
                    retries=3,
                    max_in_flight_requests_per_connection=5,
                    client_id="fraud-producer",
                )
            except (NoBrokersAvailable, KafkaError, OSError) as exc:
                last_error = exc
                if attempt >= self.config.max_connect_retries:
                    break
                delay = self._backoff_delay(attempt)
                _LOGGER.warning(
                    "kafka connection failed, retrying",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self.config.max_connect_retries,
                        "retry_in_seconds": round(delay, 2),
                        "bootstrap_servers": self.config.bootstrap_servers_string,
                        "error": type(exc).__name__,
                    },
                )
                time.sleep(delay)
            else:
                metrics.KAFKA_CONNECTED.set(1)
                self._last_flush = time.perf_counter()
                _LOGGER.info(
                    "connected to kafka",
                    extra={
                        "bootstrap_servers": self.config.bootstrap_servers_string,
                        "topic": self.config.transactions_topic,
                        "acks": self.config.acks,
                        "compression": self.config.compression_type,
                        "attempts": attempt,
                    },
                )
                return

        metrics.KAFKA_CONNECTED.set(0)
        raise PublisherError(
            f"could not reach kafka at {self.config.bootstrap_servers_string} "
            f"after {self.config.max_connect_retries} attempts. "
            "Is the stack running? Try `make up` then `make health`."
        ) from last_error

    # -- publishing -------------------------------------------------------
    def _on_delivery_error(self, exception: BaseException) -> None:
        """Record a delivery failure reported by the client.

        Registered as the error callback on every send. Without it a record
        rejected by the broker would fail silently inside its future, and the
        run would report a success count it had not earned.

        Args:
            exception: The delivery error.
        """
        self._failed_pending += 1
        metrics.PUBLISH_FAILURES.labels(
            topic=self.config.transactions_topic,
            reason=type(exception).__name__,
        ).inc()
        _LOGGER.error(
            "record delivery failed",
            extra={
                "topic": self.config.transactions_topic,
                "error": type(exception).__name__,
                "detail": str(exception),
            },
        )

    def publish(
        self, record: Mapping[str, Any], *, produced_at_ms: int | None = None
    ) -> int:
        """Queue one transaction for delivery, flushing on cadence.

        Args:
            record: A PaySim record, as produced by
                :func:`common.dataset.iter_transactions`.
            produced_at_ms: Publish timestamp in epoch milliseconds, injected
                for deterministic tests. Defaults to now.

        Returns:
            Size of the serialised payload in bytes.

        Raises:
            PublisherError: If called before :meth:`connect`, or if the client's
                buffer stays full past its timeout - which means the broker is
                not draining and continuing would only deepen the backlog.
        """
        if self._producer is None:
            raise PublisherError("publish() called before connect()")

        payload = encode_transaction(record, produced_at_ms=produced_at_ms)
        key = str(record["nameOrig"])

        try:
            future = self._producer.send(
                self.config.transactions_topic, key=key, value=payload
            )
        except KafkaTimeoutError as exc:
            metrics.PUBLISH_FAILURES.labels(
                topic=self.config.transactions_topic,
                reason=type(exc).__name__,
            ).inc()
            raise PublisherError(
                "kafka send buffer full and not draining; the broker is not "
                "keeping up with the configured rate"
            ) from exc

        future.add_errback(self._errback)
        self._sent_pending += 1
        self._bytes_pending += len(payload)

        if self._should_flush():
            self.flush()

        return len(payload)

    def _should_flush(self) -> bool:
        """Whether the flush cadence has been reached.

        Returns:
            ``True`` when either the record or the time bound is met. See
            ``PRODUCER_FLUSH_EVERY_RECORDS`` in ``.env.example`` for why both
            exist.
        """
        if self._sent_pending >= self._flush_every_records:
            return True
        return (time.perf_counter() - self._last_flush) >= self._flush_interval_seconds

    def flush(self) -> None:
        """Block until queued records are acknowledged, then update metrics.

        Counters advance here rather than at send time so that
        ``messages_published`` means "the broker confirmed it", not "we handed
        it over". Failures reported through the error callback are subtracted.

        Raises:
            PublisherError: If the flush does not complete within its timeout.
        """
        if self._producer is None or self._sent_pending == 0:
            self._last_flush = time.perf_counter()
            return

        try:
            with timed(metrics.PUBLISH_LATENCY):
                self._producer.flush(timeout=_FLUSH_TIMEOUT_SECONDS)
        except KafkaTimeoutError as exc:
            raise PublisherError(
                f"kafka flush did not complete within {_FLUSH_TIMEOUT_SECONDS:.0f}s; "
                f"{self._sent_pending} records remain unacknowledged"
            ) from exc

        acknowledged = max(0, self._sent_pending - self._failed_pending)
        if acknowledged:
            topic = self.config.transactions_topic
            metrics.MESSAGES_PUBLISHED.labels(topic=topic).inc(acknowledged)
            # Attributed to acknowledged records only, so a failing broker does
            # not inflate the bandwidth panel.
            metrics.BYTES_PUBLISHED.labels(topic=topic).inc(
                self._bytes_pending
                if not self._failed_pending
                else int(self._bytes_pending * acknowledged / self._sent_pending)
            )

        self._sent_pending = 0
        self._failed_pending = 0
        self._bytes_pending = 0
        self._last_flush = time.perf_counter()

    def close(self) -> None:
        """Flush outstanding records and close the connection.

        Idempotent, and deliberately tolerant of failure: this runs on the
        shutdown path, where raising would mask whatever caused the shutdown.
        """
        if self._producer is None:
            return

        try:
            self.flush()
        except PublisherError as exc:
            _LOGGER.error(
                "final flush failed, some records were not delivered",
                extra={"error": str(exc)},
            )

        try:
            self._producer.close(timeout=_FLUSH_TIMEOUT_SECONDS)
        except KafkaError as exc:
            _LOGGER.warning(
                "error while closing kafka producer",
                extra={"error": type(exc).__name__, "detail": str(exc)},
            )
        finally:
            self._producer = None
            metrics.KAFKA_CONNECTED.set(0)
            _LOGGER.info("kafka producer closed")

    # -- context manager --------------------------------------------------
    def __enter__(self) -> TransactionPublisher:
        """Connect on entry.

        Returns:
            The connected publisher.
        """
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush and close on exit, including when the body raised."""
        self.close()
