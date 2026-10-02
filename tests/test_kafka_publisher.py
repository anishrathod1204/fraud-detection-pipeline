"""Unit tests for the resilient Kafka publisher.

The publisher sits between the dataset reader and the broker, and its job is to
turn "read a row" into "the broker has durably accepted this row" without lying
about it. So these tests concentrate on the honesty properties rather than the
mechanics: that a delivery failure is counted as a failure and not a success,
that the acknowledged total in :meth:`flush` reflects reality, and that every
connection failure is retried and then surfaced as a single, chained error.

The Kafka client itself is stubbed (see ``tests/conftest.py``): the stub models
the two behaviours the publisher actually depends on - the key serialiser and
the error callback on a send future - so the surrounding retry, batching and
accounting logic can be exercised without a broker.

Metric assertions are written as deltas rather than absolutes, because the
counters are process-global and their absolute values depend on how many tests
have already run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from common.config import KafkaConfig
from common.schema import encode_transaction
from producer import metrics
from producer.kafka_publisher import PublisherError, TransactionPublisher
from producer import kafka_publisher as publisher_module


TRANSACTIONS_TOPIC = "transactions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_config(**overrides: Any) -> KafkaConfig:
    """Build a deterministic Kafka configuration for a test.

    Args:
        **overrides: Field values to replace in the default template.

    Returns:
        A validated :class:`KafkaConfig`.
    """
    params: dict[str, Any] = {
        "bootstrap_servers": ("localhost:9092",),
        "transactions_topic": TRANSACTIONS_TOPIC,
        "alerts_topic": "fraud_alerts",
        "dlq_topic": "transactions_dlq",
        "transactions_partitions": 6,
        "alerts_partitions": 6,
        "consumer_group": "fraud-detector",
        "linger_ms": 10,
        "batch_size": 16384,
        "compression_type": "lz4",
        "acks": "1",
        "max_connect_retries": 3,
        "retry_backoff_seconds": 0.01,
        "retry_backoff_max_seconds": 0.05,
    }
    params.update(overrides)
    return KafkaConfig(**params)


def metric_value(metric: Any) -> float:
    """Read a counter or gauge value under either the stub or real client.

    Args:
        metric: A prometheus counter, gauge, or a labelled child of one.

    Returns:
        The current numeric value.
    """
    # The test stub exposes a plain ``.value`` attribute; prometheus_client
    # exposes it via ``_value.get()``. Supporting both keeps these tests valid
    # whether or not the real dependency is installed.
    if hasattr(metric, "value"):
        return float(metric.value)
    return float(metric._value.get())  # noqa: SLF001 - documented compatibility shim


class _FakeFuture:
    """A minimal stand-in for a kafka-python send future."""

    def __init__(self, error: BaseException | None = None) -> None:
        self._error = error

    def add_errback(self, callback: Any) -> None:
        """Fire the callback immediately when an error is baked in."""
        if self._error is not None:
            callback(self._error)

    def add_callback(self, callback: Any) -> None:
        """No-op; the publisher never registers success callbacks."""


class _RecordingProducer:
    """A fake KafkaProducer that records calls and can be told to fail.

    Attributes:
        sent: ``(topic, key, value)`` tuples for every accepted send.
        flush_calls: Number of times ``flush`` was invoked.
        closed: Whether ``close`` was called.
    """

    def __init__(
        self,
        *,
        send_error: BaseException | None = None,
        deliver_errors: list[BaseException | None] | None = None,
        flush_error: BaseException | None = None,
    ) -> None:
        """Initialise the fake.

        Args:
            send_error: If set, ``send`` raises it instead of accepting.
            deliver_errors: Per-send delivery outcomes, consumed in order. A
                ``None`` entry is a successful delivery; an exception is
                reported through the send future's error callback. Entries
                beyond the list (and the whole list when omitted) succeed.
            flush_error: If set, ``flush`` raises it.
        """
        self.sent: list[tuple[str, Any, Any]] = []
        self.flush_calls = 0
        self.closed = False
        self._send_error = send_error
        self._deliver_errors = list(deliver_errors) if deliver_errors else []
        self._send_index = 0
        self._flush_error = flush_error

    def send(self, topic: str, key: Any = None, value: Any = None) -> _FakeFuture:
        """Record a send, or raise if configured to do so."""
        if self._send_error is not None:
            raise self._send_error
        self.sent.append((topic, key, value))
        error: BaseException | None = None
        if self._send_index < len(self._deliver_errors):
            error = self._deliver_errors[self._send_index]
        self._send_index += 1
        return _FakeFuture(error)

    def flush(self, timeout: float | None = None) -> None:
        """Count a flush, raising first if configured to."""
        self.flush_calls += 1
        if self._flush_error is not None:
            raise self._flush_error

    def close(self, timeout: float | None = None) -> None:
        """Mark the producer closed."""
        self.closed = True


class _FakeClock:
    """A monotonic clock the test advances by hand, for cadence tests."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def perf_counter(self) -> float:
        """Return the current fake time."""
        return self.now


def make_publisher(
    producer: _RecordingProducer | None = None,
    *,
    flush_every_records: int = 500,
    flush_interval_seconds: float = 1.0,
    config: KafkaConfig | None = None,
) -> TransactionPublisher:
    """Build a publisher with an injected producer, bypassing connect().

    Args:
        producer: Fake client to inject; defaults to a fresh one.
        flush_every_records: Record-count flush bound.
        flush_interval_seconds: Time flush bound.
        config: Override configuration.

    Returns:
        A publisher whose ``_producer`` is the injected fake.
    """
    publisher = TransactionPublisher(
        config or make_config(),
        flush_every_records=flush_every_records,
        flush_interval_seconds=flush_interval_seconds,
    )
    publisher._producer = producer or _RecordingProducer()  # noqa: SLF001
    return publisher


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------
class TestConstruction:
    """The injected flush bounds are validated up front."""

    def test_rejects_zero_flush_every_records(self) -> None:
        with pytest.raises(ValueError, match="flush_every_records must be >= 1"):
            TransactionPublisher(
                make_config(), flush_every_records=0, flush_interval_seconds=1.0
            )

    def test_rejects_negative_flush_every_records(self) -> None:
        with pytest.raises(ValueError, match="flush_every_records must be >= 1"):
            TransactionPublisher(
                make_config(), flush_every_records=-5, flush_interval_seconds=1.0
            )

    def test_rejects_non_positive_flush_interval(self) -> None:
        with pytest.raises(ValueError, match="flush_interval_seconds must be > 0"):
            TransactionPublisher(
                make_config(), flush_every_records=1, flush_interval_seconds=0.0
            )

    def test_starts_disconnected_with_clean_counters(self) -> None:
        publisher = TransactionPublisher(
            make_config(), flush_every_records=10, flush_interval_seconds=1.0
        )
        assert publisher.connected is False
        assert metric_value(metrics.KAFKA_CONNECTED) == 0.0


# ---------------------------------------------------------------------------
# Connection and retries
# ---------------------------------------------------------------------------
class TestConnect:
    """Connection failure is retried, then surfaced once, with the cause kept."""

    def test_success_sets_connected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _RecordingProducer()
        monkeypatch.setattr(publisher_module, "KafkaProducer", lambda **_: fake)
        before = metric_value(metrics.CONNECT_ATTEMPTS)

        publisher = TransactionPublisher(
            make_config(), flush_every_records=10, flush_interval_seconds=1.0
        )
        publisher.connect()

        assert publisher.connected is True
        assert metric_value(metrics.KAFKA_CONNECTED) == 1.0
        assert metric_value(metrics.CONNECT_ATTEMPTS) - before == 1

    def test_success_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def factory(**_: Any) -> _RecordingProducer:
            calls.append(1)
            return _RecordingProducer()

        monkeypatch.setattr(publisher_module, "KafkaProducer", factory)
        publisher = TransactionPublisher(
            make_config(), flush_every_records=10, flush_interval_seconds=1.0
        )
        publisher.connect()
        publisher.connect()
        assert len(calls) == 1

    def test_retry_exhaustion_raises_chained_error(
        self, monkeypatch: pytest.MonkeyPatch, kafka_module: Any
    ) -> None:
        monkeypatch.setattr(
            publisher_module,
            "KafkaProducer",
            _raiser(kafka_module.errors.NoBrokersAvailable("broker down")),
        )
        monkeypatch.setattr(publisher_module.time, "sleep", lambda _s: None)
        before = metric_value(metrics.CONNECT_ATTEMPTS)

        publisher = TransactionPublisher(
            make_config(max_connect_retries=3),
            flush_every_records=10,
            flush_interval_seconds=1.0,
        )
        with pytest.raises(PublisherError, match="could not reach kafka"):
            publisher.connect()

        # Every configured attempt was made, the cause survived, and the
        # liveness gauge reports the failure honestly.
        assert metric_value(metrics.CONNECT_ATTEMPTS) - before == 3
        assert isinstance(
            publisher.connected, bool
        )
        assert publisher.connected is False
        assert metric_value(metrics.KAFKA_CONNECTED) == 0.0

    def test_retry_exhaustion_chain_is_original_error(
        self, monkeypatch: pytest.MonkeyPatch, kafka_module: Any
    ) -> None:
        original = kafka_module.errors.NoBrokersAvailable("nope")
        monkeypatch.setattr(
            publisher_module, "KafkaProducer", _raiser(original)
        )
        monkeypatch.setattr(publisher_module.time, "sleep", lambda _s: None)

        publisher = TransactionPublisher(
            make_config(max_connect_retries=2),
            flush_every_records=10,
            flush_interval_seconds=1.0,
        )
        with pytest.raises(PublisherError) as excinfo:
            publisher.connect()
        assert excinfo.value.__cause__ is original

    def test_recovers_on_a_later_attempt(
        self, monkeypatch: pytest.MonkeyPatch, kafka_module: Any
    ) -> None:
        attempts = {"n": 0}
        fake = _RecordingProducer()

        def flaky(**_: Any) -> _RecordingProducer:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise kafka_module.errors.NoBrokersAvailable("not yet")
            return fake

        monkeypatch.setattr(publisher_module, "KafkaProducer", flaky)
        monkeypatch.setattr(publisher_module.time, "sleep", lambda _s: None)

        publisher = TransactionPublisher(
            make_config(max_connect_retries=5),
            flush_every_records=10,
            flush_interval_seconds=1.0,
        )
        publisher.connect()
        assert publisher.connected is True
        assert attempts["n"] == 3


def _raiser(exc: BaseException) -> Any:
    """Return a callable that always raises the given exception.

    Args:
        exc: The exception instance to raise.

    Returns:
        A callable accepting ``**kwargs`` and raising ``exc``.
    """

    def _raise(**_: Any) -> Any:
        raise exc

    return _raise


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------
class TestBackoff:
    """Exponential with full jitter, capped at the configured maximum."""

    def test_first_delay_is_base_plus_jitter(self) -> None:
        publisher = make_publisher(config=make_config(
            retry_backoff_seconds=1.0, retry_backoff_max_seconds=8.0
        ))
        delay = publisher._backoff_delay(1)  # noqa: SLF001
        assert 1.0 <= delay <= 1.25

    def test_delay_grows_exponentially(self) -> None:
        publisher = make_publisher(config=make_config(
            retry_backoff_seconds=1.0, retry_backoff_max_seconds=100.0
        ))
        # Attempt 3 -> 1 * 2^2 = 4s base.
        delay = publisher._backoff_delay(3)  # noqa: SLF001
        assert 4.0 <= delay <= 5.0

    def test_delay_is_capped(self) -> None:
        publisher = make_publisher(config=make_config(
            retry_backoff_seconds=1.0, retry_backoff_max_seconds=2.0
        ))
        # Attempt 10 -> 2^9 = 512s, capped to 2s base with jitter.
        delay = publisher._backoff_delay(10)  # noqa: SLF001
        assert 2.0 <= delay <= 2.5


# ---------------------------------------------------------------------------
# Publishing and keying
# ---------------------------------------------------------------------------
class TestPublish:
    """Publishing keys by account and returns the serialised size."""

    def test_publish_before_connect_raises(self, sample_record: dict[str, Any]) -> None:
        publisher = TransactionPublisher(
            make_config(), flush_every_records=10, flush_interval_seconds=1.0
        )
        with pytest.raises(PublisherError, match="before connect"):
            publisher.publish(sample_record)

    def test_publish_keys_by_name_orig(self, sample_record: dict[str, Any]) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(fake)
        publisher.publish(sample_record, produced_at_ms=123)
        topic, key, _value = fake.sent[0]
        assert topic == TRANSACTIONS_TOPIC
        assert key == sample_record["nameOrig"]

    def test_publish_returns_payload_size(self, sample_record: dict[str, Any]) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(fake)
        size = publisher.publish(sample_record, produced_at_ms=123)
        expected = len(encode_transaction(sample_record, produced_at_ms=123))
        assert size == expected
        assert len(fake.sent[0][2]) == expected

    def test_serializes_key_via_client_serializer(
        self, sample_record: dict[str, Any], kafka_module: Any
    ) -> None:
        # Against the real stub client (which applies the key serializer the
        # publisher passes) the key on the wire is UTF-8 bytes, not a str.
        publisher = TransactionPublisher(
            make_config(), flush_every_records=500, flush_interval_seconds=3600.0
        )
        publisher.connect()
        publisher.publish(sample_record, produced_at_ms=1)
        _topic, key, _value = publisher._producer.sent[0]  # noqa: SLF001
        assert key == sample_record["nameOrig"].encode("utf-8")

    def test_send_timeout_becomes_publisher_error(
        self, sample_record: dict[str, Any], kafka_module: Any
    ) -> None:
        fake = _RecordingProducer(
            send_error=kafka_module.errors.KafkaTimeoutError("buffer full")
        )
        publisher = make_publisher(fake)
        with pytest.raises(PublisherError, match="send buffer full"):
            publisher.publish(sample_record, produced_at_ms=1)

    def test_delivery_error_is_recorded_not_published(
        self, sample_record: dict[str, Any], kafka_module: Any
    ) -> None:
        fake = _RecordingProducer(
            deliver_errors=[kafka_module.errors.KafkaError("rejected")]
        )
        publisher = make_publisher(fake, flush_every_records=500)
        before = metric_value(metrics.PUBLISH_FAILURES.labels(
            topic=TRANSACTIONS_TOPIC, reason="KafkaError"
        ))
        publisher.publish(sample_record, produced_at_ms=1)
        after = metric_value(metrics.PUBLISH_FAILURES.labels(
            topic=TRANSACTIONS_TOPIC, reason="KafkaError"
        ))
        assert after - before == 1


# ---------------------------------------------------------------------------
# Flush cadence
# ---------------------------------------------------------------------------
class TestFlushCadence:
    """Either the record bound or the time bound triggers a flush."""

    def test_record_bound_triggers_flush(self, sample_record: dict[str, Any]) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(
            fake, flush_every_records=3, flush_interval_seconds=3600.0
        )
        publisher.publish(sample_record, produced_at_ms=1)
        publisher.publish(sample_record, produced_at_ms=1)
        assert fake.flush_calls == 0
        publisher.publish(sample_record, produced_at_ms=1)
        assert fake.flush_calls == 1

    def test_time_bound_triggers_flush(
        self, sample_record: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _RecordingProducer()
        clock = _FakeClock()
        monkeypatch.setattr(publisher_module.time, "perf_counter", clock.perf_counter)
        publisher = make_publisher(
            fake, flush_every_records=10_000, flush_interval_seconds=0.05
        )
        publisher.publish(sample_record, produced_at_ms=1)
        assert fake.flush_calls == 0
        clock.now += 0.06  # more than the interval has elapsed
        publisher.publish(sample_record, produced_at_ms=1)
        assert fake.flush_calls == 1


# ---------------------------------------------------------------------------
# Flush accounting
# ---------------------------------------------------------------------------
class TestFlushAccounting:
    """`messages_published` must mean the broker confirmed it."""

    def test_empty_flush_is_a_noop(self) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(fake)
        publisher.flush()
        assert fake.flush_calls == 0

    def test_acknowledged_count_matches_sends(
        self, sample_record: dict[str, Any]
    ) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(fake, flush_every_records=10_000)
        before = metric_value(
            metrics.MESSAGES_PUBLISHED.labels(topic=TRANSACTIONS_TOPIC)
        )
        for _ in range(4):
            publisher.publish(sample_record, produced_at_ms=1)
        publisher.flush()
        after = metric_value(
            metrics.MESSAGES_PUBLISHED.labels(topic=TRANSACTIONS_TOPIC)
        )
        assert after - before == 4

    def test_failed_records_are_subtracted(
        self, sample_record: dict[str, Any], kafka_module: Any
    ) -> None:
        # One of three sends reports a delivery failure, so only two are
        # acknowledged - a naive "count every send" would report three.
        fake = _RecordingProducer(
            deliver_errors=[None, kafka_module.errors.KafkaError("boom"), None]
        )
        publisher = make_publisher(fake, flush_every_records=10_000)
        before = metric_value(
            metrics.MESSAGES_PUBLISHED.labels(topic=TRANSACTIONS_TOPIC)
        )
        for _ in range(3):
            publisher.publish(sample_record, produced_at_ms=1)
        publisher.flush()
        after = metric_value(
            metrics.MESSAGES_PUBLISHED.labels(topic=TRANSACTIONS_TOPIC)
        )
        assert after - before == 2

    def test_flush_resets_pending_counters(
        self, sample_record: dict[str, Any]
    ) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(fake, flush_every_records=10_000)
        publisher.publish(sample_record, produced_at_ms=1)
        publisher.flush()
        # A second flush has nothing pending, so the client is not asked again.
        publisher.flush()
        assert fake.flush_calls == 1

    def test_flush_timeout_becomes_publisher_error(
        self, sample_record: dict[str, Any], kafka_module: Any
    ) -> None:
        fake = _RecordingProducer(
            flush_error=kafka_module.errors.KafkaTimeoutError("slow")
        )
        publisher = make_publisher(fake, flush_every_records=10_000)
        publisher.publish(sample_record, produced_at_ms=1)
        with pytest.raises(PublisherError, match="did not complete"):
            publisher.flush()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------
class TestClose:
    """Closing flushes, releases the client, and is safe to repeat."""

    def test_close_flushes_and_disconnects(
        self, sample_record: dict[str, Any]
    ) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(fake, flush_every_records=10_000)
        publisher.publish(sample_record, produced_at_ms=1)
        publisher.close()
        assert fake.closed is True
        assert fake.flush_calls == 1
        assert publisher.connected is False
        assert metric_value(metrics.KAFKA_CONNECTED) == 0.0

    def test_close_is_idempotent(self, sample_record: dict[str, Any]) -> None:
        fake = _RecordingProducer()
        publisher = make_publisher(fake, flush_every_records=10_000)
        publisher.publish(sample_record, produced_at_ms=1)
        publisher.close()
        publisher.close()  # no error, no second client close
        assert publisher.connected is False

    def test_close_without_connect_is_safe(self) -> None:
        publisher = TransactionPublisher(
            make_config(), flush_every_records=10, flush_interval_seconds=1.0
        )
        publisher.close()  # must not raise

    def test_close_survives_flush_failure(
        self, sample_record: dict[str, Any], kafka_module: Any
    ) -> None:
        # The shutdown path must not raise on a flush error, or it would mask
        # whatever triggered the shutdown.
        fake = _RecordingProducer(
            flush_error=kafka_module.errors.KafkaTimeoutError("slow")
        )
        publisher = make_publisher(fake, flush_every_records=10_000)
        publisher.publish(sample_record, produced_at_ms=1)
        publisher.close()
        assert publisher.connected is False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
class TestContextManager:
    """Entry connects; exit closes even when the body raised."""

    def test_connects_on_entry_closes_on_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _RecordingProducer()
        monkeypatch.setattr(publisher_module, "KafkaProducer", lambda **_: fake)
        publisher = TransactionPublisher(
            make_config(), flush_every_records=10_000, flush_interval_seconds=1.0
        )
        with publisher as entered:
            assert entered is publisher
            assert publisher.connected is True
        assert publisher.connected is False
        assert fake.closed is True

    def test_closes_when_body_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _RecordingProducer()
        monkeypatch.setattr(publisher_module, "KafkaProducer", lambda **_: fake)
        publisher = TransactionPublisher(
            make_config(), flush_every_records=10_000, flush_interval_seconds=1.0
        )
        with pytest.raises(RuntimeError, match="boom"):
            with publisher:
                raise RuntimeError("boom")
        assert publisher.connected is False
        assert fake.closed is True
