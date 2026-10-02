"""Unit tests for the shared configuration layer.

The configuration module is the one component every other component depends on,
so its failure modes matter more than most: a bad default here becomes a
confusing error deep inside a Spark task. These tests pin the behaviour that
other modules rely on - typed parsing, bounds enforcement, and the immutability
that keeps a run's settings from changing under it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from common import config as config_module
from common.config import (
    AppConfig,
    CassandraConfig,
    ConfigError,
    KafkaConfig,
    LoggingConfig,
    ProducerConfig,
    get_config,
    load_dotenv,
)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the config cache and any pipeline variables around each test.

    get_config() is lru_cached and reads the process environment, so without this
    a test's environment changes would leak into its neighbours and produce
    order-dependent failures.
    """
    for key in list(os.environ):
        if key.startswith(("KAFKA_", "CASSANDRA_", "PRODUCER_", "LOG_", "PAYSIM_")):
            monkeypatch.delenv(key, raising=False)
    get_config.cache_clear()
    yield
    get_config.cache_clear()


# ---------------------------------------------------------------------------
# Typed environment readers
# ---------------------------------------------------------------------------
class TestEnvReaders:
    """The private readers validate and coerce at startup, not at first use."""

    def test_int_uses_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAFKA_PRODUCER_LINGER_MS", raising=False)
        assert KafkaConfig.from_env().linger_ms == 10

    def test_int_rejects_non_numeric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_PRODUCER_LINGER_MS", "soon")
        with pytest.raises(ConfigError, match="not a valid integer"):
            KafkaConfig.from_env()

    def test_int_enforces_minimum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_PRODUCER_BATCH_SIZE", "1")
        with pytest.raises(ConfigError, match="must be >= 1024"):
            KafkaConfig.from_env()

    def test_float_rejects_non_numeric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_RETRY_BACKOFF_SECONDS", "fast")
        with pytest.raises(ConfigError, match="not a valid number"):
            KafkaConfig.from_env()

    def test_empty_string_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A blank .env entry must not become an empty hostname.
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "")
        assert KafkaConfig.from_env().bootstrap_servers == ("localhost:9092",)

    def test_csv_splits_and_trims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", " a:9092 , b:9092 ,")
        assert KafkaConfig.from_env().bootstrap_servers == ("a:9092", "b:9092")

    def test_csv_rejects_only_commas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", " , , ")
        with pytest.raises(ConfigError, match="at least one value"):
            KafkaConfig.from_env()


# ---------------------------------------------------------------------------
# Kafka configuration
# ---------------------------------------------------------------------------
class TestKafkaConfig:
    """Enum-like fields and backoff ordering are validated on construction."""

    def test_rejects_unknown_compression(self) -> None:
        with pytest.raises(ConfigError, match="must be one of"):
            KafkaConfig(
                bootstrap_servers=("localhost:9092",),
                transactions_topic="t",
                alerts_topic="a",
                dlq_topic="d",
                transactions_partitions=1,
                alerts_partitions=1,
                consumer_group="g",
                linger_ms=10,
                batch_size=1024,
                compression_type="brotli",
                acks="1",
                max_connect_retries=1,
                retry_backoff_seconds=1.0,
                retry_backoff_max_seconds=2.0,
            )

    def test_rejects_unknown_acks(self) -> None:
        with pytest.raises(ConfigError, match="must be '0', '1' or 'all'"):
            KafkaConfig(
                bootstrap_servers=("localhost:9092",),
                transactions_topic="t",
                alerts_topic="a",
                dlq_topic="d",
                transactions_partitions=1,
                alerts_partitions=1,
                consumer_group="g",
                linger_ms=10,
                batch_size=1024,
                compression_type="lz4",
                acks="2",
                max_connect_retries=1,
                retry_backoff_seconds=1.0,
                retry_backoff_max_seconds=2.0,
            )

    def test_rejects_backoff_max_below_base(self) -> None:
        with pytest.raises(ConfigError, match="BACKOFF_MAX_SECONDS must be"):
            KafkaConfig(
                bootstrap_servers=("localhost:9092",),
                transactions_topic="t",
                alerts_topic="a",
                dlq_topic="d",
                transactions_partitions=1,
                alerts_partitions=1,
                consumer_group="g",
                linger_ms=10,
                batch_size=1024,
                compression_type="lz4",
                acks="1",
                max_connect_retries=1,
                retry_backoff_seconds=5.0,
                retry_backoff_max_seconds=1.0,
            )

    def test_bootstrap_servers_string_joins(self) -> None:
        cfg = KafkaConfig.from_env()
        assert cfg.bootstrap_servers_string == "localhost:9092"

    def test_is_frozen(self) -> None:
        cfg = KafkaConfig.from_env()
        with pytest.raises(Exception):
            cfg.transactions_topic = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Producer configuration
# ---------------------------------------------------------------------------
class TestProducerConfig:
    """The producer block carries the crash-loss bound, so its bounds matter."""

    def test_defaults(self) -> None:
        cfg = ProducerConfig.from_env()
        assert cfg.tps == 50
        assert cfg.throttled is True
        assert cfg.csv_path == config_module.repo_root() / "data/paysim.csv"

    def test_zero_tps_means_unthrottled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRODUCER_TPS", "0")
        assert ProducerConfig.from_env().throttled is False

    def test_rejects_negative_tps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRODUCER_TPS", "-1")
        with pytest.raises(ConfigError, match="must be >= 0"):
            ProducerConfig.from_env()

    def test_rejects_tiny_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRODUCER_CHUNK_SIZE", "500")
        with pytest.raises(ConfigError, match="too small"):
            ProducerConfig.from_env()

    def test_rejects_flush_bound_above_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Above the cap the send buffer, not the configured cadence, decides when
        # data reaches the broker, so the crash-loss bound stops being meaningful.
        monkeypatch.setenv("PRODUCER_FLUSH_EVERY_RECORDS", "10001")
        with pytest.raises(ConfigError, match="too large"):
            ProducerConfig.from_env()

    def test_absolute_csv_path_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAYSIM_CSV_PATH", "/tmp/other.csv")
        assert ProducerConfig.from_env().csv_path == Path("/tmp/other.csv")

    def test_relative_csv_path_resolves_to_repo_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAYSIM_CSV_PATH", "data/x.csv")
        assert (
            ProducerConfig.from_env().csv_path == config_module.repo_root() / "data/x.csv"
        )


# ---------------------------------------------------------------------------
# Cassandra configuration
# ---------------------------------------------------------------------------
class TestCassandraConfig:
    """Credential pairing and batch sizing are the two failure modes here."""

    def test_credentials_absent_by_default(self) -> None:
        cfg = CassandraConfig.from_env()
        assert cfg.auth_required is False

    def test_username_without_password_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CASSANDRA_USERNAME", "cassandra")
        with pytest.raises(ConfigError, match="must be set together"):
            CassandraConfig.from_env()

    def test_credentials_paired_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CASSANDRA_USERNAME", "cassandra")
        monkeypatch.setenv("CASSANDRA_PASSWORD", "secret")
        assert CassandraConfig.from_env().auth_required is True

    def test_rejects_oversized_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CASSANDRA_WRITE_BATCH_SIZE", "501")
        with pytest.raises(ConfigError, match="too large"):
            CassandraConfig.from_env()


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
class TestLoggingConfig:
    """Level and format are validated against fixed vocabularies."""

    def test_defaults(self) -> None:
        cfg = LoggingConfig.from_env()
        assert cfg.level == "INFO"
        assert cfg.format == "text"

    def test_level_is_upper_cased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert LoggingConfig.from_env().level == "DEBUG"

    def test_rejects_unknown_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ConfigError, match="must be one of"):
            LoggingConfig.from_env()

    def test_rejects_unknown_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_FORMAT", "xml")
        with pytest.raises(ConfigError, match="must be 'text' or 'json'"):
            LoggingConfig.from_env()


# ---------------------------------------------------------------------------
# .env loading and aggregation
# ---------------------------------------------------------------------------
class TestDotenvLoading:
    """The local reader handles the subset of .env syntax the project uses."""

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        assert load_dotenv(tmp_path / "absent.env") == 0

    def test_parses_comments_quotes_and_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# a comment\n"
            "\n"
            "export KAFKA_PRODUCER_ACKS=all\n"
            "LOG_LEVEL=\"warning\"\n"
            "PRODUCER_TPS=100  # inline comment\n"
        )
        applied = load_dotenv(env_file)
        assert applied == 3
        assert os.environ["KAFKA_PRODUCER_ACKS"] == "all"
        assert os.environ["LOG_LEVEL"] == "warning"
        assert os.environ["PRODUCER_TPS"] == "100"

    def test_existing_environment_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `make produce TPS=500` must beat the .env file.
        monkeypatch.setenv("PRODUCER_TPS", "999")
        env_file = tmp_path / ".env"
        env_file.write_text("PRODUCER_TPS=100\n")
        load_dotenv(env_file)
        assert os.environ["PRODUCER_TPS"] == "999"

    def test_override_replaces_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRODUCER_TPS", "999")
        env_file = tmp_path / ".env"
        env_file.write_text("PRODUCER_TPS=100\n")
        load_dotenv(env_file, override=True)
        assert os.environ["PRODUCER_TPS"] == "100"


class TestGetConfig:
    """The aggregate is cached and read from the environment exactly once."""

    def test_is_cached(self) -> None:
        assert get_config() is get_config()

    def test_cache_clear_rereads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRODUCER_TPS", "7")
        get_config.cache_clear()
        assert get_config().producer.tps == 7

    def test_aggregate_exposes_all_sections(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, AppConfig)
        assert isinstance(cfg.kafka, KafkaConfig)
        assert isinstance(cfg.cassandra, CassandraConfig)
        assert isinstance(cfg.producer, ProducerConfig)
        assert isinstance(cfg.logging, LoggingConfig)
