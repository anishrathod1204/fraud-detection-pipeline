"""Environment-driven configuration for every component of the pipeline.

Design
------
A single module owns configuration so that the producer, the training job and
the Spark streaming job cannot drift apart. The feature-engineering settings in
particular *must* be identical between training and scoring, and sharing one
loader is what guarantees that.

Rules this module enforces:

* Nothing is hardcoded at a call site. Every tunable has an environment
  variable, a documented default, and a type.
* Configuration is read once and frozen. :func:`get_config` is cached, and the
  dataclasses are immutable, so a component cannot mutate settings halfway
  through a run and produce results that no longer match its own config.
* Invalid configuration fails at startup with a precise message, not later with
  a confusing ``ZeroDivisionError`` deep inside a hot loop.

``.env`` is parsed by a small local reader rather than ``python-dotenv``. The
file format in use here is trivial, and the fewer runtime dependencies the
Spark executors need, the fewer version conflicts there are to debug.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

__all__ = [
    "AppConfig",
    "CassandraConfig",
    "ConfigError",
    "KafkaConfig",
    "LoggingConfig",
    "get_config",
    "load_dotenv",
    "repo_root",
]


class ConfigError(ValueError):
    """Raised when an environment variable is missing or cannot be parsed.

    Subclasses :class:`ValueError` so that callers which only care about "bad
    input" need not import this module.
    """


def repo_root() -> Path:
    """Return the repository root directory.

    Resolved from this file's location rather than the process working
    directory, so relative paths in configuration (``data/paysim.csv``,
    ``training/artifacts``) mean the same thing whether a component is launched
    from the repo root, from a subdirectory, or by ``spark-submit``.

    Returns:
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> int:
    """Populate :data:`os.environ` from a ``.env`` file.

    Supports ``KEY=value`` lines, ``#`` comments, blank lines, an optional
    ``export`` prefix, and single- or double-quoted values. Anything else is
    ignored rather than raising, because a malformed comment in a local file
    should not prevent the pipeline from starting.

    Args:
        path: File to read. Defaults to ``.env`` at the repository root.
        override: When ``False`` (the default) existing environment variables
            win, which is what allows ``docker-compose.yml`` and
            ``make produce TPS=500`` to override the file.

    Returns:
        Number of variables applied to the environment.
    """
    env_path = Path(path) if path is not None else repo_root() / ".env"
    if not env_path.is_file():
        return 0

    applied = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, separator, value = line.partition("=")
        if not separator:
            continue

        key = key.strip()
        value = value.strip()
        # Strip one matching pair of surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        else:
            # Unquoted values may carry a trailing inline comment.
            value = value.split(" #", 1)[0].rstrip()

        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied += 1

    return applied


# ---------------------------------------------------------------------------
# Typed environment readers
# ---------------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    """Read a string environment variable, falling back to ``default``.

    An empty value is treated as "unset" so that a blank entry in ``.env`` does
    not silently produce an empty hostname.

    Args:
        name: Environment variable name.
        default: Value to use when unset or empty.

    Returns:
        The configured string.
    """
    value = os.environ.get(name, "")
    return value if value != "" else default


def _env_optional_str(name: str) -> str | None:
    """Read an optional string environment variable.

    Args:
        name: Environment variable name.

    Returns:
        The value, or ``None`` when unset or empty. Used for credentials, where
        "absent" and "empty string" must be distinguished.
    """
    value = os.environ.get(name, "").strip()
    return value or None


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Read an integer environment variable.

    Args:
        name: Environment variable name.
        default: Value to use when unset.
        minimum: Optional inclusive lower bound.

    Returns:
        The parsed integer.

    Raises:
        ConfigError: If the value is not an integer or is below ``minimum``.
    """
    raw = os.environ.get(name, "").strip()
    if raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name}={raw!r} is not a valid integer") from exc

    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}={value} must be >= {minimum}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read a float environment variable.

    Args:
        name: Environment variable name.
        default: Value to use when unset.
        minimum: Optional inclusive lower bound.
        maximum: Optional inclusive upper bound.

    Returns:
        The parsed float.

    Raises:
        ConfigError: If the value is not a float or falls outside the bounds.
    """
    raw = os.environ.get(name, "").strip()
    if raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name}={raw!r} is not a valid number") from exc

    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}={value} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name}={value} must be <= {maximum}")
    return value


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    """Read a comma-separated environment variable as a tuple.

    Args:
        name: Environment variable name.
        default: Comma-separated fallback value.

    Returns:
        Tuple of trimmed, non-empty items.

    Raises:
        ConfigError: If the resulting tuple is empty.
    """
    raw = _env_str(name, default)
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not items:
        raise ConfigError(f"{name}={raw!r} must contain at least one value")
    return items


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class KafkaConfig:
    """Kafka connection, topic and client-tuning settings."""

    bootstrap_servers: tuple[str, ...]
    transactions_topic: str
    alerts_topic: str
    dlq_topic: str
    transactions_partitions: int
    alerts_partitions: int
    consumer_group: str
    linger_ms: int
    batch_size: int
    compression_type: str
    acks: str
    max_connect_retries: int
    retry_backoff_seconds: float
    retry_backoff_max_seconds: float

    #: Compression codecs the Kafka protocol understands.
    VALID_COMPRESSION: ClassVar[frozenset[str]] = frozenset(
        {"none", "gzip", "snappy", "lz4", "zstd"}
    )
    #: Producer acknowledgement modes.
    VALID_ACKS: ClassVar[frozenset[str]] = frozenset({"0", "1", "all"})

    def __post_init__(self) -> None:
        """Validate enum-like fields and backoff ordering.

        Raises:
            ConfigError: If compression, acks or the backoff bounds are invalid.
        """
        if self.compression_type not in self.VALID_COMPRESSION:
            raise ConfigError(
                f"KAFKA_PRODUCER_COMPRESSION={self.compression_type!r} must be one of "
                f"{', '.join(sorted(self.VALID_COMPRESSION))}"
            )
        if self.acks not in self.VALID_ACKS:
            raise ConfigError(
                f"KAFKA_PRODUCER_ACKS={self.acks!r} must be '0', '1' or 'all'"
            )
        if self.retry_backoff_max_seconds < self.retry_backoff_seconds:
            raise ConfigError(
                "KAFKA_RETRY_BACKOFF_MAX_SECONDS must be >= KAFKA_RETRY_BACKOFF_SECONDS"
            )

    @property
    def bootstrap_servers_string(self) -> str:
        """Return bootstrap servers in the comma-separated form Spark expects.

        Returns:
            For example ``"localhost:9092"`` or ``"a:9092,b:9092"``.
        """
        return ",".join(self.bootstrap_servers)

    @classmethod
    def from_env(cls) -> KafkaConfig:
        """Build a :class:`KafkaConfig` from the environment.

        Returns:
            A validated, immutable Kafka configuration.
        """
        return cls(
            bootstrap_servers=_env_csv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            transactions_topic=_env_str("KAFKA_TRANSACTIONS_TOPIC", "transactions"),
            alerts_topic=_env_str("KAFKA_ALERTS_TOPIC", "fraud-alerts"),
            dlq_topic=_env_str("KAFKA_DLQ_TOPIC", "transactions-dlq"),
            transactions_partitions=_env_int(
                "KAFKA_TRANSACTIONS_PARTITIONS", 6, minimum=1
            ),
            alerts_partitions=_env_int("KAFKA_ALERTS_PARTITIONS", 3, minimum=1),
            consumer_group=_env_str("KAFKA_CONSUMER_GROUP", "fraud-stream-job"),
            linger_ms=_env_int("KAFKA_PRODUCER_LINGER_MS", 10, minimum=0),
            batch_size=_env_int("KAFKA_PRODUCER_BATCH_SIZE", 65536, minimum=1024),
            compression_type=_env_str("KAFKA_PRODUCER_COMPRESSION", "lz4").lower(),
            acks=_env_str("KAFKA_PRODUCER_ACKS", "1").lower(),
            max_connect_retries=_env_int("KAFKA_MAX_CONNECT_RETRIES", 8, minimum=1),
            retry_backoff_seconds=_env_float(
                "KAFKA_RETRY_BACKOFF_SECONDS", 1.0, minimum=0.01
            ),
            retry_backoff_max_seconds=_env_float(
                "KAFKA_RETRY_BACKOFF_MAX_SECONDS", 30.0, minimum=0.01
            ),
        )


@dataclass(frozen=True, slots=True)
class CassandraConfig:
    """Cassandra connection, keyspace and write-path settings."""

    contact_points: tuple[str, ...]
    port: int
    keyspace: str
    datacenter: str
    username: str | None
    password: str | None
    write_batch_size: int
    max_write_retries: int
    request_timeout_seconds: float
    raw_ttl_seconds: int

    #: Upper bound on statements per unlogged batch. Cassandra warns above 100
    #: and rejects above 1000; 500 keeps a safety margin.
    MAX_WRITE_BATCH_SIZE: ClassVar[int] = 500

    def __post_init__(self) -> None:
        """Validate credential pairing and batch sizing.

        Raises:
            ConfigError: If only one half of the credential pair is supplied, or
                the batch size exceeds Cassandra's practical limit.
        """
        if (self.username is None) != (self.password is None):
            raise ConfigError(
                "CASSANDRA_USERNAME and CASSANDRA_PASSWORD must be set together"
            )
        if self.write_batch_size > self.MAX_WRITE_BATCH_SIZE:
            raise ConfigError(
                f"CASSANDRA_WRITE_BATCH_SIZE={self.write_batch_size} is too large; "
                f"keep it <= {self.MAX_WRITE_BATCH_SIZE} to avoid coordinator "
                "batch-size warnings"
            )

    @property
    def auth_required(self) -> bool:
        """Whether credentials were supplied.

        Returns:
            ``True`` when both username and password are configured.
        """
        return self.username is not None and self.password is not None

    @classmethod
    def from_env(cls) -> CassandraConfig:
        """Build a :class:`CassandraConfig` from the environment.

        Returns:
            A validated, immutable Cassandra configuration.
        """
        return cls(
            contact_points=_env_csv("CASSANDRA_CONTACT_POINTS", "localhost"),
            port=_env_int("CASSANDRA_PORT", 9042, minimum=1),
            keyspace=_env_str("CASSANDRA_KEYSPACE", "fraud_detection"),
            datacenter=_env_str("CASSANDRA_DATACENTER", "datacenter1"),
            username=_env_optional_str("CASSANDRA_USERNAME"),
            password=_env_optional_str("CASSANDRA_PASSWORD"),
            write_batch_size=_env_int("CASSANDRA_WRITE_BATCH_SIZE", 100, minimum=1),
            max_write_retries=_env_int("CASSANDRA_MAX_WRITE_RETRIES", 5, minimum=1),
            request_timeout_seconds=_env_float(
                "CASSANDRA_REQUEST_TIMEOUT_SECONDS", 15.0, minimum=0.1
            ),
            raw_ttl_seconds=_env_int("CASSANDRA_RAW_TTL_SECONDS", 7_776_000, minimum=0),
        )


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Log verbosity and output format."""

    level: str
    format: str

    #: Output formats understood by :mod:`common.logging_config`.
    VALID_FORMATS: ClassVar[frozenset[str]] = frozenset({"text", "json"})
    #: Standard library log level names.
    VALID_LEVELS: ClassVar[frozenset[str]] = frozenset(
        {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    )

    def __post_init__(self) -> None:
        """Validate the level and format names.

        Raises:
            ConfigError: If either value is unrecognised.
        """
        if self.level not in self.VALID_LEVELS:
            raise ConfigError(
                f"LOG_LEVEL={self.level!r} must be one of "
                f"{', '.join(sorted(self.VALID_LEVELS))}"
            )
        if self.format not in self.VALID_FORMATS:
            raise ConfigError(f"LOG_FORMAT={self.format!r} must be 'text' or 'json'")

    @classmethod
    def from_env(cls) -> LoggingConfig:
        """Build a :class:`LoggingConfig` from the environment.

        Returns:
            A validated, immutable logging configuration.
        """
        return cls(
            level=_env_str("LOG_LEVEL", "INFO").upper(),
            format=_env_str("LOG_FORMAT", "text").lower(),
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Aggregate of every configuration section.

    Components receive the whole object and read the sections they need, which
    keeps function signatures stable as new settings are added.
    """

    kafka: KafkaConfig
    cassandra: CassandraConfig
    logging: LoggingConfig

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build the full application configuration from the environment.

        Returns:
            A validated, immutable configuration tree.
        """
        return cls(
            kafka=KafkaConfig.from_env(),
            cassandra=CassandraConfig.from_env(),
            logging=LoggingConfig.from_env(),
        )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the process-wide configuration, loading ``.env`` on first call.

    Cached so that repeated calls are free and every caller observes exactly the
    same values. Tests that need different settings should patch the environment
    and then call ``get_config.cache_clear()``.

    Returns:
        The shared :class:`AppConfig` instance.
    """
    load_dotenv()
    return AppConfig.from_env()
