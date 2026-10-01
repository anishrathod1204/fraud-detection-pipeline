"""Logging setup shared by every component.

Why this exists
---------------
``print`` is unusable in this pipeline. The producer, the Spark driver and the
Spark executors all write to the same terminal, Spark's own JVM logging is
interleaved with Python output, and the only way to tell which process emitted a
line is a structured prefix. Every module therefore calls
:func:`get_logger` and the entrypoints call :func:`configure_logging` exactly
once at startup.

Two formats are supported, selected by ``LOG_FORMAT``:

``text``
    Human-readable, aligned columns, in **local time**, intended for a developer
    watching a terminal.
``json``
    One JSON object per line, in **UTC** with a ``Z`` suffix, intended for log
    shipping. Emitted without any third-party dependency, because adding one to
    the Spark executor environment is a disproportionate cost for a 40-line
    formatter.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Final

from common.config import LoggingConfig, get_config

__all__ = [
    "JsonFormatter",
    "configure_logging",
    "get_logger",
]

#: Root logger name. Component loggers are children (``fraud.producer`` etc.) so
#: that one level setting controls the whole pipeline.
LOGGER_NAMESPACE: Final[str] = "fraud"

#: Third-party loggers that are far too chatty at DEBUG/INFO. kafka-python logs
#: every metadata refresh, the Cassandra driver logs every connection event, and
#: py4j logs every single JVM call - which at DEBUG is thousands of lines per
#: micro-batch.
_NOISY_LOGGERS: Final[dict[str, int]] = {
    "kafka": logging.WARNING,
    "kafka.conn": logging.ERROR,
    "cassandra": logging.WARNING,
    "cassandra.cluster": logging.WARNING,
    "py4j": logging.WARNING,
    "py4j.java_gateway": logging.ERROR,
    "urllib3": logging.WARNING,
    "matplotlib": logging.WARNING,
}

#: Attributes present on every :class:`logging.LogRecord`; anything else was
#: supplied by the caller via ``extra=`` and belongs in the structured output.
_STANDARD_RECORD_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)

#: Guards against double configuration. Spark can import an entrypoint module
#: more than once per process; adding handlers twice duplicates every line.
_configured: bool = False


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Any keyword passed through ``extra=`` is merged into the top level of the
    object, so ``logger.info("published", extra={"count": 500})`` becomes a
    queryable ``count`` field rather than text to be regex-parsed later.
    """

    #: ISO-8601-ish UTC timestamp, millisecond precision.
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a JSON string.

        Args:
            record: The record to format.

        Returns:
            A single-line JSON document terminated by no newline (the handler
            adds one).
        """
        payload: dict[str, Any] = {
            "timestamp": f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}"
            f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Caller-supplied structured context.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # default=str keeps the formatter from raising on values such as
        # Decimal or datetime, which would otherwise lose the log line entirely.
        return json.dumps(payload, default=str, separators=(",", ":"))


def _build_formatter(log_format: str) -> logging.Formatter:
    """Create the formatter for the configured output format.

    Args:
        log_format: Either ``"text"`` or ``"json"``.

    Returns:
        A configured :class:`logging.Formatter`.
    """
    if log_format == "json":
        return JsonFormatter()

    # Fixed-width level and logger columns keep multi-process output scannable.
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-8s %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_logging(
    config: LoggingConfig | None = None, *, force: bool = False
) -> None:
    """Install handlers and levels for the pipeline's logger namespace.

    Safe to call more than once: subsequent calls are no-ops unless ``force`` is
    set. Logs go to stderr rather than stdout so that a component which writes
    machine-readable output to stdout stays pipeable.

    Args:
        config: Logging settings. Defaults to the process configuration.
        force: Reconfigure even if already configured, replacing existing
            handlers. Used by tests.
    """
    global _configured
    if _configured and not force:
        return

    settings = config if config is not None else get_config().logging

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_build_formatter(settings.format))

    root = logging.getLogger(LOGGER_NAMESPACE)
    # Replace rather than append, so `force=True` cannot double up handlers.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.level)
    # Keep pipeline logs out of any handler installed on the true root logger by
    # a library (Spark's Python worker does this), which would duplicate lines.
    root.propagate = False

    for name, level in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for a module.

    Args:
        name: Usually ``__name__``. A leading ``fraud.`` is added when absent so
            that every pipeline logger is a child of :data:`LOGGER_NAMESPACE`
            and inherits its level and handler.

    Returns:
        The configured logger.
    """
    if name == "__main__" or not name:
        qualified = f"{LOGGER_NAMESPACE}.main"
    elif name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        qualified = name
    else:
        qualified = f"{LOGGER_NAMESPACE}.{name}"

    return logging.getLogger(qualified)
