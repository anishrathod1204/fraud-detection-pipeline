"""Shared test fixtures and dependency stubs.

The pipeline's third-party dependencies - kafka-python, prometheus_client,
pyspark, the Cassandra driver - are heavy and are not needed to exercise the
distributed *logic*. Two things are arranged here instead:

1. Lightweight stubs for the Kafka client and the Prometheus client are
   installed into :data:`sys.modules` when the real packages are absent, so the
   producer modules can be imported and their logic tested. When the real
   packages *are* installed (a full dev environment, CI), the stubs are not used
   and the tests run against the genuine libraries.
2. The repository root is placed on ``sys.path`` so tests can ``import common``
   and ``import producer`` without an editable install.

This is a deliberate tradeoff, stated plainly so it is not mistaken for hiding a
missing dependency: the stubs let the pure logic be verified anywhere, while the
integration tests in ``tests/test_integration.py`` are marked to require the
real stack and are skipped when it is absent.

The stub installers are plain functions, importable and callable without pytest,
so the suite can also be driven by a minimal runner when pytest itself is
unavailable. The pytest fixtures at the bottom are conditionally defined.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def module_available(name: str) -> bool:
    """Return whether a module can be imported.

    Args:
        name: Dotted module name.

    Returns:
        ``True`` when a spec is found, ``False`` otherwise.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install_prometheus_stub() -> None:
    """Install a minimal prometheus_client replacement if the real one is absent."""
    if module_available("prometheus_client"):
        return

    module = types.ModuleType("prometheus_client")

    class _Metric:
        """Records only the last value; enough to assert on increments."""

        def __init__(self, name: str, *_: Any, **__: Any) -> None:
            self.name = name
            self.value = 0.0
            self.observations: list[float] = []

        def labels(self, **_: Any) -> "_Metric":
            return self

        def set(self, value: float) -> None:
            self.value = value

        def inc(self, amount: float = 1.0) -> None:
            self.value += amount

        def observe(self, value: float) -> None:
            self.observations.append(value)

    class Counter(_Metric):
        pass

    class Gauge(_Metric):
        pass

    class Histogram(_Metric):
        pass

    module.Counter = Counter  # type: ignore[attr-defined]
    module.Gauge = Gauge  # type: ignore[attr-defined]
    module.Histogram = Histogram  # type: ignore[attr-defined]
    module.REGISTRY = object()  # type: ignore[attr-defined]
    module.start_http_server = lambda *_, **__: None  # type: ignore[attr-defined]
    sys.modules["prometheus_client"] = module


def install_kafka_stub() -> None:
    """Install a minimal kafka-python replacement if the real one is absent.

    The stub models the behaviours the publisher depends on: ``send`` returns a
    future whose ``add_errback`` fires immediately when a delivery error is baked
    in, and the client applies the ``key_serializer`` passed by the publisher. A
    test may set ``send_error`` on an instance (or replace ``KafkaProducer``
    entirely) to exercise the error paths.
    """
    if module_available("kafka"):
        return

    module = types.ModuleType("kafka")
    errors = types.ModuleType("kafka.errors")

    class _Future:
        """Mimics kafka-python's future: optional error, no-op callbacks."""

        def __init__(self, error: BaseException | None = None) -> None:
            self._error = error
            self.callbacks: list[Any] = []

        def add_errback(self, callback: Any) -> None:
            if self._error is not None and callback is not None:
                callback(self._error)

        def add_callback(self, callback: Any) -> None:
            self.callbacks.append(callback)

    class KafkaProducer:
        """Records sends; applies the key serializer like the real client."""

        def __init__(self, *_: Any, **kwargs: Any) -> None:
            self._key_serializer = kwargs.get("key_serializer")
            self.acks = kwargs.get("acks")
            self.sent: list[tuple[str, bytes | None, bytes]] = []
            self.flush_calls = 0
            self.closed = False
            self.send_error: BaseException | None = None

        def send(self, topic: str, key: Any = None, value: Any = None) -> _Future:
            if self.send_error is not None:
                raise self.send_error
            serialized_key = (
                self._key_serializer(key)
                if key is not None and self._key_serializer
                else key
            )
            self.sent.append((topic, serialized_key, value))
            return _Future()

        def flush(self, timeout: float | None = None) -> None:
            self.flush_calls += 1

        def close(self, timeout: float | None = None) -> None:
            self.closed = True

    class NoBrokersAvailable(Exception):
        pass

    class KafkaError(Exception):
        pass

    class KafkaTimeoutError(KafkaError):
        pass

    module.KafkaProducer = KafkaProducer  # type: ignore[attr-defined]
    module._Future = _Future  # type: ignore[attr-defined]
    errors.NoBrokersAvailable = NoBrokersAvailable  # type: ignore[attr-defined]
    errors.KafkaError = KafkaError  # type: ignore[attr-defined]
    errors.KafkaTimeoutError = KafkaTimeoutError  # type: ignore[attr-defined]
    module.errors = errors  # type: ignore[attr-defined]
    sys.modules["kafka"] = module
    sys.modules["kafka.errors"] = errors


# Install stubs at import time, before any test module imports producer code.
install_prometheus_stub()
install_kafka_stub()


# ---------------------------------------------------------------------------
# pytest fixtures (defined only when pytest is present)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

if pytest is not None:

    @pytest.fixture()
    def kafka_module() -> Any:
        """Return the (real or stubbed) kafka module.

        Returns:
            The ``kafka`` module in :data:`sys.modules`.
        """
        return sys.modules["kafka"]

    @pytest.fixture()
    def sample_record() -> dict[str, Any]:
        """Return one well-formed PaySim record.

        Returns:
            A dict matching :data:`common.schema.PAYSIM_COLUMNS`.
        """
        return {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C1305486145",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C553264065",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
