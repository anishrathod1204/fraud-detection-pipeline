"""Prometheus metrics bootstrap shared by the producer and the streaming job.

Scrape model
------------
Prometheus *pulls*. Each component therefore runs a tiny HTTP server on its own
port (``PRODUCER_METRICS_PORT``, ``STREAMING_METRICS_PORT``) and Prometheus
scrapes it every five seconds, as configured in
``infra/prometheus/prometheus.yml``. A push gateway would be the wrong tool
here: these are long-lived processes, not batch jobs, and the gateway's stale
metrics would misreport a dead producer as a healthy one.

Because the stack runs in Docker while the producer and streaming job run on the
host, the Prometheus config reaches them via ``host.docker.internal``. Both
targets read DOWN until their process is started - that is expected, not a
misconfiguration.

Failure policy
--------------
Metrics are observability, not function. If the exporter cannot bind its port -
almost always a previous run still holding it - the component logs a warning and
carries on unmetered rather than refusing to process transactions. A fraud
pipeline that stops detecting fraud because a dashboard is unavailable has its
priorities backwards.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from prometheus_client import REGISTRY, Gauge, Histogram, start_http_server

from common.logging_config import get_logger

__all__ = [
    "LATENCY_BUCKETS",
    "METRIC_PREFIX",
    "component_up",
    "start_metrics_server",
    "timed",
]

_LOGGER = get_logger(__name__)

#: Shared prefix for every metric in this project, so a Grafana query can select
#: the whole pipeline with ``{__name__=~"fraud_.*"}``.
METRIC_PREFIX: Final[str] = "fraud"

#: Bucket boundaries, in seconds, for publish and scoring latency. Weighted
#: towards the sub-10ms range where a local Kafka round trip actually lands;
#: the default prometheus_client buckets start at 5ms and would collapse almost
#: every observation into the first bucket, making p50 meaningless.
LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)

#: Liveness gauge, one time series per component. Prometheus's own ``up`` metric
#: already reports scrape reachability, but this distinguishes "the process is
#: running and considers itself healthy" from "the port answered".
_COMPONENT_UP: Final[Gauge] = Gauge(
    f"{METRIC_PREFIX}_component_up",
    "1 while the component is running and healthy, 0 during shutdown.",
    labelnames=("component",),
)


def component_up(component: str, *, healthy: bool = True) -> None:
    """Set the liveness gauge for a component.

    Args:
        component: Short component name, for example ``"producer"``.
        healthy: ``True`` at startup, ``False`` on graceful shutdown so a
            deliberate stop is distinguishable from a crash on the dashboard.
    """
    _COMPONENT_UP.labels(component=component).set(1.0 if healthy else 0.0)


def start_metrics_server(port: int, *, component: str) -> bool:
    """Start the Prometheus scrape endpoint for this process.

    Args:
        port: TCP port to listen on. Must be free; see the module docstring for
            what happens when it is not.
        component: Short component name, used for the liveness gauge and logs.

    Returns:
        ``True`` if the endpoint is serving, ``False`` if the port could not be
        bound. Callers should continue either way.
    """
    try:
        start_http_server(port, registry=REGISTRY)
    except OSError as exc:
        _LOGGER.warning(
            "metrics endpoint unavailable, continuing without metrics",
            extra={"component": component, "port": port, "error": str(exc)},
        )
        return False

    component_up(component, healthy=True)
    _LOGGER.info(
        "metrics endpoint listening",
        extra={"component": component, "port": port, "path": "/metrics"},
    )
    return True


@contextmanager
def timed(histogram: Histogram) -> Iterator[None]:
    """Record the duration of a block into a histogram.

    Equivalent to ``histogram.time()`` but built on
    :func:`time.perf_counter`, which is monotonic. The stock decorator uses the
    wall clock, so an NTP correction mid-run can record a negative duration and
    corrupt the bucket counts for the lifetime of the process.

    Args:
        histogram: Histogram to observe into.

    Yields:
        ``None``. The elapsed time is recorded on exit, including when the block
        raises - a failed publish still took time and hiding it would flatter
        the latency chart.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - started)
