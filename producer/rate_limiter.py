"""Rate limiting for the simulated transaction feed.

Why not ``sleep(1 / tps)`` per message
--------------------------------------
The obvious implementation sleeps a fixed slice after every message. It is
wrong for two independent reasons:

1. **Sleep granularity.** ``time.sleep`` is accurate to roughly a millisecond at
   best. At 5,000 tps the per-message slice is 0.2 ms, so every sleep overshoots
   by several multiples and the achieved rate lands far below target - the load
   test would measure the scheduler, not Kafka.
2. **Drift accumulates.** The time spent serialising and publishing is not
   counted, so the real interval is ``send_time + sleep_time`` and the feed runs
   permanently slow, by a margin that varies with broker latency.

This module paces against a *deadline* instead: it tracks how much time
*should* have elapsed for the number of records emitted so far and sleeps only
the shortfall, and only when that shortfall is large enough to be worth asking
the OS for. Publish time is absorbed automatically, and at high target rates
the limiter simply stops sleeping, which is the correct behaviour.

Catch-up is bounded
-------------------
If the producer stalls - a broker reconnect, a long GC pause - deadline pacing
would subsequently allow an unbounded burst to "repay" the lost time. A burst of
tens of thousands of messages is not a useful simulation of a payment feed and
would distort every latency percentile downstream. After
:attr:`RateLimiter.max_catchup_seconds` of accumulated lateness the limiter
forgives the debt and paces forward from the present instead.
"""

from __future__ import annotations

import time
from typing import Final

from common.logging_config import get_logger

__all__ = ["DEFAULT_MAX_CATCHUP_SECONDS", "MIN_SLEEP_SECONDS", "RateLimiter"]

_LOGGER = get_logger(__name__)

#: Shortest sleep worth requesting. Below roughly a millisecond the call costs
#: more than it delays, and the overshoot exceeds the interval being targeted.
MIN_SLEEP_SECONDS: Final[float] = 0.001

#: How far behind schedule the limiter may fall before it stops trying to catch
#: up. Two seconds is long enough to absorb a Kafka metadata refresh or a
#: rebalance without re-baselining, and short enough that the burst it permits
#: stays within one Spark micro-batch.
DEFAULT_MAX_CATCHUP_SECONDS: Final[float] = 2.0


class RateLimiter:
    """Paces a loop to a target rate using deadline scheduling.

    Not thread-safe. The producer is single-threaded by design - ordering within
    an account partition matters - so locking would add cost for no benefit.

    Attributes:
        target_tps: Records per second to aim for; 0 disables limiting.
        max_catchup_seconds: Lateness budget before the deadline is re-based.
    """

    __slots__ = (
        "_emitted",
        "_pace_origin",
        "_sleep_total",
        "_wall_origin",
        "max_catchup_seconds",
        "target_tps",
    )

    def __init__(
        self,
        target_tps: int,
        *,
        max_catchup_seconds: float = DEFAULT_MAX_CATCHUP_SECONDS,
    ) -> None:
        """Initialise the limiter and start its clock.

        Args:
            target_tps: Target records per second. Zero or negative means
                unthrottled, in which case :meth:`acquire` is a cheap no-op.
            max_catchup_seconds: Maximum accumulated lateness to try to repay.

        Raises:
            ValueError: If ``max_catchup_seconds`` is not positive.
        """
        if max_catchup_seconds <= 0.0:
            raise ValueError(
                f"max_catchup_seconds must be > 0, got {max_catchup_seconds}"
            )

        self.target_tps = max(0, target_tps)
        self.max_catchup_seconds = max_catchup_seconds
        # perf_counter is monotonic; the wall clock is not, and a mid-run NTP
        # step would otherwise make the pacer sleep for hours or not at all.
        now = time.perf_counter()
        #: Fixed origin for throughput reporting; never re-based.
        self._wall_origin = now
        #: Origin the deadline is measured from; moves when debt is forgiven.
        self._pace_origin = now
        self._emitted = 0
        self._sleep_total = 0.0

    @property
    def enabled(self) -> bool:
        """Whether rate limiting is in effect.

        Returns:
            ``False`` when the target rate is 0, meaning run as fast as possible.
        """
        return self.target_tps > 0

    @property
    def emitted(self) -> int:
        """Total records counted through :meth:`acquire`.

        Returns:
            The cumulative record count.
        """
        return self._emitted

    @property
    def total_sleep_seconds(self) -> float:
        """Cumulative time spent sleeping.

        A value near zero while the target rate is not being met means the
        producer is saturated rather than throttled - the distinction the load
        test exists to establish.

        Returns:
            Seconds slept since construction.
        """
        return self._sleep_total

    @property
    def achieved_tps(self) -> float:
        """Average records per second since construction.

        Returns:
            Records divided by elapsed wall time, or 0.0 before the first
            measurable interval.
        """
        elapsed = time.perf_counter() - self._wall_origin
        return self._emitted / elapsed if elapsed > 0.0 else 0.0

    def acquire(self, count: int = 1) -> float:
        """Count records as emitted and sleep if ahead of schedule.

        Call *after* publishing, so that publish time counts towards the
        interval rather than being added to it.

        Args:
            count: Number of records just emitted. Values above 1 let a caller
                that publishes in batches pace the batch as a unit.

        Returns:
            Seconds actually slept; 0.0 when on or behind schedule.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")

        self._emitted += count
        if not self.enabled or count == 0:
            return 0.0

        # Time this many records *should* have taken, versus what they did take.
        target_elapsed = self._emitted / self.target_tps
        actual_elapsed = time.perf_counter() - self._pace_origin
        shortfall = target_elapsed - actual_elapsed

        if shortfall >= MIN_SLEEP_SECONDS:
            time.sleep(shortfall)
            self._sleep_total += shortfall
            return shortfall

        if -shortfall > self.max_catchup_seconds:
            # Too far behind to repay without a burst. Re-base the deadline so
            # the next interval is paced from now, and say so: sustained
            # re-basing is the signal that the target rate is unreachable.
            _LOGGER.warning(
                "rate limiter behind schedule, forgiving accumulated debt",
                extra={
                    "target_tps": self.target_tps,
                    "achieved_tps": round(self.achieved_tps, 1),
                    "lateness_seconds": round(-shortfall, 3),
                },
            )
            self._pace_origin = time.perf_counter() - target_elapsed

        return 0.0

    def reset(self) -> None:
        """Restart the clock and counters.

        Used when the producer reconnects after a broker outage: the records it
        failed to publish during the outage should not be repaid as a burst.
        """
        now = time.perf_counter()
        self._wall_origin = now
        self._pace_origin = now
        self._emitted = 0
        self._sleep_total = 0.0
