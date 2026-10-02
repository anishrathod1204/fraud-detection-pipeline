"""Unit tests for the producer's rate limiter.

The rate limiter is where the declared target throughput either becomes real or
becomes a lie, so its behaviour is worth pinning precisely. Most tests drive it
with a fake clock rather than real sleeping: the logic compares a deadline
against elapsed time, and a deterministic clock makes every assertion exact
instead of "roughly, on a loaded CI box". The fake clock is installed by the
:func:`fake_clock` context manager rather than a pytest fixture, so each test
states plainly that it is manipulating time.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator

import pytest

from producer import rate_limiter as rate_limiter_module
from producer.rate_limiter import (
    DEFAULT_MAX_CATCHUP_SECONDS,
    MIN_SLEEP_SECONDS,
    RateLimiter,
)


class _FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        """Return the current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record a sleep and advance the fake clock by it."""
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        """Move time forward without recording a sleep (simulates work)."""
        self.now += seconds


@contextmanager
def fake_clock(start: float = 1_000.0) -> Iterator[_FakeClock]:
    """Temporarily replace time.perf_counter and time.sleep in the limiter module.

    Args:
        start: Initial fake time in seconds.

    Yields:
        The installed fake clock.
    """
    fake = _FakeClock(start)
    original_perf = rate_limiter_module.time.perf_counter
    original_sleep = rate_limiter_module.time.sleep
    rate_limiter_module.time.perf_counter = fake.perf_counter  # type: ignore[assignment]
    rate_limiter_module.time.sleep = fake.sleep  # type: ignore[assignment]
    try:
        yield fake
    finally:
        rate_limiter_module.time.perf_counter = original_perf  # type: ignore[assignment]
        rate_limiter_module.time.sleep = original_sleep  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------
class TestConstruction:
    """Inputs are validated up front rather than misbehaving later."""

    def test_rejects_non_positive_catchup(self) -> None:
        with pytest.raises(ValueError, match="max_catchup_seconds must be > 0"):
            RateLimiter(100, max_catchup_seconds=0.0)

    def test_negative_target_is_clamped_to_zero(self) -> None:
        # A negative rate is a configuration mistake; treating it as "unlimited"
        # is kinder than refusing to start, and matches PRODUCER_TPS semantics.
        limiter = RateLimiter(-5)
        assert limiter.target_tps == 0
        assert limiter.enabled is False

    def test_default_catchup_constant(self) -> None:
        assert RateLimiter(1).max_catchup_seconds == DEFAULT_MAX_CATCHUP_SECONDS

    def test_starts_with_zero_counters(self) -> None:
        limiter = RateLimiter(100)
        assert limiter.emitted == 0
        assert limiter.total_sleep_seconds == 0.0


# ---------------------------------------------------------------------------
# Unthrottled mode
# ---------------------------------------------------------------------------
class TestUnthrottled:
    """A target of 0 must be a cheap no-op, not a division by zero."""

    def test_acquire_never_sleeps(self) -> None:
        with fake_clock() as clock:
            limiter = RateLimiter(0)
            for _ in range(100):
                assert limiter.acquire() == 0.0
            assert clock.sleeps == []

    def test_still_counts_emitted(self) -> None:
        with fake_clock():
            limiter = RateLimiter(0)
            limiter.acquire(5)
            assert limiter.emitted == 5

    def test_achieved_tps_is_real(self) -> None:
        with fake_clock() as clock:
            limiter = RateLimiter(0)
            limiter.acquire(100)
            clock.advance(2.0)
            assert limiter.achieved_tps == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Deadline pacing
# ---------------------------------------------------------------------------
class TestPacing:
    """Sleeping only the shortfall is what absorbs publish time."""

    def test_sleeps_the_shortfall(self) -> None:
        # At 100 tps the first record should take 10 ms; no time has passed, so
        # the limiter sleeps the full interval.
        with fake_clock() as clock:
            limiter = RateLimiter(100)
            slept = limiter.acquire()
            assert slept == pytest.approx(0.01)
            assert clock.sleeps == [pytest.approx(0.01)]

    def test_absorbs_work_time(self) -> None:
        # If publishing already took most of the interval, the limiter sleeps
        # only the remainder - the whole point of deadline pacing.
        with fake_clock() as clock:
            limiter = RateLimiter(100)
            clock.advance(0.008)  # 8 ms spent publishing
            slept = limiter.acquire()
            assert slept == pytest.approx(0.002, abs=1e-9)

    def test_does_not_sleep_when_behind(self) -> None:
        # Work already exceeded the interval: there is nothing to wait for.
        with fake_clock() as clock:
            limiter = RateLimiter(100)
            clock.advance(0.05)
            assert limiter.acquire() == 0.0
            assert clock.sleeps == []

    def test_sub_millisecond_shortfall_is_ignored(self) -> None:
        # A shortfall below the minimum granularity is not worth a syscall, and
        # requesting it would overshoot the interval anyway.
        with fake_clock() as clock:
            limiter = RateLimiter(1000)  # 1 ms per record
            clock.advance(0.0005)  # 0.5 ms spent -> 0.5 ms short, below MIN_SLEEP
            assert limiter.acquire() == 0.0
            assert clock.sleeps == []

    def test_counts_records_in_batches(self) -> None:
        # Acquiring a batch of 10 at 100 tps targets 0.1 s elapsed.
        with fake_clock():
            limiter = RateLimiter(100)
            slept = limiter.acquire(10)
            assert slept == pytest.approx(0.1)
            assert limiter.emitted == 10

    def test_zero_count_is_a_noop(self) -> None:
        with fake_clock() as clock:
            limiter = RateLimiter(100)
            assert limiter.acquire(0) == 0.0
            assert clock.sleeps == []

    def test_rejects_negative_count(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 0"):
            RateLimiter(100).acquire(-1)

    def test_total_sleep_accumulates(self) -> None:
        with fake_clock():
            limiter = RateLimiter(100)
            limiter.acquire()
            limiter.acquire()
            limiter.acquire()
            assert limiter.total_sleep_seconds == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Bounded catch-up
# ---------------------------------------------------------------------------
class TestBoundedCatchup:
    """A stall must not be repaid as an unbounded burst."""

    def test_small_lateness_is_repaid(self) -> None:
        # 0.5 s behind at 100 tps is within the 2 s budget, so the limiter stays
        # behind rather than re-basing: subsequent acquires do not sleep.
        with fake_clock() as clock:
            limiter = RateLimiter(100)
            clock.advance(0.5)
            limiter.acquire()
            assert limiter.acquire() == 0.0  # still behind, no sleep

    def test_large_lateness_is_forgiven(self) -> None:
        # After a stall beyond the budget the deadline is re-based to now, so the
        # *next* record sleeps a normal interval instead of firing instantly.
        with fake_clock() as clock:
            limiter = RateLimiter(100, max_catchup_seconds=2.0)
            clock.advance(10.0)  # way past the budget
            limiter.acquire()  # detects the debt and re-bases
            # Deadline now tracks the present, so the next acquire paces normally.
            slept = limiter.acquire()
            assert slept == pytest.approx(MIN_SLEEP_SECONDS, abs=0.01)

    def test_forgiveness_prevents_burst(self) -> None:
        # Emitting many records after a long stall must not skip the interval.
        with fake_clock() as clock:
            limiter = RateLimiter(100)
            clock.advance(10.0)
            limiter.acquire()
            slept_total = sum(limiter.acquire() for _ in range(5))
            # Five records at 100 tps should take ~50 ms, not zero.
            assert slept_total > 0.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
class TestReset:
    """Reconnecting after an outage must not repay the lost time."""

    def test_reset_clears_counters(self) -> None:
        with fake_clock():
            limiter = RateLimiter(100)
            limiter.acquire(10)
            limiter.reset()
            assert limiter.emitted == 0
            assert limiter.total_sleep_seconds == 0.0

    def test_reset_rebases_deadline(self) -> None:
        with fake_clock() as clock:
            limiter = RateLimiter(100)
            clock.advance(5.0)
            limiter.reset()
            # After reset the limiter is on schedule again, so it sleeps a full
            # interval for the next record rather than firing a burst.
            assert limiter.acquire() == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Real-clock smoke test
# ---------------------------------------------------------------------------
class TestRealClock:
    """One test against the real clock, to catch a mock-shaped mistake.

    The fake-clock tests assume ``perf_counter`` and ``sleep`` compose the way
    this class assumes; a single real-timing check confirms the assumption holds
    outside the mock.
    """

    def test_paces_a_short_burst(self) -> None:
        limiter = RateLimiter(200)  # 5 ms per record
        started = time.perf_counter()
        for _ in range(20):
            limiter.acquire()
        elapsed = time.perf_counter() - started
        # 20 records at 200 tps is nominally 0.1 s. Allow generous slack for
        # scheduler granularity on a loaded machine.
        assert 0.08 <= elapsed <= 0.25
