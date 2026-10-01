"""Kafka transaction producer.

Replays the PaySim dataset onto the ``transactions`` topic at a controlled rate,
standing in for the real-time feed a bank would have. Messages are keyed by
originating account so that all activity for one account lands on a single
partition - a property the streaming job's per-account velocity features depend
on for correctness.
"""

from __future__ import annotations

__all__: list[str] = []
