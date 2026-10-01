"""Shared library for the fraud detection pipeline.

Everything in this package is imported by at least two of the four runtime
components (producer, training, streaming, dashboard). Keeping configuration,
schema and feature engineering here is what prevents training/serving skew:
there is exactly one implementation of each, not one per component.
"""

from __future__ import annotations

__version__ = "1.0.0"
