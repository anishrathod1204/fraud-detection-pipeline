"""Unit tests for the PaySim-faithful synthetic dataset generator.

The generator stands in for a 470 MB Kaggle download, and the *statistical*
properties it promises are exactly what the rest of the pipeline is built on:
that fraud is scarce and confined to two transaction types, that balances
reconcile, and that a given seed reproduces a given file. None of those are
visible from the schema alone, so they are pinned here rather than trusted.

The tests are deliberately small - a few thousand rows each - because every
property under test is structural, not statistical-precision, and a large frame
would only slow the suite down. Where a property *is* statistical (fraud
prevalence, merchant balances) the assertion is a range or an invariant rather
than an exact value, since the whole point is that the output is random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.schema import FRAUD_CAPABLE_TYPES, PAYSIM_COLUMNS
from scripts.generate_dataset import (
    FLAG_THRESHOLD_AMOUNT,
    GenerationError,
    GeneratorParams,
    generate,
)

#: Two-decimal rounding on each of two operands gives a worst-case 0.01 of
#: slack on a reconciled difference; anything above this is a real error.
_RECON_TOL = 0.02


def make_frame(rows: int = 4000, *, seed: int = 42, steps: int = 120) -> pd.DataFrame:
    """Generate a small frame for the structural assertions.

    Args:
        rows: Number of rows.
        seed: RNG seed.
        steps: Number of simulated hours.

    Returns:
        A DataFrame with the canonical PaySim columns.
    """
    return generate(
        GeneratorParams(
            rows=rows, steps=steps, fraud_rate=0.00129, seed=seed, chunk_rows=rows
        )
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class TestSchema:
    """The output must be indistinguishable from a real PaySim export, shape-wise."""

    def test_columns_and_order(self) -> None:
        assert list(make_frame().columns) == list(PAYSIM_COLUMNS)

    def test_row_count(self) -> None:
        assert len(make_frame(rows=1234)) == 1234

    def test_types_are_known(self) -> None:
        frame = make_frame()
        assert set(frame["type"].unique()).issubset(
            {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
        )

    def test_label_columns_are_binary(self) -> None:
        frame = make_frame()
        for column in ("isFraud", "isFlaggedFraud"):
            assert set(frame[column].unique()).issubset({0, 1})

    def test_no_negative_money(self) -> None:
        frame = make_frame()
        money = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        assert bool((frame[money] >= 0).all().all())


# ---------------------------------------------------------------------------
# Fraud structure
# ---------------------------------------------------------------------------
class TestFraudStructure:
    """Fraud is scarce, and only where the schema says it can be."""

    def test_fraud_only_in_capable_types(self) -> None:
        frame = make_frame()
        observed = set(frame.loc[frame["isFraud"] == 1, "type"].unique())
        assert observed <= FRAUD_CAPABLE_TYPES

    def test_fraud_is_scarce(self) -> None:
        # Not a precision test: just that the imbalance is real and not a
        # convenient 5%.
        frame = make_frame(rows=10000)
        rate = float(frame["isFraud"].mean())
        assert 0.0 < rate < 0.02

    def test_zero_fraud_rate_is_allowed(self) -> None:
        frame = generate(
            GeneratorParams(rows=500, steps=20, fraud_rate=0.0, seed=1, chunk_rows=500)
        )
        assert int(frame["isFraud"].sum()) == 0

    def test_fraud_rate_above_capable_fraction_raises(self) -> None:
        # 0.45 clears the params guard (< 0.5) but exceeds the ~43% of rows that
        # are TRANSFER/CASH_OUT, so the generator must refuse rather than
        # silently under-deliver.
        with pytest.raises(GenerationError, match="TRANSFER or CASH_OUT"):
            generate(
                GeneratorParams(
                    rows=1000, steps=20, fraud_rate=0.45, seed=1, chunk_rows=1000
                )
            )


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
class TestFlaggedFraud:
    """isFlaggedFraud must reproduce PaySim's shipped rule exactly."""

    def test_rule_matches_shipped_definition(self) -> None:
        frame = make_frame()
        expected = ((frame["type"] == "TRANSFER") & (frame["amount"] > FLAG_THRESHOLD_AMOUNT))
        assert bool((frame["isFlaggedFraud"].astype(bool) == expected).all())

    def test_threshold_is_strict(self) -> None:
        # Exactly at the threshold is *not* flagged, matching PaySim's ``>``.
        frame = make_frame()
        at_threshold = frame["amount"] == FLAG_THRESHOLD_AMOUNT
        assert not bool(frame.loc[at_threshold, "isFlaggedFraud"].any())


# ---------------------------------------------------------------------------
# Balance reconciliation
# ---------------------------------------------------------------------------
class TestReconciliation:
    """Balances must move by exactly the amount - the basis of the delta features."""

    def test_origin_balance_moves_by_amount(self) -> None:
        frame = make_frame()
        effect = np.where(frame["type"] == "CASH_IN", frame["amount"], -frame["amount"])
        delta = frame["newbalanceOrig"] - frame["oldbalanceOrg"]
        assert float(np.abs(delta - effect).max()) <= _RECON_TOL

    def test_customer_destination_credited_exactly(self) -> None:
        frame = make_frame()
        credited = frame["type"].isin(["CASH_IN", *FRAUD_CAPABLE_TYPES]) & frame[
            "nameDest"
        ].str.startswith("C")
        delta = (frame["newbalanceDest"] - frame["oldbalanceDest"])[credited]
        assert float(np.abs(delta - frame["amount"][credited]).max()) <= _RECON_TOL

    def test_merchant_destination_balances_are_zero(self) -> None:
        frame = make_frame()
        merchants = frame[frame["nameDest"].str.startswith("M")]
        assert merchants.empty or bool(
            (merchants[["oldbalanceDest", "newbalanceDest"]] == 0).all().all()
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    """A seed must fully determine the output, or the pipeline is not testable."""

    def test_same_seed_same_bytes(self) -> None:
        first = make_frame(seed=99)
        second = make_frame(seed=99)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seed_differs(self) -> None:
        first = make_frame(seed=1)
        second = make_frame(seed=2)
        assert not first["amount"].equals(second["amount"])


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
class TestGeneratorParams:
    """Bad parameters must fail fast, before any allocation."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rows": 0},
            {"steps": 0},
            {"fraud_rate": -0.1},
            {"fraud_rate": 0.6},
            {"chunk_rows": 0},
        ],
    )
    def test_rejects_bad_values(self, kwargs: dict[str, float]) -> None:
        base = {"rows": 100, "steps": 10, "fraud_rate": 0.01, "seed": 1, "chunk_rows": 100}
        base.update(kwargs)
        with pytest.raises(GenerationError):
            GeneratorParams(**base)  # type: ignore[arg-type]
