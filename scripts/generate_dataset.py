#!/usr/bin/env python3
"""Generate a PaySim-faithful synthetic dataset.

Why this exists
---------------
The real PaySim file (``paysim.csv``, ~6.36M rows) is distributed on Kaggle and
is not committed here - it is a 470 MB artifact, and shipping it in git would be
both enormous and a licensing question. Everything downstream reads a CSV with
*exactly* the PaySim schema, so this generator produces one that preserves the
statistical structure that makes the pipeline meaningful:

* the eleven columns and their dtypes, in the real file's order;
* **class imbalance**: ``isFraud`` turns up in roughly 0.13% of rows, not a
  convenient 5%, so the recall-first threshold tuning is exercised for real;
* fraud is confined to ``TRANSFER`` and ``CASH_OUT`` - the simulated attack
  drains an account by transfer, then liquidates by cash-out;
* account balances are *reconciled*: after a customer-to-customer transaction
  the origin balance drops by exactly the amount and the destination balance
  rises by exactly it, so the balance-delta features computed downstream carry
  signal rather than noise;
* ``isFlaggedFraud`` reproduces the shipped rule (TRANSFER above 200,000), which
  the naive baseline in :mod:`docs.model_evaluation` is compared against.

The output is large by default (a full 6.36M rows) because the streaming job and
the load test need the volume to be realistic. ``--rows`` scales it down for a
fast local smoke test.

Usage::

    python -m scripts.generate_dataset --rows 6362620 --out data/paysim.csv
    python -m scripts.generate_dataset --rows 200000 --out data/paysim_small.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from common.config import repo_root
from common.logging_config import configure_logging, get_logger
from common.schema import FRAUD_CAPABLE_TYPES, PAYSIM_COLUMNS

__all__ = ["GeneratorParams", "generate", "write_dataset", "main"]

_LOGGER = get_logger(__name__)

#: PaySim's simulated clock: one ``step`` is one hour, and the published dataset
#: spans 744 steps (31 days). Reproduced so the time axis matches the real file.
DEFAULT_STEPS: Final[int] = 744

#: Fraud prevalence in the real PaySim file, to three significant figures.
DEFAULT_FRAUD_RATE: Final[float] = 0.00129

#: The threshold in PaySim's shipped ``isFlaggedFraud`` rule.
FLAG_THRESHOLD_AMOUNT: Final[float] = 200_000.0

#: Transaction-type mix. CASH_OUT and PAYMENT dominate; DEBIT is rare. Fractions
#: are relative weights and are normalised before use, so they may be edited
#: without keeping a running total equal to one.
_TYPE_MIX: Final[dict[str, float]] = {
    "PAYMENT": 0.34,
    "CASH_OUT": 0.35,
    "CASH_IN": 0.22,
    "TRANSFER": 0.08,
    "DEBIT": 0.01,
}

#: Log-normal parameters for transaction amounts. Real amounts are heavy-tailed:
#: a clustering of small payments and a long thin tail into the millions. The
#: median lands near 75,000 and the mean near 180,000, close to the real file.
_AMOUNT_LOG_MEAN: Final[float] = 11.0
_AMOUNT_LOG_SIGMA: Final[float] = 1.8
_AMOUNT_MAX: Final[float] = 10_000_000.0

#: Destination accounts whose id begins ``C`` are customers and carry a real
#: balance history; ``M`` are merchants, which PaySim leaves at zero balance. The
#: prefixes are the only structural distinction between the two account spaces.
_CUSTOMER_PREFIX: Final[str] = "C"
_MERCHANT_PREFIX: Final[str] = "M"

#: Transaction types whose destination is a customer rather than a merchant.
_CUSTOMER_DEST_TYPES: Final[tuple[str, ...]] = ("TRANSFER", "CASH_OUT")


class GenerationError(ValueError):
    """Raised when the requested parameters cannot produce a valid dataset."""


class GeneratorParams:
    """Validated bundle of the generator's knobs.

    A plain class with ``__slots__`` rather than a dataclass because it is built
    once from parsed arguments and never compared or copied; kept separate from
    :mod:`common.config` because it describes the *data*, not the pipeline's
    runtime settings.
    """

    __slots__ = ("rows", "steps", "fraud_rate", "seed", "chunk_rows")

    def __init__(
        self,
        *,
        rows: int,
        steps: int,
        fraud_rate: float,
        seed: int,
        chunk_rows: int,
    ) -> None:
        """Initialise and validate the parameters.

        Args:
            rows: Total number of transactions to generate.
            steps: Number of simulated hourly steps.
            fraud_rate: Target fraction of rows that are fraudulent.
            seed: RNG seed, so a given configuration reproduces byte for byte.
            chunk_rows: Rows buffered before a flush to disk, bounding memory.

        Raises:
            GenerationError: If any parameter is out of range.
        """
        if rows < 1:
            raise GenerationError(f"rows must be >= 1, got {rows}")
        if steps < 1:
            raise GenerationError(f"steps must be >= 1, got {steps}")
        if not 0.0 <= fraud_rate < 0.5:
            raise GenerationError(f"fraud_rate must be in [0, 0.5), got {fraud_rate}")
        if chunk_rows < 1:
            raise GenerationError(f"chunk_rows must be >= 1, got {chunk_rows}")

        self.rows = rows
        self.steps = steps
        self.fraud_rate = fraud_rate
        self.seed = seed
        self.chunk_rows = chunk_rows


def _make_account_ids(rng: np.random.Generator, count: int, prefix: str) -> np.ndarray:
    """Create an array of unique PaySim-style account identifiers.

    The numeric suffixes are an arbitrary sample from the 10-digit space without
    replacement. They are *drawn* by permuting only ``count`` values rather than
    asking the generator to permute the whole 10^10 space, which would try to
    allocate an 80 GB index array.

    Args:
        rng: Seeded generator.
        count: How many identifiers to create.
        prefix: ``"C"`` for customers, ``"M"`` for merchants.

    Returns:
        A ``numpy`` array of ``<U11`` strings such as ``"C0123456789"``.
    """
    # Draw `count` distinct 10-digit suffixes directly, sidestepping the huge
    # permutation array that `rng.choice(10**10, replace=False)` would build.
    suffixes = rng.choice(
        np.iinfo(np.int64).max, size=count, replace=False
    ) % 10_000_000_000
    return np.char.add(prefix, np.char.zfill(suffixes.astype("U11"), 10))


def _sample_amounts(rng: np.random.Generator, count: int) -> np.ndarray:
    """Draw log-normal transaction amounts, rounded to PaySim's precision.

    Args:
        rng: Seeded generator.
        count: Number of amounts.

    Returns:
        Float array of amounts in ``(0, _AMOUNT_MAX]``.
    """
    amounts = rng.lognormal(_AMOUNT_LOG_MEAN, _AMOUNT_LOG_SIGMA, size=count)
    amounts = np.clip(amounts, 1.0, _AMOUNT_MAX)
    return np.round(amounts, 2)


def _assign_types(rng: np.random.Generator, count: int) -> np.ndarray:
    """Assign a transaction type to each row according to :data:`_TYPE_MIX`.

    Args:
        rng: Seeded generator.
        count: Number of rows.

    Returns:
        Object array of type strings.
    """
    types = list(_TYPE_MIX)
    weights = np.array([_TYPE_MIX[t] for t in types], dtype=np.float64)
    weights /= weights.sum()
    return rng.choice(types, size=count, p=weights)


def _walk_balances(
    owner: np.ndarray,
    effect: np.ndarray,
    start: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply signed effects to account balances in the given row order.

    ``owner`` and ``effect`` must already be sorted so that all rows of one
    account are contiguous and in time order. For each row the pre-transaction
    balance is the account's running balance and the post-transaction balance is
    that plus the row's ``effect``.

    Args:
        owner: Account id per row, grouped and time-ordered.
        effect: Signed change to that owner's balance, matching ``owner``.
        start: Opening balance per account id.

    Returns:
        ``(old, new)`` arrays aligned to the input order.
    """
    n = len(owner)
    old = np.empty(n, dtype=np.float64)
    new = np.empty(n, dtype=np.float64)
    if n == 0:
        return old, new

    # Prefix sums give the effect of all rows up to and including each position.
    cum = np.cumsum(effect)
    prev = np.concatenate(([0.0], cum[:-1]))

    is_first = np.empty(n, dtype=bool)
    is_first[0] = True
    is_first[1:] = owner[1:] != owner[:-1]
    starts = np.flatnonzero(is_first)
    group_of = np.cumsum(is_first) - 1
    group_start_prev = prev[starts]

    opening = np.fromiter(
        (start[account] for account in owner[starts]), dtype=np.float64, count=len(starts)
    )
    running_before = opening[group_of] + (prev - group_start_prev[group_of])
    old[:] = running_before
    new[:] = running_before + effect
    return old, new


def generate(params: GeneratorParams) -> pd.DataFrame:
    """Generate a complete dataset in memory.

    For the default 6.36M rows this needs roughly 2-3 GB resident, which is why
    :func:`write_dataset` chunks *across calls*; this function itself builds one
    frame, so it is meant for tests and modest ``--rows`` values.

    The columns are produced in one vectorised pass, with the balance columns
    reconciled afterwards. The reconciliation is the part worth stating precisely:

    1. every customer account opens with a balance large enough to cover its
       outflows, so legitimate rows never go negative;
    2. rows are walked per account in step order, applying the signed amount, so
       ``oldbalanceOrg`` is the balance before the row and ``newbalanceOrig`` the
       balance after it;
    3. a destination that is a customer is credited the same amount in the same
       walk, so ``newbalanceDest - oldbalanceDest == amount`` holds for TRANSFER
       and CASH_OUT, matching the real file's invariant. A merchant destination
       keeps PaySim's structural zero balances.

    Args:
        params: Validated generation parameters.

    Returns:
        A DataFrame with exactly the PaySim columns, in order.

    Raises:
        GenerationError: If the requested fraud rate exceeds the number of
            TRANSFER/CASH_OUT rows available to hold it.
    """
    rng = np.random.default_rng(params.seed)
    n = params.rows

    # Steps cycle so the stream covers the whole time axis regardless of whether
    # params.rows divides params.steps.
    step = (np.arange(n, dtype=np.int64) % params.steps) + 1

    tx_type = _assign_types(rng, n)
    amount = _sample_amounts(rng, n)

    # --- fraud assignment --------------------------------------------------
    is_fraud = np.zeros(n, dtype=np.int8)
    fraud_capable = np.isin(tx_type, list(FRAUD_CAPABLE_TYPES))
    capable_idx = np.flatnonzero(fraud_capable)
    n_fraud = int(round(params.fraud_rate * n))
    if n_fraud > len(capable_idx):
        raise GenerationError(
            f"requested {n_fraud} fraud rows but only {len(capable_idx)} rows are "
            "TRANSFER or CASH_OUT; lower --fraud-rate or raise --rows"
        )
    fraud_positions = (
        rng.choice(capable_idx, size=n_fraud, replace=False) if n_fraud else np.empty(0, np.int64)
    )
    is_fraud[fraud_positions] = 1

    # Fraudulent drains sit at the top of the amount distribution - the attacker
    # empties the account. This also makes isFlaggedFraud a non-trivial baseline:
    # the shipped rule (TRANSFER over 200k) catches most of them.
    if len(fraud_positions):
        high = rng.uniform(0.6, 1.0, size=len(fraud_positions))
        amount[fraud_positions] = np.round(FLAG_THRESHOLD_AMOUNT * (1.0 + 4.0 * high), 2)

    # --- accounts ----------------------------------------------------------
    # Roughly one account per two transactions keeps histories short and the id
    # pool small; the floor guarantees at least a couple of distinct ids.
    n_accounts = max(2, n // 2)
    customer_ids = _make_account_ids(rng, n_accounts, _CUSTOMER_PREFIX)
    merchant_ids = _make_account_ids(rng, max(1, n // 8), _MERCHANT_PREFIX)

    orig_index = rng.integers(0, n_accounts, size=n)
    name_orig = customer_ids[orig_index]

    # The destination is a customer for TRANSFER and CASH_OUT, a merchant
    # otherwise. This mirrors PaySim: money is moved between customers, then
    # cashed out; payments and debits go to merchants.
    dest_is_customer_type = np.isin(tx_type, list(_CUSTOMER_DEST_TYPES))
    name_dest = np.empty(n, dtype=name_orig.dtype)
    name_dest[dest_is_customer_type] = customer_ids[
        rng.integers(0, n_accounts, size=int(dest_is_customer_type.sum()))
    ]
    name_dest[~dest_is_customer_type] = merchant_ids[
        rng.integers(0, len(merchant_ids), size=n - int(dest_is_customer_type.sum()))
    ]

    # --- balances ----------------------------------------------------------
    # Origin effect: an inflow for CASH_IN, an outflow for everything else.
    orig_effect = np.where(tx_type == "CASH_IN", amount, -amount)
    # Destination effect: credits flow in for CASH_IN, TRANSFER and CASH_OUT; a
    # PAYMENT/DEBIT merchant destination is left untouched.
    dest_effect = np.where(
        np.isin(tx_type, ("CASH_IN",) + _CUSTOMER_DEST_TYPES), amount, 0.0
    )

    # Opening balances are derived, not drawn: each account opens at the total
    # magnitude of its own outflows plus a random buffer. That is the tightest
    # bound that provably keeps the running balance non-negative for any arrival
    # order, so the balance never needs clamping - and clamping would break the
    # reconciliation invariant (new = old + signed amount) that the downstream
    # balance-error feature depends on. Deriving the opening sidesteps the
    # problem instead of papering over it.
    outflow = np.where(orig_effect < 0.0, -orig_effect, 0.0)
    total_out = np.zeros(n_accounts, dtype=np.float64)
    np.add.at(total_out, orig_index, outflow)
    buffer = rng.uniform(0.0, float(amount.max()) if n else 1.0, size=n_accounts)
    opening = total_out + np.where(total_out > 0.0, buffer, 0.0)
    # Merchants hold no balance in PaySim, so they are pinned to zero.
    start_balances = dict(zip(customer_ids.tolist(), opening.tolist(), strict=True))
    start_balances.update({mid: 0.0 for mid in merchant_ids.tolist()})

    # Walk origins in (account, step) order so each account's running balance is
    # coherent. lexsort keeps the last key as primary, so sort by account first,
    # then step.
    orig_order = np.lexsort((step, name_orig))
    old_orig_sorted, new_orig_sorted = _walk_balances(
        name_orig[orig_order], orig_effect[orig_order], start_balances
    )
    old_balance_orig = np.empty(n, dtype=np.float64)
    new_balance_orig = np.empty(n, dtype=np.float64)
    old_balance_orig[orig_order] = old_orig_sorted
    new_balance_orig[orig_order] = new_orig_sorted

    # Walk destinations the same way. A merchant destination has no balance in
    # the dataset (its columns are zero), so only customer destinations are
    # walked; the rest default to zero below.
    dest_order = np.lexsort((step, name_dest))
    _, new_dest_sorted = _walk_balances(
        name_dest[dest_order], dest_effect[dest_order], start_balances
    )
    new_balance_dest = np.zeros(n, dtype=np.float64)
    new_balance_dest[dest_order] = new_dest_sorted
    # oldbalanceDest is the pre-transaction balance: the post-balance of that
    # same destination's previous row, or the opening balance for its first row.
    old_balance_dest = new_balance_dest - dest_effect

    # Customers open with a positive balance, so a destination's first row has a
    # meaningful oldbalance; merchants are pinned to zero on both sides.
    merchant_mask = np.char.startswith(name_dest, _MERCHANT_PREFIX)
    old_balance_dest[merchant_mask] = 0.0
    new_balance_dest[merchant_mask] = 0.0

    is_flagged = ((tx_type == "TRANSFER") & (amount > FLAG_THRESHOLD_AMOUNT)).astype(np.int8)

    frame = pd.DataFrame(
        {
            "step": step.astype(np.int32),
            "type": tx_type.astype(str),
            "amount": np.round(amount, 2),
            "nameOrig": name_orig.astype(str),
            "oldbalanceOrg": np.round(old_balance_orig, 2),
            "newbalanceOrig": np.round(new_balance_orig, 2),
            "nameDest": name_dest.astype(str),
            "oldbalanceDest": np.round(old_balance_dest, 2),
            "newbalanceDest": np.round(new_balance_dest, 2),
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )
    return frame[list(PAYSIM_COLUMNS)]


def write_dataset(params: GeneratorParams, out_path: Path) -> dict[str, float]:
    """Generate a dataset and write it to CSV, in memory-bounded chunks.

    Generating 6.36M rows as one frame is fine on a workstation but not on a
    small CI box, so the work is split into chunks of ``params.chunk_rows`` and
    each chunk is appended to the file. Each chunk is generated from its own
    seed, so a chunk's contents do not depend on how the rows were divided; the
    file is reproducible for a given ``(seed, chunk_rows)`` pair.

    Args:
        params: Validated generation parameters.
        out_path: Destination CSV path. Parent directories are created.

    Returns:
        Summary statistics: ``rows``, ``fraud_rows``, ``fraud_rate`` and
        ``flagged_rows``.

    Raises:
        GenerationError: If a chunk cannot satisfy the fraud rate.
        OSError: If the destination cannot be written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    total_fraud = 0
    total_flagged = 0
    total_rows = 0
    header_written = False

    remaining = params.rows
    offset = 0
    while remaining > 0:
        chunk_n = min(params.chunk_rows, remaining)
        # Derive a chunk seed from the base seed and the row offset, so the
        # concatenation is stable and each chunk is independent.
        chunk_params = GeneratorParams(
            rows=chunk_n,
            steps=params.steps,
            fraud_rate=params.fraud_rate,
            seed=params.seed + offset,
            chunk_rows=chunk_n,
        )
        frame = generate(chunk_params)
        frame.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True

        total_rows += len(frame)
        total_fraud += int(frame["isFraud"].sum())
        total_flagged += int(frame["isFlaggedFraud"].sum())
        offset += chunk_n
        remaining -= chunk_n

        _LOGGER.info(
            "wrote chunk",
            extra={"rows_written": total_rows, "target_rows": params.rows},
        )

    return {
        "rows": float(total_rows),
        "fraud_rows": float(total_fraud),
        "fraud_rate": (total_fraud / total_rows) if total_rows else 0.0,
        "flagged_rows": float(total_flagged),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Generate a PaySim-schema synthetic dataset.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6_362_620,
        help="total number of transactions (default: the real PaySim row count)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"number of hourly steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=DEFAULT_FRAUD_RATE,
        help=f"fraction of fraudulent rows (default: {DEFAULT_FRAUD_RATE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="rows generated per in-memory chunk (default: 500000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "data" / "paysim.csv",
        help="output CSV path (default: data/paysim.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a recoverable failure.
    """
    args = build_parser().parse_args(argv)
    configure_logging()

    try:
        params = GeneratorParams(
            rows=args.rows,
            steps=args.steps,
            fraud_rate=args.fraud_rate,
            seed=args.seed,
            chunk_rows=args.chunk_rows,
        )
    except GenerationError as exc:
        _LOGGER.error("invalid parameters", extra={"error": str(exc)})
        return 1

    _LOGGER.info(
        "generating dataset",
        extra={
            "rows": params.rows,
            "steps": params.steps,
            "fraud_rate": params.fraud_rate,
            "out": str(args.out),
        },
    )
    try:
        summary = write_dataset(params, args.out)
    except (OSError, GenerationError) as exc:
        _LOGGER.error("dataset generation failed", extra={"error": str(exc)})
        return 1

    _LOGGER.info(
        "dataset generated",
        extra={
            "out": str(args.out),
            "rows": int(summary["rows"]),
            "fraud_rows": int(summary["fraud_rows"]),
            "fraud_rate": round(summary["fraud_rate"], 6),
            "flagged_rows": int(summary["flagged_rows"]),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
