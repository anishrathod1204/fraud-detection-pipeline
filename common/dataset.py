"""Bounded-memory access to the PaySim CSV.

The Kaggle file is ~470 MB and 6,362,620 rows. Nothing in this project loads it
whole: the producer streams it row by row, and the training job takes a
systematic sample. Both go through this module so there is one definition of
"reading the dataset", including header validation and dtype handling.

Sampling
--------
:func:`read_frame` subsamples by **systematic random sampling**: the file is
divided into consecutive blocks of ``stride`` rows and exactly one row is kept
from each block, at a seeded random offset within it.

* Taking the first *N* rows would cover only the first few simulated days -
  PaySim is strictly time-ordered by ``step`` - so the model would never see
  later-period behaviour.
* Oversampling the 0.13% fraud rows would be worse than useless: it inflates
  precision relative to production and makes the evaluation report a lie.
* A *fixed* offset (plain every-*k*-th-row) is cheaper still, but aliases: any
  periodicity in the data whose period shares a factor with ``stride`` is either
  amplified or erased. A seeded random offset per block removes the aliasing
  while keeping one row per block, so class prevalence and time coverage are both
  preserved, and the seed keeps it reproducible.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from common.logging_config import get_logger
from common.schema import PAYSIM_COLUMNS, PAYSIM_DTYPES

__all__ = [
    "DatasetError",
    "count_data_rows",
    "iter_transactions",
    "read_frame",
    "validate_header",
]

_LOGGER = get_logger(__name__)

#: Read buffer for the newline-counting row count. 1 MiB measurably beats the
#: default 8 KiB on a 470 MB file without meaningful memory cost.
_COUNT_BUFFER_BYTES: Final[int] = 1024 * 1024

#: Default seed for sample offsets. Fixed so two training runs see identical
#: data; override per call when deliberately varying the sample.
DEFAULT_SAMPLE_SEED: Final[int] = 42


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or its header is not PaySim's."""


def validate_header(csv_path: Path) -> tuple[str, ...]:
    """Check that a CSV has exactly the PaySim columns.

    Run before any long read so a wrong file fails in milliseconds rather than
    after streaming a million rows of something unexpected.

    Args:
        csv_path: Path to the CSV.

    Returns:
        The header columns, in file order.

    Raises:
        DatasetError: If the file is missing, empty, or its columns do not match
            :data:`common.schema.PAYSIM_COLUMNS` as a set.
    """
    if not csv_path.is_file():
        raise DatasetError(
            f"dataset not found at {csv_path}. "
            "Generate one with `make dataset`, or set PAYSIM_CSV_PATH to the "
            "Kaggle PaySim CSV."
        )

    try:
        header_frame = pd.read_csv(csv_path, nrows=0)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"dataset at {csv_path} is empty") from exc

    columns = tuple(str(column) for column in header_frame.columns)
    missing = set(PAYSIM_COLUMNS) - set(columns)
    unexpected = set(columns) - set(PAYSIM_COLUMNS)

    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise DatasetError(
            f"{csv_path} does not look like a PaySim export: {'; '.join(details)}"
        )

    return columns


def count_data_rows(csv_path: Path) -> int:
    """Count data rows in a CSV, excluding the header.

    Counts newlines over a buffered binary read instead of parsing the file.
    This is roughly two orders of magnitude faster than a pandas pass and is
    accurate for PaySim, whose fields never contain embedded newlines.

    Args:
        csv_path: Path to the CSV.

    Returns:
        Number of data rows. Zero for a header-only file.

    Raises:
        DatasetError: If the file does not exist.
    """
    if not csv_path.is_file():
        raise DatasetError(f"dataset not found at {csv_path}")

    newlines = 0
    trailing_byte = b""
    with csv_path.open("rb") as handle:
        while chunk := handle.read(_COUNT_BUFFER_BYTES):
            newlines += chunk.count(b"\n")
            trailing_byte = chunk[-1:]

    # A final line with no trailing newline still holds a row.
    if trailing_byte and trailing_byte != b"\n":
        newlines += 1

    return max(0, newlines - 1)


def iter_transactions(
    csv_path: Path,
    *,
    chunk_size: int,
    fraud_only: bool = False,
    max_records: int | None = None,
    loop: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield PaySim rows one at a time with bounded memory.

    Memory stays at roughly ``chunk_size`` rows regardless of file size, which
    is the whole point: the producer must be able to stream a 6M-row file on a
    laptop.

    Args:
        csv_path: Path to the CSV.
        chunk_size: Rows per pandas chunk.
        fraud_only: Yield only rows with ``isFraud == 1``. Used by the demo
            replay mode, where waiting for organic fraud at 0.13% prevalence
            would mean roughly 770 normal transactions between each one.
        max_records: Stop after this many yielded rows. ``None`` means the whole
            file.
        loop: Restart from the beginning on reaching EOF. Lets a demo or load
            test run indefinitely without a larger dataset.

    Yields:
        One dict per row, keyed by PaySim column name, values already coerced to
        the dtypes in :data:`common.schema.PAYSIM_DTYPES`.

    Raises:
        DatasetError: If the dataset is missing or has the wrong header, or if
            ``fraud_only`` is requested and the file holds no fraud rows (which
            would otherwise spin forever under ``loop``).
    """
    validate_header(csv_path)

    emitted = 0
    passes = 0

    while True:
        passes += 1
        emitted_this_pass = 0

        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            dtype=dict(PAYSIM_DTYPES),
            # Guards against a subtly reordered export: select by name, in the
            # canonical order, so downstream positional assumptions hold.
            usecols=list(PAYSIM_COLUMNS),
        )

        for chunk in reader:
            frame = chunk[list(PAYSIM_COLUMNS)]
            if fraud_only:
                frame = frame[frame["isFraud"] == 1]
                if frame.empty:
                    continue

            # to_dict("records") is materially faster than itertuples() plus
            # per-row dict construction, and chunk_size bounds the allocation.
            for record in frame.to_dict("records"):
                # 'type' arrives as a pandas StringDtype scalar; the JSON encoder
                # needs a plain str.
                record["type"] = str(record["type"])
                yield record

                emitted += 1
                emitted_this_pass += 1
                if max_records is not None and emitted >= max_records:
                    _LOGGER.info(
                        "reached max_records limit",
                        extra={"max_records": max_records, "passes": passes},
                    )
                    return

        if emitted_this_pass == 0:
            # Nothing matched on a full pass. Looping would spin on an empty
            # result set, so stop and say why.
            if fraud_only:
                raise DatasetError(
                    f"{csv_path} contains no rows with isFraud == 1; "
                    "--replay-fraud-only has nothing to stream"
                )
            raise DatasetError(f"{csv_path} contains no data rows")

        if not loop:
            return

        _LOGGER.info(
            "reached end of dataset, restarting",
            extra={"pass": passes, "rows_emitted": emitted},
        )


def read_frame(
    csv_path: Path,
    *,
    max_rows: int | None = None,
    chunk_size: int = 200_000,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> pd.DataFrame:
    """Load the dataset into a DataFrame, optionally systematically sampled.

    Reads in chunks and keeps one row per block of ``stride`` rows, so peak
    memory is one chunk plus the retained sample rather than the whole file. See
    the module docstring for why this sampling scheme was chosen.

    Args:
        csv_path: Path to the CSV.
        max_rows: Approximate upper bound on returned rows. ``None`` loads
            everything. The result may land slightly under this bound because
            the stride is an integer.
        chunk_size: Rows per pandas chunk while reading.
        seed: Seed for the per-block sample offsets.

    Returns:
        A DataFrame with the PaySim columns in canonical order and a reset index.

    Raises:
        DatasetError: If the dataset is missing, has the wrong header, or is
            empty.
    """
    validate_header(csv_path)

    total_rows = count_data_rows(csv_path)
    if total_rows == 0:
        raise DatasetError(f"{csv_path} contains no data rows")

    if max_rows is None or max_rows >= total_rows:
        stride = 1
        block_offsets = None
    else:
        # Integer stride that yields at most max_rows: ceil(total / max_rows).
        stride = -(-total_rows // max_rows)
        block_count = -(-total_rows // stride)
        # One offset per block, drawn once up front. block_count <= max_rows, so
        # this array is never larger than the sample itself.
        block_offsets = np.random.default_rng(seed).integers(
            low=0, high=stride, size=block_count, dtype=np.int64
        )

    _LOGGER.info(
        "loading dataset",
        extra={
            "path": str(csv_path),
            "total_rows": total_rows,
            "max_rows": max_rows,
            "stride": stride,
            "sampling": "all" if stride == 1 else "systematic-random",
        },
    )

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype=dict(PAYSIM_DTYPES),
        usecols=list(PAYSIM_COLUMNS),
    )

    retained: list[pd.DataFrame] = []
    position = 0
    for chunk in reader:
        if block_offsets is None:
            retained.append(chunk)
        else:
            # Global row positions, so block identity is continuous across chunk
            # boundaries; restarting per chunk would bias toward chunk starts.
            positions = np.arange(position, position + len(chunk), dtype=np.int64)
            blocks = positions // stride
            keep = (positions - blocks * stride) == block_offsets[blocks]
            if keep.any():
                retained.append(chunk.loc[keep])
        position += len(chunk)

    frame = pd.concat(retained, ignore_index=True, copy=False)
    frame = frame[list(PAYSIM_COLUMNS)]

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[:max_rows]

    _LOGGER.info(
        "dataset loaded",
        extra={
            "rows": len(frame),
            "fraud_rows": int(frame["isFraud"].sum()),
            "fraud_rate": round(float(frame["isFraud"].mean()), 6),
            "memory_mb": round(frame.memory_usage(deep=True).sum() / 1e6, 1),
        },
    )
    return frame.reset_index(drop=True)
