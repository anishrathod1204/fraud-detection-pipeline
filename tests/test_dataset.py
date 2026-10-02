"""Unit tests for the bounded-memory PaySim CSV reader.

This module is the boundary between the 470 MB file on disk and everything the
rest of the pipeline does, so its contract matters in a way a pure helper's does
not: if header validation passes a wrong file, or sampling quietly drops the
fraud class, every later stage is wrong without raising. The tests therefore
pin the boundary conditions - a missing file, a misordered header, an empty
file, a chunk boundary - and the two sampling properties the module claims:
class prevalence is preserved and the sample is reproducible from its seed.

Fixtures write tiny synthetic CSVs to ``tmp_path``, so the tests run in
milliseconds and never touch the real dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from common import dataset as dataset_module
from common.dataset import (
    DEFAULT_SAMPLE_SEED,
    DatasetError,
    count_data_rows,
    iter_transactions,
    read_frame,
    validate_header,
)
from common.schema import PAYSIM_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def write_paysim_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    header: tuple[str, ...] = PAYSIM_COLUMNS,
) -> Path:
    """Write a PaySim-shaped CSV with a controllable header.

    Args:
        path: Destination path.
        rows: Row dicts, keyed by canonical column name.
        header: Column order to write; defaults to the canonical order.

    Returns:
        The path written.
    """
    frame = pd.DataFrame(rows, columns=list(header))
    frame.to_csv(path, index=False)
    return path


def make_row(**overrides: Any) -> dict[str, Any]:
    """Build one valid PaySim row as a plain dict.

    Args:
        **overrides: Values to replace in the default template.

    Returns:
        A row dict with all eleven columns.
    """
    row: dict[str, Any] = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 900.0,
        "nameDest": "M1",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def small_csv(tmp_path: Path) -> Path:
    """A ten-row CSV: steps 1-9 normal plus one fraud row at step 10 (10%)."""
    rows = [make_row(step=i, nameOrig=f"C{i}") for i in range(1, 10)]
    rows.append(
        make_row(step=10, type="TRANSFER", nameOrig="C9", isFraud=1, amount=5000.0)
    )
    return write_paysim_csv(tmp_path / "small.csv", rows)


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------
class TestValidateHeader:
    """A wrong file must fail fast, before any long read."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="dataset not found"):
            validate_header(tmp_path / "absent.csv")

    def test_accepts_canonical_header(self, small_csv: Path) -> None:
        assert validate_header(small_csv) == PAYSIM_COLUMNS

    def test_accepts_reordered_header(self, tmp_path: Path) -> None:
        # Column *order* is not part of the contract; the set is.
        shuffled = tuple(reversed(PAYSIM_COLUMNS))
        path = write_paysim_csv(
            tmp_path / "reordered.csv", [make_row()], header=shuffled
        )
        assert set(validate_header(path)) == set(PAYSIM_COLUMNS)

    def test_rejects_missing_column(self, tmp_path: Path) -> None:
        header = PAYSIM_COLUMNS[:-1]  # drop isFlaggedFraud
        path = write_paysim_csv(tmp_path / "short.csv", [], header=header)
        with pytest.raises(DatasetError, match="missing"):
            validate_header(path)

    def test_rejects_unexpected_column(self, tmp_path: Path) -> None:
        header = PAYSIM_COLUMNS + ("extra",)
        path = write_paysim_csv(tmp_path / "wide.csv", [], header=header)
        with pytest.raises(DatasetError, match="unexpected"):
            validate_header(path)

    def test_rejects_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(DatasetError, match="empty"):
            validate_header(path)


# ---------------------------------------------------------------------------
# Row counting
# ---------------------------------------------------------------------------
class TestCountDataRows:
    """Counting must exclude the header and handle a missing trailing newline."""

    def test_counts_excluding_header(self, small_csv: Path) -> None:
        assert count_data_rows(small_csv) == 10

    def test_header_only_is_zero(self, tmp_path: Path) -> None:
        path = write_paysim_csv(tmp_path / "headeronly.csv", [])
        assert count_data_rows(path) == 0

    def test_no_trailing_newline_counts_last_row(self, tmp_path: Path) -> None:
        path = write_paysim_csv(tmp_path / "notrail.csv", [make_row(), make_row()])
        # Strip the trailing newline pandas writes.
        text = path.read_text().rstrip("\n")
        path.write_text(text)
        assert count_data_rows(path) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="not found"):
            count_data_rows(tmp_path / "absent.csv")


# ---------------------------------------------------------------------------
# Row iteration
# ---------------------------------------------------------------------------
class TestIterTransactions:
    """Streaming must be bounded, ordered and filterable."""

    def test_yields_all_rows_in_order(self, small_csv: Path) -> None:
        records = list(iter_transactions(small_csv, chunk_size=3))
        assert len(records) == 10
        assert [r["step"] for r in records] == list(range(1, 11))

    def test_streams_across_chunk_boundary(self, small_csv: Path) -> None:
        # chunk_size of 3 means three full chunks and a remainder; the count must
        # not depend on where the boundary falls.
        records = list(iter_transactions(small_csv, chunk_size=3))
        assert len(records) == 10

    def test_type_is_plain_str(self, small_csv: Path) -> None:
        record = next(iter_transactions(small_csv, chunk_size=5))
        assert type(record["type"]) is str

    def test_fraud_only_filters(self, small_csv: Path) -> None:
        records = list(iter_transactions(small_csv, chunk_size=4, fraud_only=True))
        assert len(records) == 1
        assert records[0]["isFraud"] == 1

    def test_max_records_stops_early(self, small_csv: Path) -> None:
        records = list(iter_transactions(small_csv, chunk_size=3, max_records=4))
        assert len(records) == 4

    def test_loop_restarts(self, small_csv: Path) -> None:
        records = list(
            iter_transactions(small_csv, chunk_size=3, max_records=13, loop=True)
        )
        assert len(records) == 13
        # Rows 1-10 then 1-3 again.
        assert [r["step"] for r in records[:10]] == list(range(1, 11))
        assert [r["step"] for r in records[10:]] == [1, 2, 3]

    def test_missing_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="dataset not found"):
            list(iter_transactions(tmp_path / "absent.csv", chunk_size=3))

    def test_fraud_only_with_no_fraud_raises(self, tmp_path: Path) -> None:
        # Without this guard, --loop would spin forever on an empty result set.
        rows = [make_row(step=i) for i in range(3)]
        path = write_paysim_csv(tmp_path / "nofraud.csv", rows)
        with pytest.raises(DatasetError, match="no rows with isFraud"):
            list(iter_transactions(path, chunk_size=2, fraud_only=True))


# ---------------------------------------------------------------------------
# Frame loading and sampling
# ---------------------------------------------------------------------------
class TestReadFrame:
    """Systematic sampling must preserve prevalence and be reproducible."""

    def test_loads_all_when_under_bound(self, small_csv: Path) -> None:
        frame = read_frame(small_csv, max_rows=100, chunk_size=4)
        assert len(frame) == 10
        assert list(frame.columns) == list(PAYSIM_COLUMNS)

    def test_index_is_reset(self, small_csv: Path) -> None:
        frame = read_frame(small_csv, chunk_size=4)
        assert list(frame.index) == list(range(len(frame)))

    def test_missing_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="dataset not found"):
            read_frame(tmp_path / "absent.csv")

    def test_sampling_respects_max_rows(self, tmp_path: Path) -> None:
        rows = [make_row(step=i, nameOrig=f"C{i}") for i in range(1000)]
        path = write_paysim_csv(tmp_path / "big.csv", rows)
        frame = read_frame(path, max_rows=100, chunk_size=200)
        # ceil(1000/100) = 10 stride -> exactly 100 sampled rows.
        assert len(frame) == 100

    def test_sampling_is_deterministic(self, tmp_path: Path) -> None:
        rows = [make_row(step=i, nameOrig=f"C{i}") for i in range(500)]
        path = write_paysim_csv(tmp_path / "big.csv", rows)
        first = read_frame(path, max_rows=50, chunk_size=100)[["step"]]
        second = read_frame(path, max_rows=50, chunk_size=100)[["step"]]
        assert first.equals(second)

    def test_different_seed_changes_the_sample(self, tmp_path: Path) -> None:
        rows = [make_row(step=i, nameOrig=f"C{i}") for i in range(500)]
        path = write_paysim_csv(tmp_path / "big.csv", rows)
        a = read_frame(path, max_rows=50, chunk_size=100, seed=1)["step"].tolist()
        b = read_frame(path, max_rows=50, chunk_size=100, seed=999)["step"].tolist()
        assert a != b

    def test_prevalence_is_preserved(self, tmp_path: Path) -> None:
        # Interleave fraud every tenth row; a stride sampling that dropped the
        # fraud class would be the failure this test exists to catch.
        rows = [
            make_row(step=i, nameOrig=f"C{i}", isFraud=1 if i % 10 == 0 else 0)
            for i in range(1000)
        ]
        path = write_paysim_csv(tmp_path / "prevalent.csv", rows)
        frame = read_frame(path, max_rows=200, chunk_size=200)
        observed = float(frame["isFraud"].mean())
        # 10% in the source; the sample must stay near it, not collapse or spike.
        assert 0.04 <= observed <= 0.16

    def test_empty_data_raises(self, tmp_path: Path) -> None:
        path = write_paysim_csv(tmp_path / "headeronly.csv", [])
        with pytest.raises(DatasetError, match="no data rows"):
            read_frame(path)

    def test_default_seed_constant(self) -> None:
        assert DEFAULT_SAMPLE_SEED == 42
