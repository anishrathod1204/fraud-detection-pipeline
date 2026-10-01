"""PaySim transaction schema, wire format and validation.

One module defines the contract between every stage of the pipeline: the CSV on
disk, the JSON on the Kafka topic, the pandas frame the model is trained on, and
the Spark ``StructType`` used to parse the stream. Defining it once is what
makes "the streaming job applies the same features as training" enforceable
rather than aspirational.

Naming
------
PaySim ships camelCase column names, two of which are inconsistent
(``oldbalanceOrg`` but ``newbalanceOrig``). Those names are preserved exactly on
the CSV and on the Kafka wire so the real Kaggle file works unmodified, and are
mapped to snake_case for internal use and for Cassandra columns. The mapping
lives in :data:`COLUMN_TO_FIELD`.

Wire envelope
-------------
Messages on the ``transactions`` topic carry the eleven PaySim columns plus
:data:`PRODUCED_AT_FIELD`, an epoch-millisecond publish timestamp added by the
producer. The streaming job subtracts it from its own clock to measure true
end-to-end latency, which the Spark micro-batch metrics cannot show on their own.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Final, Mapping

__all__ = [
    "ALERT_BUCKET_MINUTES",
    "AMOUNT_COLUMNS",
    "COLUMN_TO_FIELD",
    "FRAUD_CAPABLE_TYPES",
    "LABEL_COLUMNS",
    "MalformedTransactionError",
    "PAYSIM_COLUMNS",
    "PAYSIM_DTYPES",
    "PAYSIM_FIELD_TYPES",
    "PRODUCED_AT_FIELD",
    "TRANSACTION_TYPES",
    "alert_bucket",
    "decode_transaction",
    "encode_transaction",
    "normalise_record",
]


class MalformedTransactionError(ValueError):
    """Raised when a record cannot be coerced into a valid transaction.

    The streaming job catches this and routes the offending payload to the
    dead-letter topic instead of failing the micro-batch - a single bad message
    must never stall the stream.
    """


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
#: The eleven PaySim columns, in the order they appear in the Kaggle CSV.
PAYSIM_COLUMNS: Final[tuple[str, ...]] = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

#: Wire/CSV column -> logical type. ``"long"``, ``"double"`` and ``"string"`` are
#: deliberately Spark's type vocabulary so :mod:`streaming` can build a
#: ``StructType`` straight from this mapping without a second lookup table.
PAYSIM_FIELD_TYPES: Final[Mapping[str, str]] = {
    "step": "long",
    "type": "string",
    "amount": "double",
    "nameOrig": "string",
    "oldbalanceOrg": "double",
    "newbalanceOrig": "double",
    "nameDest": "string",
    "oldbalanceDest": "double",
    "newbalanceDest": "double",
    "isFraud": "long",
    "isFlaggedFraud": "long",
}

#: pandas dtypes for ``read_csv``. Specified explicitly for two reasons: dtype
#: inference on a chunked read can assign different types to different chunks of
#: the same column, and float32 halves the memory footprint of a 6M-row frame at
#: no cost to a tree ensemble's accuracy.
PAYSIM_DTYPES: Final[Mapping[str, str]] = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

#: Canonical camelCase -> snake_case mapping. Also the Cassandra column names.
COLUMN_TO_FIELD: Final[Mapping[str, str]] = {
    "step": "step",
    "type": "tx_type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud_label",
    "isFlaggedFraud": "is_flagged_fraud",
}

#: Monetary columns, grouped for bulk numeric handling.
AMOUNT_COLUMNS: Final[tuple[str, ...]] = (
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

#: Ground-truth columns. Never used as model inputs - the model is unsupervised
#: and these exist purely for offline evaluation and dashboard comparison.
LABEL_COLUMNS: Final[tuple[str, ...]] = ("isFraud", "isFlaggedFraud")

#: Every transaction type present in PaySim.
TRANSACTION_TYPES: Final[tuple[str, ...]] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

#: The only two types that ever carry fraud in PaySim: the simulated attack
#: drains an account by TRANSFER and then liquidates via CASH_OUT. Useful for
#: sanity-checking a generated dataset, never as a model feature - hardcoding it
#: as a rule would be label leakage.
FRAUD_CAPABLE_TYPES: Final[frozenset[str]] = frozenset({"TRANSFER", "CASH_OUT"})

#: Field name carrying the producer's publish timestamp, epoch milliseconds.
PRODUCED_AT_FIELD: Final[str] = "producedAtMs"

#: Width of a ``fraud_alerts`` partition bucket, in minutes. Changing this
#: changes the Cassandra partitioning scheme and invalidates existing buckets.
ALERT_BUCKET_MINUTES: Final[int] = 5


# ---------------------------------------------------------------------------
# Validation and coercion
# ---------------------------------------------------------------------------
def _coerce(column: str, value: Any) -> Any:
    """Coerce a single raw field to its declared type.

    Args:
        column: PaySim column name.
        value: Raw value from JSON or CSV.

    Returns:
        The value as :class:`int`, :class:`float` or :class:`str`.

    Raises:
        MalformedTransactionError: If the value is ``None`` or not convertible.
    """
    kind = PAYSIM_FIELD_TYPES[column]

    if value is None:
        raise MalformedTransactionError(f"field {column!r} is null")

    try:
        if kind == "long":
            # float() first so "1.0" and 1.0 both work; JSON producers in other
            # languages routinely emit integral values as floats.
            return int(float(value))
        if kind == "double":
            return float(value)
        text = str(value).strip()
        if not text:
            raise MalformedTransactionError(f"field {column!r} is empty")
        return text
    except MalformedTransactionError:
        raise
    except (TypeError, ValueError) as exc:
        raise MalformedTransactionError(
            f"field {column!r} value {value!r} is not a valid {kind}"
        ) from exc


def normalise_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a raw record and return it with coerced types.

    Checks that all eleven PaySim columns are present, coerces each to its
    declared type, and applies the domain constraints that a model cannot
    sensibly be asked to score around: non-negative money and a known
    transaction type.

    Args:
        raw: Mapping of PaySim column names to raw values.

    Returns:
        A new dict with the eleven columns coerced, plus
        :data:`PRODUCED_AT_FIELD` when it was present in the input.

    Raises:
        MalformedTransactionError: If a column is missing, untypeable, or
            violates a domain constraint.
    """
    missing = [column for column in PAYSIM_COLUMNS if column not in raw]
    if missing:
        raise MalformedTransactionError(
            f"missing required field(s): {', '.join(missing)}"
        )

    record: dict[str, Any] = {
        column: _coerce(column, raw[column]) for column in PAYSIM_COLUMNS
    }

    if record["type"] not in TRANSACTION_TYPES:
        raise MalformedTransactionError(
            f"unknown transaction type {record['type']!r}; "
            f"expected one of {', '.join(TRANSACTION_TYPES)}"
        )
    if record["step"] < 0:
        raise MalformedTransactionError(f"step must be >= 0, got {record['step']}")

    for column in AMOUNT_COLUMNS:
        if record[column] < 0.0:
            raise MalformedTransactionError(
                f"{column} must be >= 0, got {record[column]}"
            )

    # Preserve the producer timestamp when present so latency can be measured
    # downstream; absence is legitimate for records replayed straight from CSV.
    if PRODUCED_AT_FIELD in raw and raw[PRODUCED_AT_FIELD] is not None:
        try:
            record[PRODUCED_AT_FIELD] = int(raw[PRODUCED_AT_FIELD])
        except (TypeError, ValueError) as exc:
            raise MalformedTransactionError(
                f"{PRODUCED_AT_FIELD} value {raw[PRODUCED_AT_FIELD]!r} is not an integer"
            ) from exc

    return record


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------
def encode_transaction(
    record: Mapping[str, Any], *, produced_at_ms: int | None = None
) -> bytes:
    """Serialise a transaction for publication to Kafka.

    Args:
        record: Mapping of PaySim column names to values.
        produced_at_ms: Publish time in epoch milliseconds. Defaults to now.
            Injected rather than read from the clock internally so tests are
            deterministic.

    Returns:
        Compact UTF-8 JSON bytes.
    """
    payload = {column: record[column] for column in PAYSIM_COLUMNS}
    payload[PRODUCED_AT_FIELD] = (
        produced_at_ms
        if produced_at_ms is not None
        else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    )
    # separators= strips the whitespace json.dumps adds by default: roughly a 10%
    # payload reduction, which at 5k messages/sec is worth having.
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_transaction(payload: bytes | str) -> dict[str, Any]:
    """Parse and validate a transaction from a Kafka message payload.

    Args:
        payload: Raw message value, JSON bytes or string.

    Returns:
        A validated, type-coerced record.

    Raises:
        MalformedTransactionError: If the payload is not a JSON object, is not
            decodable UTF-8, or fails validation.
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        parsed = json.loads(text)
    except UnicodeDecodeError as exc:
        raise MalformedTransactionError("payload is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise MalformedTransactionError(f"payload is not valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise MalformedTransactionError(
            f"payload must be a JSON object, got {type(parsed).__name__}"
        )

    return normalise_record(parsed)


# ---------------------------------------------------------------------------
# Cassandra partition helpers
# ---------------------------------------------------------------------------
def alert_bucket(moment: datetime, *, window_minutes: int = ALERT_BUCKET_MINUTES) -> str:
    """Compute the ``fraud_alerts`` partition key for a timestamp.

    Alerts are bucketed into fixed UTC windows so that "show recent alerts" is a
    bounded single-partition read. The returned key is the window's start time
    formatted ``yyyyMMddHHmm``; ``12:07:31`` with a five-minute window yields
    ``...1205``.

    Naive datetimes are treated as UTC rather than local time. Interpreting them
    as local would make bucket keys depend on the machine's timezone and split
    one logical window across two partitions in a mixed deployment.

    Args:
        moment: Detection timestamp, timezone-aware or naive-UTC.
        window_minutes: Bucket width. Must divide 60 evenly so buckets align to
            the hour.

    Returns:
        The bucket key, for example ``"202610011205"``.

    Raises:
        ValueError: If ``window_minutes`` does not divide 60 evenly.
    """
    if window_minutes <= 0 or 60 % window_minutes != 0:
        raise ValueError(
            f"window_minutes={window_minutes} must be a positive divisor of 60"
        )

    moment_utc = (
        moment.astimezone(timezone.utc)
        if moment.tzinfo is not None
        else moment.replace(tzinfo=timezone.utc)
    )
    floored_minute = (moment_utc.minute // window_minutes) * window_minutes
    return f"{moment_utc:%Y%m%d%H}{floored_minute:02d}"
