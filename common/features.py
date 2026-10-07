"""Feature engineering shared by training and the live scorer.

Using ONE implementation for both guarantees there is no train/serve skew.
"""
import numpy as np
import pandas as pd

from common.schema import REQUIRED_FOR_SCORING, TX_TYPES

FEATURE_COLUMNS = [
    "log_amount",
    "is_cash_in", "is_cash_out", "is_debit", "is_payment", "is_transfer",
    "error_orig", "error_dest",
    "orig_drained", "amount_to_balance", "dest_unreported",
    "hour",
]


def _signed_log(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction rows into the numeric matrix the model expects."""
    missing = [c for c in REQUIRED_FOR_SCORING if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    amount = df["amount"].astype(float)
    old_o = df["oldbalanceOrg"].astype(float)
    new_o = df["newbalanceOrig"].astype(float)
    old_d = df["oldbalanceDest"].astype(float)
    new_d = df["newbalanceDest"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(amount.clip(lower=0))
    for t in TX_TYPES:
        out[f"is_{t.lower()}"] = (df["type"] == t).astype(float)

    # Accounting identities: a legitimate transaction balances, fraud often does not.
    out["error_orig"] = _signed_log(new_o + amount - old_o)
    out["error_dest"] = _signed_log(old_d + amount - new_d)
    out["orig_drained"] = ((new_o == 0) & (old_o > 0)).astype(float)
    out["amount_to_balance"] = (amount / (old_o + 1.0)).clip(upper=10.0)
    out["dest_unreported"] = ((old_d == 0) & (new_d == 0) & (amount > 0)).astype(float)
    out["hour"] = (df["step"].astype(float) % 24)
    return out[FEATURE_COLUMNS]
