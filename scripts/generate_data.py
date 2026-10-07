"""Generate a PaySim-shaped synthetic dataset (same columns as the Kaggle file).

If you have the real PaySim CSV, just copy it to data/paysim.csv and skip this.
Usage: python scripts/generate_data.py --rows 200000 --fraud-rate 0.01
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import RAW_COLUMNS  # noqa: E402


def generate(rows: int, fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(round(rows * fraud_rate))
    n_legit = rows - n_fraud

    # ---------------- legitimate ----------------
    types = rng.choice(
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_legit, p=[0.35, 0.34, 0.22, 0.084, 0.006],
    )
    mu = {"PAYMENT": (7.0, 1.0), "CASH_IN": (11.5, 1.0), "CASH_OUT": (11.5, 1.0),
          "TRANSFER": (11.8, 1.1), "DEBIT": (6.0, 1.2)}
    amount = np.array([rng.lognormal(*mu[t]) for t in types]).round(2)

    old_o = np.zeros(n_legit)
    new_o = np.zeros(n_legit)
    spends = np.isin(types, ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT"])
    cash_in = types == "CASH_IN"
    old_o[spends] = amount[spends] * (1.0 + rng.lognormal(1.0, 1.0, spends.sum()))
    new_o[spends] = old_o[spends] - amount[spends]
    old_o[cash_in] = rng.lognormal(11.0, 2.0, cash_in.sum())
    new_o[cash_in] = old_o[cash_in] + amount[cash_in]

    # a small share of honest accounts legitimately empty themselves
    tc = np.isin(types, ["TRANSFER", "CASH_OUT"])
    drain = tc & (rng.random(n_legit) < 0.01)
    old_o[drain] = amount[drain]
    new_o[drain] = 0.0

    old_d = np.zeros(n_legit)
    new_d = np.zeros(n_legit)
    has_dest = np.isin(types, ["TRANSFER", "CASH_OUT"])
    old_d[has_dest] = rng.lognormal(11.0, 2.0, has_dest.sum()) * (rng.random(has_dest.sum()) > 0.25)
    new_d[has_dest] = old_d[has_dest] + amount[has_dest]
    # some destinations don't report balances (zeros), as in real PaySim
    unreported = has_dest & (rng.random(n_legit) < 0.03)
    old_d[unreported] = 0.0
    new_d[unreported] = 0.0

    legit = pd.DataFrame({"type": types, "amount": amount, "oldbalanceOrg": old_o.round(2),
                          "newbalanceOrig": new_o.round(2), "oldbalanceDest": old_d.round(2),
                          "newbalanceDest": new_d.round(2), "isFraud": 0})

    # ---------------- fraud ----------------
    f_types = rng.choice(["TRANSFER", "CASH_OUT"], size=n_fraud)
    f_old = rng.lognormal(12.3, 1.2, n_fraud).round(2)           # victim balance
    f_old_d = rng.lognormal(11.0, 2.0, n_fraud) * (rng.random(n_fraud) > 0.6)
    f_new_d = np.where(rng.random(n_fraud) < 0.5, 0.0, f_old_d + f_old)
    f_old_d = np.where(f_new_d == 0.0, 0.0, f_old_d)
    fraud = pd.DataFrame({"type": f_types, "amount": f_old, "oldbalanceOrg": f_old,
                          "newbalanceOrig": 0.0, "oldbalanceDest": f_old_d.round(2),
                          "newbalanceDest": np.round(f_new_d, 2), "isFraud": 1})

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["step"] = rng.integers(1, 744, size=len(df))
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    # np.char.add works on every numpy version ("C" + str_array fails on numpy 1.x)
    df["nameOrig"] = np.char.add("C", rng.integers(10**8, 10**9, size=len(df)).astype(str))
    is_merchant = df["type"].isin(["PAYMENT", "DEBIT", "CASH_IN"]).to_numpy()
    df["nameDest"] = np.char.add(np.where(is_merchant, "M", "C"),
                                 rng.integers(10**8, 10**9, size=len(df)).astype(str))
    df["isFlaggedFraud"] = 0
    return df[RAW_COLUMNS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--fraud-rate", type=float, default=0.01, help="real PaySim is ~0.0013")
    ap.add_argument("--out", default="data/paysim.csv")
    ap.add_argument("--force", action="store_true", help="overwrite existing file")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} already exists - keeping it (use --force to regenerate).")
        sys.exit(0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    data = generate(a.rows, a.fraud_rate)
    data.to_csv(a.out, index=False)
    print(f"Wrote {len(data):,} rows ({int(data.isFraud.sum()):,} fraud) to {a.out}")
