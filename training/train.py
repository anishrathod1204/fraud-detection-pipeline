"""Train an Isolation Forest anomaly detector and write an evaluation report.

The model itself is UNSUPERVISED (it never sees labels). Labels are used only afterwards,
to pick an alert threshold on a holdout set and to report precision / recall.
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config  # noqa: E402
from common.features import FEATURE_COLUMNS, build_features  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402

log = get_logger("train")

MAX_TRAIN_ROWS = 300_000
MAX_HOLDOUT_ROWS = 1_000_000


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1 on the holdout set."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": thr, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "alerts": int(pred.sum()),
        "holdout_rows": int(len(y_true)), "holdout_fraud": int(y_true.sum()),
    }


def main(data_file: str = None, model_path: str = None, report_path: str = "docs/model_evaluation.md") -> dict:
    data_file = data_file or config.DATA_FILE
    model_path = model_path or config.MODEL_PATH
    if not os.path.exists(data_file):
        log.error("Data file %s not found. Run the data step first (python scripts/generate_data.py).", data_file)
        sys.exit(1)

    log.info("Loading %s", data_file)
    df = pd.read_csv(data_file)
    log.info("Loaded %s rows, fraud rate %.4f%%", f"{len(df):,}", 100 * df["isFraud"].mean())

    train_df, hold_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["isFraud"])
    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(MAX_TRAIN_ROWS, random_state=42)
    if len(hold_df) > MAX_HOLDOUT_ROWS:
        hold_df = hold_df.sample(MAX_HOLDOUT_ROWS, random_state=42)

    X_train = build_features(train_df)
    X_hold = build_features(hold_df)
    y_hold = hold_df["isFraud"].to_numpy()

    log.info("Fitting IsolationForest on %s rows (labels NOT used)", f"{len(X_train):,}")
    t0 = time.time()
    model = IsolationForest(n_estimators=200, max_samples=1024, contamination="auto",
                            random_state=42, n_jobs=-1)
    model.fit(X_train.to_numpy())
    log.info("Fit done in %.1fs", time.time() - t0)

    scores = -model.score_samples(X_hold.to_numpy())    # higher = more anomalous
    thr = pick_threshold(y_hold, scores)
    metrics = evaluate(y_hold, scores, thr)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "feature_columns": FEATURE_COLUMNS, "metrics": metrics}, model_path)
    log.info("Saved model to %s", model_path)

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# Model evaluation\n\n"
            f"Isolation Forest (unsupervised), {len(X_train):,} training rows, "
            f"{metrics['holdout_rows']:,} holdout rows ({metrics['holdout_fraud']:,} fraud).\n\n"
            "| Metric | Value |\n|---|---|\n"
            f"| ROC AUC | {metrics['roc_auc']:.4f} |\n"
            f"| PR AUC (average precision) | {metrics['pr_auc']:.4f} |\n"
            f"| Alert threshold (anomaly score) | {metrics['threshold']:.4f} |\n"
            f"| Precision | {metrics['precision']:.4f} |\n"
            f"| Recall | {metrics['recall']:.4f} |\n"
            f"| F1 | {metrics['f1']:.4f} |\n"
            f"| True positives / False positives / False negatives | {metrics['tp']} / {metrics['fp']} / {metrics['fn']} |\n\n"
            "Labels are used only to choose the threshold and report these numbers; the model never trains on them.\n"
        )
    log.info("ROC-AUC %.4f | PR-AUC %.4f | precision %.3f | recall %.3f | threshold %.4f",
             metrics["roc_auc"], metrics["pr_auc"], metrics["precision"], metrics["recall"], thr)
    return metrics


if __name__ == "__main__":
    main()
