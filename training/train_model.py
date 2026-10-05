"""Training entrypoint for the fraud anomaly detection pipeline.

Usage
-----
python training/train_model.py [options]

All flags have defaults from :class:`~common.config.ModelConfig` which is
populated from environment variables, so CI and Docker runs need no extra args.

Example (fast dev run, no autoencoder)::

    python training/train_model.py \\
        --sample-rows 50000 \\
        --artifact-dir training/artifacts \\
        --no-autoencoder

The script exits with code 0 on success and 1 on any handled error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure the repository root is on sys.path so ``import common``
# works regardless of the working directory the caller uses.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from common.config import AppConfig, load_dotenv
from common.dataset import DatasetError, read_frame
from common.features import FEATURE_NAMES, FeatureScaler, build_batch_features
from common.logging_config import get_logger
from training.anomaly import Autoencoder, IsolationForest
from training.bundle import ModelBundle, save_bundle
from training.evaluation import (
    ThresholdChoice,
    average_precision,
    confusion_at_threshold,
    roc_auc,
    select_threshold,
)

_LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, applying config defaults.

    Args:
        argv: Argument list. ``None`` reads from :data:`sys.argv`.

    Returns:
        Parsed namespace.
    """
    # We need the config to populate defaults, but the config itself may depend
    # on .env, so load that first with no override (env wins).
    load_dotenv()
    cfg = AppConfig.from_env()

    parser = argparse.ArgumentParser(
        description="Train the fraud anomaly detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=cfg.model.training_sample_rows,
        help="Rows to sample from the PaySim CSV (prevalence-preserving).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=cfg.model.artifact_dir,
        help="Directory to write the model bundle.",
    )
    parser.add_argument(
        "--no-autoencoder",
        action="store_true",
        default=False,
        help="Skip autoencoder training (faster dev runs).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("training/artifacts/eval_report.md"),
        help="Where to write the markdown evaluation report.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset loading and feature engineering
# ---------------------------------------------------------------------------
def _load_and_featurize(
    cfg: AppConfig, sample_rows: int
) -> tuple[np.ndarray, np.ndarray, FeatureScaler]:
    """Load the CSV, build features, fit the scaler and return scaled matrix.

    Args:
        cfg: Full application config.
        sample_rows: Target rows to sample.

    Returns:
        ``(scaled_matrix, labels, fitted_scaler)`` where ``scaled_matrix`` is
        ``(n, n_features)`` float64, ``labels`` is ``(n,)`` int8.

    Raises:
        DatasetError: If the CSV is missing or malformed.
        SystemExit: (propagated) if the user asks for help.
    """
    csv_path = cfg.producer.csv_path
    _LOGGER.info("loading dataset", extra={"path": str(csv_path), "max_rows": sample_rows})

    t0 = time.perf_counter()
    frame = read_frame(
        csv_path,
        max_rows=sample_rows,
        seed=cfg.model.random_seed,
    )
    elapsed = time.perf_counter() - t0
    _LOGGER.info(
        "dataset loaded",
        extra={"rows": len(frame), "elapsed_s": round(elapsed, 2)},
    )

    labels = frame["isFraud"].to_numpy(dtype=np.int8)

    t1 = time.perf_counter()
    feature_frame = build_batch_features(
        frame,
        velocity_window=cfg.features.velocity_window_steps,
    )
    _LOGGER.info(
        "features built",
        extra={"shape": list(feature_frame.shape), "elapsed_s": round(time.perf_counter() - t1, 2)},
    )

    matrix = feature_frame.values  # (n, n_features) float64 already from build_batch_features

    t2 = time.perf_counter()
    scaler = FeatureScaler.fit(matrix, FEATURE_NAMES)
    scaled = scaler.transform(matrix)
    _LOGGER.info(
        "scaler fitted and applied",
        extra={"elapsed_s": round(time.perf_counter() - t2, 2)},
    )

    return scaled, labels, scaler

# ---------------------------------------------------------------------------
# Isolation Forest training and threshold selection
# ---------------------------------------------------------------------------
def _train_forest(
    scaled: np.ndarray, labels: np.ndarray, cfg: AppConfig
) -> tuple[IsolationForest, np.ndarray, ThresholdChoice]:
    """Fit the isolation forest and select the operating threshold.

    Args:
        scaled: Scaled feature matrix ``(n, n_features)``.
        labels: Ground-truth fraud labels ``(n,)``.
        cfg: Full application configuration.

    Returns:
        ``(forest, if_scores, threshold_choice)``
    """
    _LOGGER.info(
        "training IsolationForest",
        extra={
            "n_estimators": cfg.model.isolation_forest_n_estimators,
            "max_samples": cfg.model.isolation_forest_max_samples,
        },
    )
    t0 = time.perf_counter()
    forest = IsolationForest(
        n_estimators=cfg.model.isolation_forest_n_estimators,
        max_samples=cfg.model.isolation_forest_max_samples,
    ).fit(scaled, seed=cfg.model.random_seed)
    _LOGGER.info(
        "IsolationForest fitted",
        extra={"elapsed_s": round(time.perf_counter() - t0, 2)},
    )

    if_scores = forest.score_samples(scaled)

    _LOGGER.info(
        "selecting threshold",
        extra={"min_recall": cfg.model.target_min_recall},
    )
    choice = select_threshold(
        labels.astype(np.float64),
        if_scores,
        min_recall=cfg.model.target_min_recall,
    )
    _LOGGER.info(
        "threshold selected",
        extra={
            "threshold": round(choice.threshold, 6),
            "precision": round(choice.precision, 4),
            "recall": round(choice.recall, 4),
            "f1": round(choice.f1, 4),
            "alerts_per_1000": round(choice.alerts_per_1000, 2),
        },
    )
    return forest, if_scores, choice

# ---------------------------------------------------------------------------
# Autoencoder training
# ---------------------------------------------------------------------------
def _train_autoencoder(
    scaled: np.ndarray, cfg: AppConfig
) -> tuple[Autoencoder, np.ndarray]:
    """Fit the autoencoder and compute reconstruction errors.

    Args:
        scaled: Scaled feature matrix ``(n, n_features)``.
        cfg: Full application configuration.

    Returns:
        ``(autoencoder, ae_scores)`` where ``ae_scores`` are per-row MSE errors
        (higher = more anomalous).
    """
    n_features = scaled.shape[1]
    _LOGGER.info(
        "training Autoencoder",
        extra={
            "n_features": n_features,
            "hidden": 16,
            "latent": cfg.model.autoencoder_latent_dim,
            "epochs": cfg.model.autoencoder_epochs,
        },
    )
    t0 = time.perf_counter()
    ae = Autoencoder(
        n_features=n_features,
        hidden=16,
        latent=cfg.model.autoencoder_latent_dim,
    ).fit(
        scaled,
        epochs=cfg.model.autoencoder_epochs,
        batch_size=cfg.model.autoencoder_batch_size,
        learning_rate=cfg.model.autoencoder_learning_rate,
        seed=cfg.model.random_seed,
    )
    ae_scores = ae.reconstruction_error(scaled)
    _LOGGER.info(
        "Autoencoder fitted",
        extra={
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "mean_mse": round(float(ae_scores.mean()), 6),
        },
    )
    return ae, ae_scores

# ---------------------------------------------------------------------------
# Evaluation Report
# ---------------------------------------------------------------------------
def _write_report(
    path: Path,
    cfg: AppConfig,
    n_rows: int,
    n_fraud: int,
    if_pr_auc: float,
    if_roc_auc: float,
    choice: ThresholdChoice,
    ae_pr_auc: float | None = None,
) -> None:
    """Format and write the markdown evaluation report.

    Args:
        path: Destination path for the report.
        cfg: Full application configuration.
        n_rows: Total rows scored.
        n_fraud: Fraud rows scored.
        if_pr_auc: IsolationForest precision-recall AUC.
        if_roc_auc: IsolationForest ROC AUC.
        choice: Selected threshold operating point.
        ae_pr_auc: Autoencoder PR-AUC, if trained.
    """
    fraud_rate = n_fraud / max(1, n_rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    ae_section = ""
    if ae_pr_auc is not None:
        ae_section = f"""## Autoencoder
| Metric | Value |
|--------|-------|
| PR-AUC | {ae_pr_auc:.4f} |
"""

    report = f"""# Fraud Detection Model — Evaluation Report

Generated: {timestamp}

## Dataset
| Metric | Value |
|--------|-------|
| Rows sampled | {n_rows:,} |
| Fraud rows | {n_fraud:,} |
| Fraud rate | {fraud_rate:.6f} |

## Isolation Forest
| Metric | Value |
|--------|-------|
| PR-AUC | {if_pr_auc:.4f} |
| ROC-AUC | {if_roc_auc:.4f} |
| Threshold | {choice.threshold:.6f} |
| Precision | {choice.precision:.4f} |
| Recall | {choice.recall:.4f} |
| F1 | {choice.f1:.4f} |
| True Positives | {choice.true_positives:,} |
| False Positives | {choice.false_positives:,} |
| False Negatives | {choice.false_negatives:,} |
| Alerts per 1000 transactions | {choice.alerts_per_1000:.2f} |

{ae_section}
## Threshold Selection Rationale
The model targets a minimum recall floor of {cfg.model.target_min_recall * 100:.1f}%.
In fraud detection, minimising false negatives (missed fraud) matters more than
minimising false positives (false alarms), because the financial and regulatory cost
of a missed attack far outweighs the operational cost of an analyst reviewing an alert.
This operating point achieves {choice.precision * 100:.1f}% precision at the required recall,
generating {choice.alerts_per_1000:.2f} alerts per 1000 transactions. This provides a bounded
queue size for the fraud operations team while guaranteeing the recall target is met.

## Artifacts
Artifacts written to: `{cfg.model.artifact_path.resolve()}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    _LOGGER.info(
        "evaluation report written",
        extra={"path": str(path.resolve())},
    )


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run the training pipeline.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 on success).
    """
    args = _parse_args(argv)
    cfg = AppConfig.from_env()

    try:
        scaled, labels, scaler = _load_and_featurize(cfg, args.sample_rows)
        forest, if_scores, choice = _train_forest(scaled, labels, cfg)

        ae, ae_scores = None, None
        if not args.no_autoencoder:
            ae, ae_scores = _train_autoencoder(scaled, cfg)

        _LOGGER.info("saving model bundle", extra={"dir": str(args.artifact_dir)})
        bundle = ModelBundle(
            scaler=scaler,
            forest=forest,
            autoencoder=ae,
            feature_names=FEATURE_NAMES,
            velocity_window=cfg.features.velocity_window_steps,
            threshold=choice.threshold,
            metadata={"evaluation": choice.to_dict()},
        )
        save_bundle(bundle, args.artifact_dir)

        _LOGGER.info("computing evaluation metrics")
        labels_float = labels.astype(np.float64)
        if_pr_auc = average_precision(labels_float, if_scores)
        if_roc = roc_auc(labels_float, if_scores)
        ae_pr_auc = None
        if ae_scores is not None:
            ae_pr_auc = average_precision(labels_float, ae_scores)

        _write_report(
            args.report_path,
            cfg,
            n_rows=len(labels),
            n_fraud=int(labels.sum()),
            if_pr_auc=if_pr_auc,
            if_roc_auc=if_roc,
            choice=choice,
            ae_pr_auc=ae_pr_auc,
        )
        _LOGGER.info("training complete")
        return 0

    except DatasetError as exc:
        _LOGGER.error("dataset error", extra={"error": str(exc)})
        return 1
    except Exception as exc:
        _LOGGER.exception("unhandled error during training", extra={"error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
