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
