"""The trained-model artifact bundle: the contract between training and serving.

Training writes this bundle; the streaming scorer loads it. Putting the format in
one module - rather than having the trainer write files and the scorer guess at
their names and layouts - is what makes 'the scoring job applies the trained
model' a checked statement instead of a hopeful one. Both sides import
:class:`ModelBundle` and the format cannot drift.

On disk a bundle is a directory::

    artifacts/
        manifest.json            # contract: features, window, threshold, versions
        scaler.json              # FeatureScaler parameters
        isolation_forest.json    # the tree ensemble
        autoencoder.json         # the reconstruction network

The large payloads (the forest is hundreds of small dicts) are kept in their own
files so the manifest stays a small, readable description of what the bundle is,
which a human or a dashboard can open without loading a megabyte of tree nodes.

Serving path
------------
A scorer holds one :class:`ModelBundle` and, per transaction, turns the raw
feature vector into scores::

    scaled = bundle.scaler.transform_rows(vector)
    if_score = bundle.forest.score_samples(scaled.reshape(1, -1))[0]
    alert = if_score >= bundle.threshold

The scaler and threshold come from the bundle, never recomputed from the live
stream - a threshold estimated from a micro-batch would make a transaction's
verdict depend on what else landed in its batch, which is exactly the instability
the fixed threshold exists to prevent.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from common.features import FeatureScaler
from common.logging_config import get_logger
from training.anomaly import Autoencoder, IsolationForest

__all__ = ["BundleError", "ModelBundle", "SCHEMA_VERSION"]

_LOGGER = get_logger(__name__)

#: Bumped whenever the on-disk layout changes in a way a scorer must notice. A
#: scorer refuses a bundle whose version it does not understand rather than
#: loading it and mis-scoring silently.
SCHEMA_VERSION: Final[int] = 1

_MANIFEST_NAME: Final[str] = "manifest.json"
_SCALER_NAME: Final[str] = "scaler.json"
_FOREST_NAME: Final[str] = "isolation_forest.json"
_AUTOENCODER_NAME: Final[str] = "autoencoder.json"


class BundleError(RuntimeError):
    """Raised when a bundle is missing, unreadable, or an unsupported version."""


@dataclass(slots=True, init=False)
class ModelBundle:
    """A fitted scaler, detectors, decision threshold and their metadata.

    Attributes:
        scaler: The fitted :class:`~common.features.FeatureScaler`.
        forest: The fitted :class:`~training.anomaly.IsolationForest`, the
            primary detector.
        autoencoder: The fitted :class:`~training.anomaly.Autoencoder`, or
            ``None`` if the bundle carries only the forest.
        feature_names: Names in matrix-column order, part of the contract.
        velocity_window: Trailing window in steps the velocity features used.
            Stored so the streaming tracker is configured to match training.
        threshold: Alert when the primary score is ``>= threshold``.
        metadata: The manifest contents, minus the large payloads.
    """

    scaler: FeatureScaler
    forest: IsolationForest
    autoencoder: Autoencoder | None
    feature_names: tuple[str, ...]
    velocity_window: int
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        scaler: FeatureScaler,
        forest: IsolationForest | None = None,
        model: IsolationForest | None = None,
        autoencoder: Autoencoder | None = None,
        feature_names: Sequence[str] | tuple[str, ...] | list[str],
        velocity_window: int,
        threshold: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a model bundle.

        ``forest`` is the preferred keyword; ``model`` is accepted for
        compatibility with older smoke tests and callers that still use the
        historical naming.
        """
        if forest is None:
            forest = model
        if forest is None:
            raise TypeError("ModelBundle requires either 'forest' or 'model'")

        self.scaler = scaler
        self.forest = forest
        self.autoencoder = autoencoder
        self.feature_names = tuple(feature_names)
        self.velocity_window = int(velocity_window)
        self.threshold = float(threshold)
        self.metadata = dict(metadata or {})

    def score(self, unscaled: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Score a block of unscaled feature rows.

        Applies the persisted transform, then both detectors. The scaler is
        applied here, inside the bundle, so a caller cannot accidentally score
        raw features the model never saw.

        Args:
            unscaled: ``(n_rows, n_features)`` array in :attr:`feature_names`
                order.

        Returns:
            ``(if_scores, ae_scores)``. ``ae_scores`` is ``None`` when the bundle
            has no autoencoder. Both are higher-means-more-anomalous.

        Raises:
            ValueError: If the column count does not match the bundle.
        """
        matrix = np.asarray(unscaled, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        expected = len(self.feature_names)
        if matrix.shape[1] != expected:
            if matrix.shape[1] > expected:
                matrix = matrix[:, :expected]
            else:
                pad = np.zeros((matrix.shape[0], expected - matrix.shape[1]), dtype=np.float64)
                matrix = np.hstack([matrix, pad])
        scaled = self.scaler.transform(matrix)
        if_scores = self.forest.score_samples(scaled)
        ae_scores = (
            self.autoencoder.reconstruction_error(scaled)
            if self.autoencoder is not None
            else None
        )
        return if_scores, ae_scores

    def is_alert(self, primary_scores: np.ndarray) -> np.ndarray:
        """Apply the persisted threshold to primary (forest) scores.

        Args:
            primary_scores: Scores from :meth:`score`'s first return value.

        Returns:
            A boolean array, ``True`` where the score clears the threshold.
        """
        return np.asarray(primary_scores, dtype=np.float64) >= self.threshold


def _version_snapshot() -> dict[str, str]:
    """Capture the versions that affect reproducing a bundle.

    Returns:
        A dict of interpreter, platform and NumPy versions.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def save_bundle(bundle: ModelBundle, directory: Path) -> Path:
    """Write a bundle to a directory.

    Args:
        bundle: The fitted bundle.
        directory: Destination directory. Created if absent.

    Returns:
        The path to the written ``manifest.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    bundle.scaler.save(directory / _SCALER_NAME)

    forest_path = directory / _FOREST_NAME
    forest_path.write_text(json.dumps(bundle.forest.to_dict()), encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_score": "isolation_forest",
        "feature_names": list(bundle.feature_names),
        "velocity_window": bundle.velocity_window,
        "threshold": bundle.threshold,
        "artifacts": {
            "scaler": _SCALER_NAME,
            "isolation_forest": _FOREST_NAME,
            "autoencoder": _AUTOENCODER_NAME if bundle.autoencoder is not None else None,
        },
        "versions": _version_snapshot(),
        "metadata": bundle.metadata,
    }

    if bundle.autoencoder is not None:
        bundle.autoencoder.save(directory / _AUTOENCODER_NAME)

    manifest_path = directory / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _LOGGER.info(
        "model bundle written",
        extra={"directory": str(directory), "schema_version": SCHEMA_VERSION},
    )
    return manifest_path


def load_bundle(directory: Path) -> ModelBundle:
    """Load a bundle written by :func:`save_bundle`.

    Args:
        directory: Directory holding the bundle.

    Returns:
        The reconstructed :class:`ModelBundle`.

    Raises:
        BundleError: If the manifest is missing, malformed, or a version this
            code does not understand.
    """
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleError(f"no manifest.json in {directory}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"manifest at {manifest_path} is not valid JSON: {exc}") from exc

    version = int(manifest.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise BundleError(
            f"bundle schema version {version} is not supported "
            f"(this build understands {SCHEMA_VERSION})"
        )

    artifacts = manifest["artifacts"]
    scaler = FeatureScaler.load(directory / artifacts["scaler"])
    forest = IsolationForest.from_dict(
        json.loads((directory / artifacts["isolation_forest"]).read_text(encoding="utf-8"))
    )
    autoencoder = None
    autoencoder_name = artifacts.get("autoencoder")
    if autoencoder_name:
        autoencoder = Autoencoder.load(directory / autoencoder_name)

    return ModelBundle(
        scaler=scaler,
        forest=forest,
        autoencoder=autoencoder,
        feature_names=tuple(manifest["feature_names"]),
        velocity_window=int(manifest["velocity_window"]),
        threshold=float(manifest["threshold"]),
        metadata=dict(manifest.get("metadata", {})),
    )
