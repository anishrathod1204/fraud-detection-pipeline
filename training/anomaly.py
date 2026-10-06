"""Unsupervised anomaly models, implemented in NumPy.

Both detectors are trained on transactions *without labels* - fraud is 0.13% of
the data, so a supervised classifier would fit the majority class and never learn
the boundary. Anomaly detection instead learns what normal looks like and flags
what does not fit. Labels are used only afterwards, offline, to choose an
operating threshold and to report precision and recall.

Two models ship, because they fail differently:

* :class:`IsolationForest` - the primary detector. It isolates an observation by
  recursive random splits and scores by how many splits it took; anomalies sit in
  sparse regions and are isolated in few splits. Robust to distance concentration
  in high dimension, which is exactly where a distance model struggles.
* :class:`Autoencoder` - a compact tanh network trained to reconstruct normal
  transactions. Its reconstruction error is a second anomaly score, on a
  different geometry; when the two agree a case is scored confidently, and the
  disagreement itself is informative on the dashboard.

Why hand-rolled
---------------
``scikit-learn`` would be the obvious choice and the pipeline is written to use
it - but this environment has no network access and no sklearn, and a training
job that cannot run cannot be verified. These implementations follow the papers
closely enough to be faithful (the forest reproduces Liu et al.'s path-length
score and sklearn's ``_average_path_length`` exactly; the autoencoder is ordinary
backprop) while depending only on NumPy, so the whole pipeline trains and is
tested here. The artifact format is plain JSON + ``.npz``, so swapping in
sklearn later would only change how the trees are built, not how they are served.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from common.logging_config import get_logger

__all__ = ["Autoencoder", "IsolationForest", "IsolationTree"]

_LOGGER = get_logger(__name__)

#: Euler-Mascheroni constant, used by the harmonic-number approximation in the
#: average path length of an unsuccessful BST search (equation 1 of the paper).
_EULER_GAMMA: Final[float] = 0.5772156649015329


def _average_path_length(n: np.ndarray | float) -> np.ndarray:
    """Expected path length of an unsuccessful search in a binary search tree.

    This is ``c(n)`` from Liu et al.; dividing an observed path length by it
    normalises scores so trees trained on different subsample sizes are
    comparable. The two special cases (n <= 1 and n == 2) are the paper's exact
    formulae; the general case uses the harmonic-number approximation.

    Args:
        n: Number of samples, scalar or array.

    Returns:
        ``c(n)``, same shape as the input.
    """
    n = np.asarray(n, dtype=np.float64)
    result = np.zeros_like(n)
    mask_1 = n <= 1.0
    mask_2 = np.isclose(n, 2.0)
    mask_gt = ~(mask_1 | mask_2)

    result[mask_2] = 1.0
    n_gt = n[mask_gt]
    harmonic = np.log(n_gt - 1.0) + _EULER_GAMMA
    result[mask_gt] = 2.0 * harmonic - (2.0 * (n_gt - 1.0) / n_gt)
    return result


class IsolationTree:
    """One isolation tree: a binary tree of random feature/threshold splits.

    Grown by :meth:`fit`. The split rule picks a feature uniformly at random and
    a threshold uniformly between that feature's observed min and max, recursing
    until every point is isolated or the depth limit is reached - a partial tree
    is what makes the method fast and why leaves carry a size for the path-length
    correction.
    """

    #: Recursion depth beyond which the tree stops is set per-fit; a tree grown
    #: to the subsample's expected depth is the paper's recommendation.
    __slots__ = ("_feature", "_threshold", "_left", "_right", "_size", "_depth")

    def __init__(self) -> None:
        """Create an empty tree node."""
        self._feature: int = -1
        self._threshold: float = 0.0
        self._left: IsolationTree | None = None
        self._right: IsolationTree | None = None
        self._size: int = 0
        self._depth: int = 0

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a leaf (holds a set of un-split points).

        Returns:
            ``True`` for a leaf.
        """
        return self._left is None

    def fit(
        self,
        samples: np.ndarray,
        depth: int,
        limit: int,
        rng: np.random.Generator,
    ) -> "IsolationTree":
        """Grow this subtree over a set of rows.

        Args:
            samples: ``(n_rows, n_features)`` array.
            depth: Current depth (root is 0).
            limit: Maximum depth, ``ceil(log2(subsample_size))``.
            rng: Seeded generator.

        Returns:
            ``self``, for chaining.
        """
        n_rows, n_features = samples.shape
        self._depth = depth
        self._size = n_rows

        # Stop when the node is small, deep enough, or every point is identical
        # on every feature (no split is possible).
        if depth >= limit or n_rows <= 1:
            return self
        widths = samples.max(axis=0) - samples.min(axis=0)
        splittable = np.flatnonzero(widths > 0.0)
        if splittable.size == 0:
            return self

        feature = int(rng.choice(splittable))
        column = samples[:, feature]
        low = float(column.min())
        high = float(column.max())
        threshold = float(rng.uniform(low, high))

        # A degenerate draw (threshold at an extreme) would send every point to
        # one side; resample once rather than recurse without progress.
        if threshold <= low or threshold >= high:
            threshold = (low + high) / 2.0

        left_mask = column < threshold
        right_mask = ~left_mask
        if not left_mask.any() or not right_mask.any():
            return self

        self._feature = feature
        self._threshold = threshold
        self._left = IsolationTree().fit(samples[left_mask], depth + 1, limit, rng)
        self._right = IsolationTree().fit(samples[right_mask], depth + 1, limit, rng)
        return self

    def path_lengths(self, samples: np.ndarray, depth: int = 0) -> np.ndarray:
        """Compute the isolation path length for each row.

        An external node (leaf) contributes its depth plus the expected path
        length of an unsuccessful search over its remaining points - the
        adjustment that makes partially grown trees unbiased.

        Args:
            samples: ``(n_rows, n_features)`` array.
            depth: Depth of this node, used internally during recursion.

        Returns:
            A length-``n_rows`` array of path lengths.
        """
        if self.is_leaf:
            return depth + _average_path_length(float(self._size))

        # Vectorised descent: only the rows routed left recurse left.
        left_mask = samples[:, self._feature] < self._threshold
        lengths = np.empty(samples.shape[0], dtype=np.float64)
        if left_mask.any():
            lengths[left_mask] = self._left.path_lengths(  # type: ignore[union-attr]
                samples[left_mask], depth + 1
            )
        if (~left_mask).any():
            lengths[~left_mask] = self._right.path_lengths(  # type: ignore[union-attr]
                samples[~left_mask], depth + 1
            )
        return lengths

    def to_dict(self) -> dict[str, Any]:
        """Serialise the tree to nested dicts.

        Returns:
            A JSON-friendly node, leaves carrying their size.
        """
        if self.is_leaf:
            return {"size": self._size}
        return {
            "feature": self._feature,
            "threshold": self._threshold,
            "left": self._left.to_dict(),  # type: ignore[union-attr]
            "right": self._right.to_dict(),  # type: ignore[union-attr]
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IsolationTree":
        """Rebuild a tree from :meth:`to_dict` output.

        Args:
            payload: The node dict.

        Returns:
            The reconstructed tree.
        """
        node = cls()
        if "size" in payload:
            node._size = int(payload["size"])
            return node
        node._feature = int(payload["feature"])
        node._threshold = float(payload["threshold"])
        node._left = cls.from_dict(payload["left"])
        node._right = cls.from_dict(payload["right"])
        return node


class IsolationForest:
    """An ensemble of isolation trees, scored by mean normalised path length.

    The anomaly score follows the paper: ``s(x, n) = 2 ** (-E[h(x)] / c(n))``,
    where ``E[h(x)]`` is the mean path length across trees and ``c(n)`` the
    expected path length at the training subsample size. Scores near 1 are
    anomalous; near 0.5, normal. The ensemble averages out the randomness of any
    single tree's splits.
    """

    __slots__ = (
        "_trees",
        "_n_estimators",
        "_max_samples",
        "_subsample_size",
        "_normaliser",
        "_n_features",
    )

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_samples: int = 65_536,
    ) -> None:
        """Configure the forest.

        Args:
            n_estimators: Number of trees.
            max_samples: Rows drawn per tree. The paper's small subsample (256 is
                the suggested default; larger skews toward normal points) is what
                makes isolation work - a full-data tree barely isolates anything.
        """
        self._trees: list[IsolationTree] = []
        self._n_estimators = int(n_estimators)
        self._max_samples = int(max_samples)
        self._subsample_size = 0
        self._normaliser = 1.0
        self._n_features = 0

    @property
    def n_features(self) -> int:
        """Number of features the forest was trained on.

        Returns:
            The fitted feature count, or 0 before fitting.
        """
        return self._n_features

    def fit(self, matrix: np.ndarray, *, seed: int = 42) -> "IsolationForest":
        """Grow the forest on a training matrix.

        Args:
            matrix: ``(n_rows, n_features)`` float array.
            seed: RNG seed; identical seeds give identical forests.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``matrix`` is empty or not 2-D.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("cannot fit IsolationForest on an empty matrix")

        n_rows, n_features = matrix.shape
        self._n_features = n_features
        subsample = min(self._max_samples, n_rows)
        self._subsample_size = subsample
        limit = int(math.ceil(math.log2(subsample))) if subsample > 1 else 0
        self._normaliser = float(_average_path_length(float(subsample)))

        rng = np.random.default_rng(seed)
        self._trees = []
        for _ in range(self._n_estimators):
            # Sample without replacement when the subsample covers most of the
            # data, with replacement otherwise - matching sklearn's behaviour and
            # keeping trees diverse.
            if subsample == n_rows:
                idx = rng.permutation(n_rows)
            else:
                idx = rng.choice(n_rows, size=subsample, replace=False)
            tree = IsolationTree().fit(matrix[idx], 0, limit, rng)
            self._trees.append(tree)

        _LOGGER.info(
            "isolation forest fitted",
            extra={
                "n_estimators": self._n_estimators,
                "subsample": subsample,
                "n_features": n_features,
            },
        )
        return self

    def score_samples(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the anomaly score for each row.

        Args:
            matrix: ``(n_rows, n_features)`` array.

        Returns:
            Scores in ``(0, 1]``; higher means more anomalous.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the feature count does not match training.
        """
        if not self._trees:
            raise RuntimeError("IsolationForest must be fitted before scoring")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape[1] != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features, got {matrix.shape[1]}"
            )

        # Mean path length across trees, then the paper's exponential score.
        total = np.zeros(matrix.shape[0], dtype=np.float64)
        for tree in self._trees:
            total += tree.path_lengths(matrix)
        mean_path = total / len(self._trees)
        denominator = self._normaliser if self._normaliser > 0.0 else 1.0
        return np.power(2.0, -mean_path / denominator)

    def decision_function(self, matrix: np.ndarray) -> np.ndarray:
        """Return a signed anomaly score: positive is anomalous.

        Mirrors sklearn's ``decision_function`` convention (anomaly score shifted
        by the offset), kept so downstream code reads the same either way.

        Args:
            matrix: ``(n_rows, n_features)`` array.

        Returns:
            Scores where larger means more anomalous; centred near 0.
        """
        return self.score_samples(matrix) - 0.5

    def depth_estimates(self, matrix: np.ndarray) -> np.ndarray:
        """Return the mean isolation depth, for explainability.

        The raw path length is more interpretable than the exponential score -
        "isolated in 3 splits" is meaningful to an operator in a way "score 0.72"
        is not.

        Args:
            matrix: ``(n_rows, n_features)`` array.

        Returns:
            Mean path length per row.
        """
        if not self._trees:
            raise RuntimeError("IsolationForest must be fitted before scoring")
        matrix = np.asarray(matrix, dtype=np.float64)
        total = np.zeros(matrix.shape[0], dtype=np.float64)
        for tree in self._trees:
            total += tree.path_lengths(matrix)
        return total / len(self._trees)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the forest to a JSON-friendly dict.

        Returns:
            The hyperparameters and the full tree ensemble.
        """
        return {
            "n_estimators": self._n_estimators,
            "max_samples": self._max_samples,
            "subsample_size": self._subsample_size,
            "normaliser": self._normaliser,
            "n_features": self._n_features,
            "trees": [tree.to_dict() for tree in self._trees],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IsolationForest":
        """Rebuild a forest from :meth:`to_dict` output.

        Args:
            payload: The serialised forest.

        Returns:
            The reconstructed :class:`IsolationForest`.
        """
        forest = cls(
            n_estimators=int(payload["n_estimators"]),
            max_samples=int(payload["max_samples"]),
        )
        forest._subsample_size = int(payload["subsample_size"])
        forest._normaliser = float(payload["normaliser"])
        forest._n_features = int(payload["n_features"])
        forest._trees = [IsolationTree.from_dict(node) for node in payload["trees"]]
        return forest


class Autoencoder:
    """A small tanh autoencoder, trained by backprop to reconstruct normal rows.

    Architecture is ``n_features -> hidden -> latent -> hidden -> n_features``,
    fully connected with ``tanh`` activations and a linear output. Trained only
    on the (normal-dominated) training sample, it learns the manifold of ordinary
    transactions; a fraudulent row, lying off that manifold, reconstructs poorly,
    and the per-row mean squared error becomes the anomaly score.

    Why tanh
    --------
    The inputs are standardised (zero mean, unit variance) but not bounded, and
    tanh's range and smooth gradient make it the conventional choice for a
    reconstruction autoencoder over standardised tabular data. The output layer is
    linear because the targets are unbounded standardised values, not
    probabilities.

    Training is plain mini-batch gradient descent with Adam. The network is small
    (a few thousand parameters), so this is fast and dependency-free, and it is
    deterministic given a seed - a property the evaluation report depends on.
    """

    __slots__ = (
        "_w1", "_b1", "_w2", "_b2", "_w3", "_b3", "_w4", "_b4",
        "_n_features", "_hidden", "_latent",
    )

    def __init__(
        self,
        *,
        n_features: int,
        hidden: int = 16,
        latent: int = 6,
    ) -> None:
        """Allocate an untrained network.

        Args:
            n_features: Input (and output) dimension.
            hidden: Width of the hidden layers.
            latent: Width of the bottleneck layer.
        """
        self._n_features = n_features
        self._hidden = hidden
        self._latent = latent
        # Weights are zero-initialised here; fit() reseeds them with a proper
        # scheme. Keeping the attribute set means the shape is defined up front.
        self._w1 = np.zeros((n_features, hidden))
        self._b1 = np.zeros(hidden)
        self._w2 = np.zeros((hidden, latent))
        self._b2 = np.zeros(latent)
        self._w3 = np.zeros((latent, hidden))
        self._b3 = np.zeros(hidden)
        self._w4 = np.zeros((hidden, n_features))
        self._b4 = np.zeros(n_features)

    def _init_parameters(self, rng: np.random.Generator) -> None:
        """Initialise weights with scaled uniform draws (Glorot-style).

        Args:
            rng: Seeded generator.
        """
        def glorot(fan_in: int, fan_out: int) -> np.ndarray:
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self._w1 = glorot(self._n_features, self._hidden)
        self._b1 = np.zeros(self._hidden)
        self._w2 = glorot(self._hidden, self._latent)
        self._b2 = np.zeros(self._latent)
        self._w3 = glorot(self._latent, self._hidden)
        self._b3 = np.zeros(self._hidden)
        self._w4 = glorot(self._hidden, self._n_features)
        self._b4 = np.zeros(self._n_features)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        """Run the forward pass, returning activations for backprop.

        Args:
            x: ``(batch, n_features)`` input.

        Returns:
            ``(h1, h2, h3, out)`` activations.
        """
        h1 = np.tanh(x @ self._w1 + self._b1)
        h2 = np.tanh(h1 @ self._w2 + self._b2)
        h3 = np.tanh(h2 @ self._w3 + self._b3)
        out = h3 @ self._w4 + self._b4
        return h1, h2, h3, out

    def fit(
        self,
        matrix: np.ndarray,
        *,
        epochs: int = 12,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> "Autoencoder":
        """Train the autoencoder to reconstruct the training matrix.

        Args:
            matrix: ``(n_rows, n_features)`` standardised array.
            epochs: Passes over the data.
            batch_size: Rows per gradient step.
            learning_rate: Adam step size.
            seed: RNG seed.

        Returns:
            ``self``, trained.

        Raises:
            ValueError: If ``matrix`` is empty or the width does not match.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("cannot fit Autoencoder on an empty matrix")
        if matrix.shape[1] != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features, got {matrix.shape[1]}"
            )

        rng = np.random.default_rng(seed)
        self._init_parameters(rng)

        # Adam optimiser state, one entry per parameter tensor.
        params = [self._w1, self._b1, self._w2, self._b2, self._w3, self._b3,
                  self._w4, self._b4]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        n_rows = matrix.shape[0]

        for epoch in range(epochs):
            order = rng.permutation(n_rows)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_rows, batch_size):
                idx = order[start:start + batch_size]
                x = matrix[idx]
                h1, h2, h3, out = self._forward(x)

                error = out - x
                epoch_loss += float(np.mean(error ** 2))
                n_batches += 1

                # Backprop through the linear output and three tanh layers;
                # d/dz tanh(z) = 1 - tanh(z)^2.
                scale = 2.0 * error / x.shape[0]
                g_w4 = h3.T @ scale
                g_b4 = scale.sum(axis=0)
                dh3 = (scale @ self._w4.T) * (1.0 - h3 ** 2)
                g_w3 = h2.T @ dh3
                g_b3 = dh3.sum(axis=0)
                dh2 = (dh3 @ self._w3.T) * (1.0 - h2 ** 2)
                g_w2 = h1.T @ dh2
                g_b2 = dh2.sum(axis=0)
                dh1 = (dh2 @ self._w2.T) * (1.0 - h1 ** 2)
                g_w1 = x.T @ dh1
                g_b1 = dh1.sum(axis=0)

                grads = [g_w1, g_b1, g_w2, g_b2, g_w3, g_b3, g_w4, g_b4]
                step += 1
                for i, (param, grad) in enumerate(zip(params, grads)):
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (grad ** 2)
                    m_hat = m[i] / (1.0 - beta1 ** step)
                    v_hat = v[i] / (1.0 - beta2 ** step)
                    param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        _LOGGER.info(
            "autoencoder trained",
            extra={"epochs": epochs, "final_loss": round(epoch_loss / max(1, n_batches), 6)},
        )
        return self

    def reconstruction_error(self, matrix: np.ndarray) -> np.ndarray:
        """Return per-row mean squared reconstruction error.

        Args:
            matrix: ``(n_rows, n_features)`` array.

        Returns:
            One error per row; larger means more anomalous.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        _, _, _, out = self._forward(matrix)
        return np.mean((out - matrix) ** 2, axis=1)

    def to_dict(self) -> dict[str, Any]:
        """Serialise weights to a JSON-friendly dict.

        Returns:
            Architecture hyperparameters and flat weight lists.
        """
        return {
            "n_features": self._n_features,
            "hidden": self._hidden,
            "latent": self._latent,
            "w1": self._w1.tolist(), "b1": self._b1.tolist(),
            "w2": self._w2.tolist(), "b2": self._b2.tolist(),
            "w3": self._w3.tolist(), "b3": self._b3.tolist(),
            "w4": self._w4.tolist(), "b4": self._b4.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Autoencoder":
        """Rebuild an autoencoder from :meth:`to_dict` output.

        Args:
            payload: The serialised network.

        Returns:
            The reconstructed :class:`Autoencoder`.
        """
        net = cls(
            n_features=int(payload["n_features"]),
            hidden=int(payload["hidden"]),
            latent=int(payload["latent"]),
        )
        net._w1 = np.asarray(payload["w1"], dtype=np.float64)
        net._b1 = np.asarray(payload["b1"], dtype=np.float64)
        net._w2 = np.asarray(payload["w2"], dtype=np.float64)
        net._b2 = np.asarray(payload["b2"], dtype=np.float64)
        net._w3 = np.asarray(payload["w3"], dtype=np.float64)
        net._b3 = np.asarray(payload["b3"], dtype=np.float64)
        net._w4 = np.asarray(payload["w4"], dtype=np.float64)
        net._b4 = np.asarray(payload["b4"], dtype=np.float64)
        return net

    def save(self, path: Path) -> None:
        """Write the network to a JSON file.

        Args:
            path: Destination path. Parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Autoencoder":
        """Load a network written by :meth:`save`.

        Args:
            path: Source path.

        Returns:
            The reconstructed :class:`Autoencoder`.
        """
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
