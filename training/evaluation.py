"""Ranking and threshold metrics for the fraud model, in NumPy.

The model is unsupervised, so the training job never optimises against labels.
Labels are used exactly once, offline and after the fact, to answer two
questions the model itself cannot: *where should the alert threshold sit* and
*how good is the detector once it sits there*. This module is the whole of that
evaluation, kept separate from the training driver so the metric definitions are
testable in isolation and cannot quietly change under the pipeline.

Everything here is hand-rolled for the same reason the models are: this
environment has no scikit-learn, and a metric that cannot run cannot be trusted.
Each function is written to agree with its scikit-learn counterpart to numerical
precision, and a test pins that agreement on cases with known answers.

Two metrics matter for a fraud detector:

precision-recall area (``average_precision``)
    The honest headline. Fraud is 0.13% of traffic, so ROC-AUC is dominated by
    the vast true-negative mass and reads reassuringly high for a detector that
    is useless in practice. PR-AUC weights the rare positive class directly, so
    it falls to the base rate when the detector is no better than chance and
    rises only when detections are genuinely enriched. It is the metric the
    model card reports first.

recall-first threshold selection (:func:`select_threshold`)
    A fraud team sets a recall floor - "catch at least 95% of fraud" - and then
    wants the fewest false alarms that still clears it. So the choice is: among
    every operating point on the curve, take those meeting the recall floor and
    pick the one with the highest precision. Higher thresholds buy precision at
    the cost of recall; the floor is the constraint, precision is the objective.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "ThresholdChoice",
    "average_precision",
    "confusion_at_threshold",
    "precision_recall_curve",
    "roc_auc",
    "select_threshold",
]

#: Guard against dividing by zero when a class is absent from a split. Small
#: enough never to perturb a real ratio.
_EPSILON: Final[float] = 1e-12


class ThresholdChoice:
    """An operating point and the confusion counts it produces.

    A plain container rather than a dict so the training driver and the report
    writer share one typed object, and the field names here are the field names
    in the evaluation report.

    Attributes:
        threshold: Predict positive when the score is ``>= threshold``.
        precision: True positives over predicted positives.
        recall: True positives over actual positives.
        f1: Harmonic mean of precision and recall.
        true_positives: Correctly flagged frauds.
        false_positives: Legitimate transactions flagged.
        true_negatives: Legitimate transactions passed.
        false_negatives: Frauds missed.
        alerts_per_1000: Flags per thousand transactions scored - the operator's
            lived cost of the threshold, more legible than a raw count.
    """

    __slots__ = (
        "threshold",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "alerts_per_1000",
    )

    def __init__(
        self,
        *,
        threshold: float,
        precision: float,
        recall: float,
        f1: float,
        true_positives: int,
        false_positives: int,
        true_negatives: int,
        false_negatives: int,
        alerts_per_1000: float,
    ) -> None:
        """Store a computed operating point. Use :func:`select_threshold` or
        :func:`confusion_at_threshold` to build one rather than calling this
        directly."""
        self.threshold = threshold
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
        self.alerts_per_1000 = alerts_per_1000

    def to_dict(self) -> dict[str, float | int]:
        """Serialise the choice for the evaluation report.

        Returns:
            A JSON-friendly dict of every field.
        """
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "alerts_per_1000": self.alerts_per_1000,
        }


def _validate(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and check a label/score pair.

    Args:
        labels: Ground-truth 0/1 array.
        scores: Anomaly scores, higher meaning more anomalous.

    Returns:
        ``(labels, scores)`` as float64 arrays of equal length.

    Raises:
        ValueError: If lengths differ or the arrays are empty.
    """
    labels = np.asarray(labels, dtype=np.float64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"labels and scores must be the same length, got "
            f"{labels.shape[0]} and {scores.shape[0]}"
        )
    if labels.shape[0] == 0:
        raise ValueError("cannot compute metrics on empty input")
    return labels, scores


def precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the precision-recall curve by sweeping the score cut-off.

    Follows the scikit-learn convention: rows are ordered by descending score,
    and each prefix of that order is treated as the set flagged at the
    corresponding threshold. The returned arrays are padded with the
    ``(precision=1, recall=0)`` sentinel the caller and ``average_precision``
    both expect.

    Args:
        labels: Ground-truth 0/1 array.
        scores: Anomaly scores, higher meaning more anomalous.

    Returns:
        ``(precision, recall, thresholds)``. ``thresholds`` is one shorter than
        the other two, matching scikit-learn - each threshold is the score at
        which a transaction stops being flagged.

    Raises:
        ValueError: On length mismatch or empty input.
    """
    labels, scores = _validate(labels, scores)
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    true_positives = np.cumsum(labels_sorted)
    false_positives = np.cumsum(1.0 - labels_sorted)
    total_positives = labels_sorted.sum()

    precision = true_positives / np.maximum(true_positives + false_positives, _EPSILON)
    recall = true_positives / max(total_positives, _EPSILON)

    # Pad with the (1, 0) sentinel so the curve has a defined start.
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return precision, recall, scores_sorted


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the area under the precision-recall curve (PR-AUC).

    Uses the step-wise sum of precision weighted by the increase in recall
    between consecutive operating points - the same definition as
    ``sklearn.metrics.average_precision_score`` - rather than a trapezoidal
    rule, which would credit the curve between two points with no recall gain.

    Args:
        labels: Ground-truth 0/1 array.
        scores: Anomaly scores, higher meaning more anomalous.

    Returns:
        PR-AUC in ``[0, 1]``. Returns ``0.0`` when the input holds no positive
        labels, since the curve is undefined and a detector cannot be credited
        for finding nothing.

    Raises:
        ValueError: On length mismatch or empty input.
    """
    labels, scores = _validate(labels, scores)
    if labels.sum() == 0:
        return 0.0

    precision, recall, _ = precision_recall_curve(labels, scores)

    # Enforce the monotone-decreasing precision the step integral assumes: for
    # each recall level take the best precision achieved at or above it.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # Sum precision over the recall increments. np.diff over a decreasing-then
    # clipped recall can be negative only through float noise; the mask drops it.
    recall_steps = np.diff(recall)
    valid = recall_steps > 0.0
    return float(np.sum(precision[1:][valid] * recall_steps[valid]))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the area under the ROC curve, by rank.

    Implemented as the Mann-Whitney U statistic: the probability that a random
    positive is scored above a random negative, which is exactly the ROC area.
    Ties are handled by assigning average ranks, so constant scores yield 0.5 as
    they should.

    ROC-AUC is reported for comparison with the literature but is not the metric
    to steer by here - see the module docstring.

    Args:
        labels: Ground-truth 0/1 array.
        scores: Anomaly scores, higher meaning more anomalous.

    Returns:
        ROC-AUC in ``[0, 1]``, or ``nan`` when a class is absent (the quantity is
        undefined, and ``nan`` propagates honestly into the report).

    Raises:
        ValueError: On length mismatch or empty input.
    """
    labels, scores = _validate(labels, scores)
    n_positives = int(labels.sum())
    n_negatives = labels.shape[0] - n_positives
    if n_positives == 0 or n_negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)

    # Average the ranks of tied scores so a tie is neither rewarded nor punished.
    index = 0
    n = sorted_scores.shape[0]
    while index < n:
        end = index
        while end + 1 < n and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        if end > index:
            average_rank = (index + 1 + end + 1) / 2.0
            ranks[order[index : end + 1]] = average_rank
        index = end + 1

    positive_rank_sum = ranks[labels == 1].sum()
    u_statistic = positive_rank_sum - n_positives * (n_positives + 1) / 2.0
    return float(u_statistic / (n_positives * n_negatives))


def confusion_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> ThresholdChoice:
    """Count the confusion matrix at a fixed threshold.

    Args:
        labels: Ground-truth 0/1 array.
        scores: Anomaly scores, higher meaning more anomalous.
        threshold: Predict positive when the score is ``>= threshold``.

    Returns:
        A :class:`ThresholdChoice` with counts and derived rates.

    Raises:
        ValueError: On length mismatch or empty input.
    """
    labels, scores = _validate(labels, scores)
    predicted = scores >= threshold

    true_positives = int(np.sum(predicted & (labels == 1)))
    false_positives = int(np.sum(predicted & (labels == 0)))
    false_negatives = int(np.sum(~predicted & (labels == 1)))
    true_negatives = int(np.sum(~predicted & (labels == 0)))

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    alerts_per_1000 = 1000.0 * (true_positives + false_positives) / labels.shape[0]

    return ThresholdChoice(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        alerts_per_1000=float(alerts_per_1000),
    )


def select_threshold(
    labels: np.ndarray, scores: np.ndarray, *, min_recall: float
) -> ThresholdChoice:
    """Choose the most precise threshold that still meets a recall floor.

    Sweeps every distinct score as a candidate cut-off, keeps the operating
    points whose recall is at least ``min_recall``, and returns the one with the
    highest precision. Because precision rises and recall falls as the threshold
    rises, this is the operating point an analyst wants: the fewest false alarms
    that still catch the required share of fraud.

    Args:
        labels: Ground-truth 0/1 array.
        scores: Anomaly scores, higher meaning more anomalous.
        min_recall: Required recall floor in ``(0, 1]``.

    Returns:
        The chosen :class:`ThresholdChoice`. If even the lowest threshold cannot
        reach the floor - the detector simply misses too much fraud - the point
        of maximum achievable recall is returned instead, so the caller gets the
        best available rather than an error.

    Raises:
        ValueError: On length mismatch, empty input, or a floor outside
            ``(0, 1]``.
    """
    if not 0.0 < min_recall <= 1.0:
        raise ValueError(f"min_recall must be in (0, 1], got {min_recall}")
    labels, scores = _validate(labels, scores)
    if labels.sum() == 0:
        raise ValueError("cannot select a threshold without any positive labels")

    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    cumulative_positives = np.cumsum(labels_sorted)
    cumulative_flagged = np.arange(1, labels_sorted.shape[0] + 1, dtype=np.float64)
    recall = cumulative_positives / labels_sorted.sum()
    precision = cumulative_positives / cumulative_flagged

    meets_floor = np.flatnonzero(recall >= min_recall)
    if meets_floor.size:
        # Highest precision among the qualifying prefixes. Ties resolve to the
        # earliest (highest) threshold, i.e. the fewest alerts.
        candidate_indices = meets_floor
        best_local = int(np.argmax(precision[candidate_indices]))
        best_index = int(candidate_indices[best_local])
    else:
        # Floor unreachable: report the best recall actually available.
        best_index = int(np.argmax(recall))

    threshold = float(scores_sorted[best_index])
    # Recompute against the exact ``>= threshold`` rule, which may admit a few
    # tied scores beyond the chosen prefix. The recomputed counts are what the
    # report shows, so the two never disagree.
    return confusion_at_threshold(labels, scores, threshold)
