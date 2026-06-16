"""Validation metrics that require ground-truth labels."""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from baseframework.metrics.cmm import get_cmm
from baseframework.tuning.types import RunArtifacts


def cmm_metric(
    X: np.ndarray,
    y: np.ndarray | None,
    artifacts: RunArtifacts,
    *,
    k: int = 5,
) -> float:
    """Compute CMM when ground-truth labels are available."""

    if y is None:
        return float("nan")
    weights = np.ones(len(X))
    return float(get_cmm(np.asarray(X), np.asarray(y), np.asarray(artifacts.point_labels), weights, k=k))


def nmi_metric(X: np.ndarray, y: np.ndarray | None, artifacts: RunArtifacts) -> float:
    """Compute normalized mutual information on final point labels."""

    del X
    if y is None:
        return float("nan")
    return float(normalized_mutual_info_score(np.asarray(y), np.asarray(artifacts.point_labels)))


def ari_metric(X: np.ndarray, y: np.ndarray | None, artifacts: RunArtifacts) -> float:
    """Compute adjusted Rand index on final point labels."""

    del X
    if y is None:
        return float("nan")
    return float(adjusted_rand_score(np.asarray(y), np.asarray(artifacts.point_labels)))


def purity_metric(X: np.ndarray, y: np.ndarray | None, artifacts: RunArtifacts) -> float:
    """Paper-style purity, ignoring the noise label as a cluster."""

    del X
    if y is None:
        return float("nan")
    return float(_purity(np.asarray(y), np.asarray(artifacts.point_labels)))


def accuracy_metric(X: np.ndarray, y: np.ndarray | None, artifacts: RunArtifacts) -> float:
    """Paper-style accuracy with noise excluded from the denominator."""

    del X
    if y is None:
        return float("nan")
    return float(_accuracy(np.asarray(y), np.asarray(artifacts.point_labels)))


def true_k_metric(X: np.ndarray, y: np.ndarray | None, artifacts: RunArtifacts) -> float:
    """Expose the number of non-noise ground-truth classes for supervised objectives."""

    del X, artifacts
    if y is None:
        return 0.0
    y_arr = np.asarray(y)
    return float(len(np.unique(y_arr[y_arr != -1])))


def make_external_metric_bundle(
    *,
    cmm_k: int = 5,
) -> dict[str, callable]:
    """Return the default validation metric bundle."""

    return {
        "cmm": partial(cmm_metric, k=cmm_k),
        "nmi": nmi_metric,
        "ari": ari_metric,
        "purity": purity_metric,
        "accuracy": accuracy_metric,
        "true_k": true_k_metric,
    }


def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    correct = 0
    for cluster_id in np.unique(y_pred):
        if cluster_id == -1:
            continue
        members = y_true[y_pred == cluster_id]
        if len(members):
            _, counts = np.unique(members, return_counts=True)
            correct += int(counts.max())
    return correct / n


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = 0
    den = 0
    for cluster_id in np.unique(y_pred):
        if cluster_id == -1:
            continue
        members = y_true[y_pred == cluster_id]
        if len(members):
            _, counts = np.unique(members, return_counts=True)
            num += int(counts.max())
            den += int(len(members))
    return num / den if den > 0 else 0.0
