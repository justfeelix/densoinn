"""Internal, label-free metrics for tuning stream-clustering models."""

from __future__ import annotations

from functools import partial

import numpy as np

from baseframework.metrics.TSindex import tempsil
from baseframework.tuning.types import RunArtifacts


def temporal_silhouette_metric(
    X: np.ndarray,
    y: np.ndarray | None,
    artifacts: RunArtifacts,
    *,
    s: int = 200,
    kn: int = 200,
    c: float = 1.0,
    ignore_noise: bool = True,
) -> float:
    """Compute Temporal Silhouette on final point labels without ground truth."""

    del y
    points = np.asarray(X)
    timestamps = np.asarray(artifacts.timestamps)
    labels = np.asarray(artifacts.point_labels, dtype=int)

    if ignore_noise:
        mask = labels != -1
        points = points[mask]
        timestamps = timestamps[mask]
        labels = labels[mask]

    if len(points) == 0:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")

    effective_kn = max(1, min(kn, len(points) - 1))
    _, _, score = tempsil(timestamps, points, labels, s=s, kn=effective_kn, c=c)
    return float(score)


def make_internal_metric_bundle(
    *,
    ts_window: int = 200,
    ts_kn: int = 200,
    ts_c: float = 1.0,
    ignore_noise: bool = True,
) -> dict[str, callable]:
    """Return the default label-free metric bundle for tuning."""

    return {
        "ts": partial(
            temporal_silhouette_metric,
            s=ts_window,
            kn=ts_kn,
            c=ts_c,
            ignore_noise=ignore_noise,
        ),
    }
