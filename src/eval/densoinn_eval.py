"""Pure evaluation helpers for DenSOINN parameter studies."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np

from baseframework.tuning.types import EvaluationRecord, RunArtifacts
from src.densoinn import DenSOINN


MetricFn = Callable[[np.ndarray, np.ndarray | None, RunArtifacts], float]


def iter_batches(
    X: np.ndarray,
    batch_size: int,
    points_per_timestep: int = 1,
):
    """Yield ``(batch_index, batch, timestamps)`` over the full stream."""

    n = len(X)
    num_batches = n // batch_size
    for batch_index in range(num_batches + 1):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, n)
        if start >= end:
            break
        batch = X[start:end]
        timestamps = np.arange(start, end) // max(points_per_timestep, 1)
        yield batch_index, batch, timestamps


def reconstruct_point_labels(
    densoinn: DenSOINN,
    X: np.ndarray,
    assignments: list[int],
    *,
    label_mode: str = "nearest_survivor",
) -> list[int]:
    """Reconstruct one final cluster label per point using the chosen strategy."""

    if label_mode == "nearest_survivor":
        return list(map(int, densoinn.predict_points(np.asarray(X))))
    if label_mode == "summary_lookup":
        return list(map(int, densoinn.predict(assignments=assignments)))
    raise ValueError(f"unsupported label_mode: {label_mode}")


def run_stream(
    X: np.ndarray,
    *,
    gamma: float,
    stream_speed: int,
    epsilon: float,
    max_nodes: int,
    alpha: float,
    intrinsic_dim: int,
    batch_size: int = 500,
    points_per_timestep: int = 1,
    label_mode: str = "nearest_survivor",
    on_batch=None,
):
    """Run DenSOINN on a full stream and return the model plus run artifacts."""

    densoinn = DenSOINN(
        gamma=gamma,
        stream_speed=stream_speed,
        epsilon=epsilon,
        max_nodes=max_nodes,
        alpha=alpha,
        intrinsic_dim=intrinsic_dim,
    )

    started = perf_counter()
    assignments: list[int] = []
    timestamps_seen: list[int] = []
    for batch_index, batch, timestamps in iter_batches(X, batch_size, points_per_timestep):
        results = densoinn.learn(batch, timestamps)
        assignments.extend(int(result[0]) for result in results)
        timestamps_seen.extend(int(timestamp) for timestamp in timestamps)
        if on_batch is not None:
            on_batch(batch_index, batch, timestamps, densoinn, assignments)

    summaries = densoinn.online_phase.get_summary()
    node_labels = densoinn.predict()
    point_labels = reconstruct_point_labels(
        densoinn,
        X,
        assignments,
        label_mode=label_mode,
    )
    runtime_sec = perf_counter() - started

    artifacts = RunArtifacts(
        assignments=assignments,
        timestamps=timestamps_seen,
        summaries=summaries,
        node_labels=node_labels,
        point_labels=point_labels,
        diagnostics={
            "runtime_sec": runtime_sec,
            "label_mode": label_mode,
        },
    )
    return densoinn, artifacts


def evaluate_params(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    gamma: float,
    stream_speed: int,
    epsilon: float,
    max_nodes: int,
    alpha: float,
    intrinsic_dim: int,
    batch_size: int = 500,
    points_per_timestep: int = 1,
    label_mode: str = "nearest_survivor",
    internal_metric_fns: dict[str, MetricFn] | None = None,
    external_metric_fns: dict[str, MetricFn] | None = None,
    return_artifacts: bool = False,
) -> EvaluationRecord:
    """Run DenSOINN once and collect metrics, diagnostics, and optional artifacts."""

    _, artifacts = run_stream(
        X,
        gamma=gamma,
        stream_speed=stream_speed,
        epsilon=epsilon,
        max_nodes=max_nodes,
        alpha=alpha,
        intrinsic_dim=intrinsic_dim,
        batch_size=batch_size,
        points_per_timestep=points_per_timestep,
        label_mode=label_mode,
    )

    diagnostics = _collect_diagnostics(
        labels=np.asarray(artifacts.point_labels, dtype=int),
        summaries=artifacts.summaries,
        assignments=artifacts.assignments,
    )
    diagnostics.update(artifacts.diagnostics)

    internal_metrics = _run_metric_bundle(
        metric_fns=internal_metric_fns,
        X=X,
        y=y,
        artifacts=artifacts,
    )
    external_metrics = _run_metric_bundle(
        metric_fns=external_metric_fns,
        X=X,
        y=y,
        artifacts=artifacts,
    )

    return EvaluationRecord(
        params={
            "gamma": gamma,
            "stream_speed": stream_speed,
            "epsilon": epsilon,
            "max_nodes": max_nodes,
            "alpha": alpha,
            "intrinsic_dim": intrinsic_dim,
        },
        internal_metrics=internal_metrics,
        external_metrics=external_metrics,
        diagnostics=diagnostics,
        artifacts=artifacts if return_artifacts else None,
    )


def _run_metric_bundle(
    *,
    metric_fns: dict[str, MetricFn] | None,
    X: np.ndarray,
    y: np.ndarray | None,
    artifacts: RunArtifacts,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, metric_fn in (metric_fns or {}).items():
        metrics[name] = float(metric_fn(X, y, artifacts))
    return metrics


def _collect_diagnostics(
    *,
    labels: np.ndarray,
    summaries: dict[int, dict[str, Any]],
    assignments: list[int],
    tiny_cluster_size: int = 5,
) -> dict[str, float]:
    n_points = int(len(labels))
    noise_mask = labels == -1
    non_noise = labels[~noise_mask]
    unique_clusters = np.unique(non_noise)
    cluster_sizes = np.array([np.sum(labels == cluster_id) for cluster_id in unique_clusters], dtype=int)

    n_singletons = int(np.sum(cluster_sizes == 1)) if len(cluster_sizes) else 0
    n_tiny = int(np.sum(cluster_sizes < tiny_cluster_size)) if len(cluster_sizes) else 0
    tiny_points = int(sum(size for size in cluster_sizes if size < tiny_cluster_size))

    return {
        "n_points": n_points,
        "n_clusters": int(len(unique_clusters)),
        "noise_frac": float(np.mean(noise_mask)) if n_points else 1.0,
        "labeled_frac": float(np.mean(~noise_mask)) if n_points else 0.0,
        "n_summaries": int(len(summaries)),
        "n_summary_components": int(_count_components(summaries)),
        "n_assigned_summaries": int(len(set(assignments))),
        "summary_ratio": float(len(summaries) / max(n_points, 1)),
        "min_cluster_size": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
        "max_cluster_size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "mean_cluster_size": float(cluster_sizes.mean()) if len(cluster_sizes) else 0.0,
        "median_cluster_size": float(np.median(cluster_sizes)) if len(cluster_sizes) else 0.0,
        "n_singleton_clusters": n_singletons,
        "singleton_cluster_frac": float(n_singletons / max(len(unique_clusters), 1)) if len(unique_clusters) else 0.0,
        "n_tiny_clusters": n_tiny,
        "tiny_cluster_frac": float(n_tiny / max(len(unique_clusters), 1)) if len(unique_clusters) else 0.0,
        "tiny_cluster_point_frac": float(tiny_points / max(int(np.sum(~noise_mask)), 1)) if len(non_noise) else 0.0,
    }


def _count_components(summaries: dict[int, dict[str, Any]]) -> int:
    visited: set[int] = set()
    n_components = 0
    for node_id in summaries:
        if node_id in visited:
            continue
        n_components += 1
        queue = deque([node_id])
        visited.add(node_id)
        while queue:
            current = queue.popleft()
            for neighbor in summaries[current]["neighbors"]:
                if neighbor in summaries and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return n_components
