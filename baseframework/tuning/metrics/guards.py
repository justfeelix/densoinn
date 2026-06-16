"""Guard rails for rejecting degenerate clusterings before ranking them."""

from __future__ import annotations

from typing import Any

from baseframework.tuning.types import EvaluationRecord


def basic_clustering_guard(
    record: EvaluationRecord,
    *,
    min_clusters: int = 2,
    min_labeled_points: int = 20,
    max_noise_frac: float = 0.98,
) -> tuple[bool, str, dict[str, float]]:
    """Reject clearly unusable clusterings before objective scoring."""

    diagnostics = record.diagnostics
    n_clusters = int(diagnostics.get("n_clusters", 0))
    n_points = int(diagnostics.get("n_points", 0))
    noise_frac = float(diagnostics.get("noise_frac", 1.0))
    labeled_points = int(round(float(diagnostics.get("labeled_frac", 0.0)) * n_points))

    details = {
        "n_clusters": float(n_clusters),
        "noise_frac": noise_frac,
        "labeled_points": float(labeled_points),
    }
    if n_clusters < min_clusters:
        return False, f"degenerate: {n_clusters} cluster(s)", details
    if labeled_points < min_labeled_points:
        return False, f"too few labeled points: {labeled_points}", details
    if noise_frac > max_noise_frac:
        return False, f"too much noise: {noise_frac:.3f}", details
    return True, "ok", details
