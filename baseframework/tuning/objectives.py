"""Objective functions that rank evaluation records without owning the run logic."""

from __future__ import annotations

import math

from baseframework.tuning.metrics.guards import basic_clustering_guard
from baseframework.tuning.types import EvaluationRecord, ObjectiveResult


def make_unsupervised_objective(
    *,
    metric_name: str = "ts",
    noise_penalty_weight: float = 0.25,
    tiny_cluster_penalty_weight: float = 0.25,
    fragmentation_penalty_weight: float = 0.5,
    summary_penalty_weight: float = 0.0,
    min_clusters: int = 2,
    min_labeled_points: int = 20,
    max_noise_frac: float = 0.98,
):
    """Rank candidates using label-free metrics plus structural guard rails."""

    def objective(record: EvaluationRecord) -> ObjectiveResult:
        valid, reason, details = basic_clustering_guard(
            record,
            min_clusters=min_clusters,
            min_labeled_points=min_labeled_points,
            max_noise_frac=max_noise_frac,
        )
        if not valid:
            return ObjectiveResult(
                name="unsupervised",
                score=float("-inf"),
                valid=False,
                reason=reason,
                details=details,
            )

        base_metric = float(record.internal_metrics.get(metric_name, float("nan")))
        if math.isnan(base_metric):
            return ObjectiveResult(
                name="unsupervised",
                score=float("-inf"),
                valid=False,
                reason=f"missing internal metric: {metric_name}",
                details=details,
            )

        noise_penalty = noise_penalty_weight * float(record.diagnostics.get("noise_frac", 0.0))
        tiny_penalty = tiny_cluster_penalty_weight * float(record.diagnostics.get("tiny_cluster_point_frac", 0.0))
        n_clusters = int(record.diagnostics.get("n_clusters", 0))
        n_components = int(record.diagnostics.get("n_summary_components", 0))
        n_summaries = int(record.diagnostics.get("n_summaries", 0))
        if n_summaries > n_components:
            fragmentation_excess = max(n_clusters - n_components, 0) / max(n_summaries - n_components, 1)
        else:
            fragmentation_excess = 0.0
        fragmentation_penalty = fragmentation_penalty_weight * fragmentation_excess
        summary_penalty = summary_penalty_weight * float(record.diagnostics.get("summary_ratio", 0.0))
        score = base_metric - noise_penalty - tiny_penalty - fragmentation_penalty - summary_penalty

        return ObjectiveResult(
            name="unsupervised",
            score=score,
            valid=True,
            reason="ok",
            details={
                "base_metric": base_metric,
                "noise_penalty": noise_penalty,
                "tiny_cluster_penalty": tiny_penalty,
                "fragmentation_penalty": fragmentation_penalty,
                "fragmentation_excess": fragmentation_excess,
                "summary_penalty": summary_penalty,
                **details,
            },
        )

    return objective


def make_unsupervised_v2_objective(
    *,
    ts_weight: float = 0.35,
    coverage_weight: float = 0.20,
    cluster_margin_weight: float = 0.25,
    peak_confidence_weight: float = 0.20,
    summary_ratio_penalty_weight: float = 0.20,
    min_clusters: int = 2,
    min_labeled_points: int = 20,
    max_noise_frac: float = 0.98,
):
    """Rank candidates with a small aggregate of TS, coverage, and DenSOINN-native diagnostics."""

    weights = {
        "ts_term": ts_weight,
        "coverage_term": coverage_weight,
        "cluster_margin_term": cluster_margin_weight,
        "peak_confidence_term": peak_confidence_weight,
    }
    if any(weight < 0.0 for weight in weights.values()) or summary_ratio_penalty_weight < 0.0:
        raise ValueError("unsupervised_v2 weights must be non-negative")
    if sum(weights.values()) <= 0.0:
        raise ValueError("unsupervised_v2 requires at least one positive weight")

    def objective(record: EvaluationRecord) -> ObjectiveResult:
        valid, reason, details = basic_clustering_guard(
            record,
            min_clusters=min_clusters,
            min_labeled_points=min_labeled_points,
            max_noise_frac=max_noise_frac,
        )
        if not valid:
            return ObjectiveResult(
                name="unsupervised_v2",
                score=float("-inf"),
                valid=False,
                reason=reason,
                details=details,
            )

        ts = float(record.internal_metrics.get("ts", float("nan")))
        coverage = float(record.diagnostics.get("labeled_frac", float("nan")))
        cluster_margin = float(record.diagnostics.get("cluster_margin", float("nan")))
        peak_confidence = float(record.diagnostics.get("peak_confidence", float("nan")))
        summary_ratio = float(record.diagnostics.get("summary_ratio", 0.0))
        if any(math.isnan(value) for value in (ts, coverage, cluster_margin)):
            return ObjectiveResult(
                name="unsupervised_v2",
                score=float("-inf"),
                valid=False,
                reason="missing unsupervised_v2 score term",
                details=details,
            )

        ts_term = _clip01((ts + 1.0) / 2.0)
        coverage_term = _clip01(coverage)
        cluster_margin_term = _clip01(cluster_margin)
        peak_confidence_term = float("nan") if math.isnan(peak_confidence) else _clip01(peak_confidence)
        summary_penalty = summary_ratio_penalty_weight * _clip01(summary_ratio)
        score_terms = {
            "ts_term": ts_term,
            "coverage_term": coverage_term,
            "cluster_margin_term": cluster_margin_term,
            "peak_confidence_term": peak_confidence_term,
        }
        weighted_sum = sum(
            weights[name] * value
            for name, value in score_terms.items()
            if weights[name] > 0.0 and not math.isnan(value)
        )
        active_weight = sum(
            weights[name]
            for name, value in score_terms.items()
            if weights[name] > 0.0 and not math.isnan(value)
        )
        if active_weight <= 0.0:
            return ObjectiveResult(
                name="unsupervised_v2",
                score=float("-inf"),
                valid=False,
                reason="no active unsupervised_v2 score terms",
                details=details,
            )
        score = weighted_sum / active_weight - summary_penalty

        return ObjectiveResult(
            name="unsupervised_v2",
            score=score,
            valid=True,
            reason="ok",
            details={
                "ts_raw": ts,
                "ts_term": ts_term,
                "coverage_term": coverage_term,
                "cluster_margin_term": cluster_margin_term,
                "peak_confidence_term": peak_confidence_term,
                "summary_penalty": summary_penalty,
                "active_weight": active_weight,
                **details,
            },
        )

    return objective


def make_unsupervised_v3_objective(
    *,
    ts_weight: float = 0.25,
    cluster_margin_weight: float = 0.25,
    cluster_separation_weight: float = 0.25,
    cluster_balance_weight: float = 0.15,
    peak_confidence_weight: float = 0.10,
    summary_ratio_penalty_weight: float = 0.30,
    tiny_cluster_penalty_weight: float = 0.10,
    min_clusters: int = 2,
    min_labeled_points: int = 20,
    max_noise_frac: float = 0.98,
):
    """Rank candidates with confidence plus compact label-free structure checks."""

    weights = {
        "ts_term": ts_weight,
        "cluster_margin_term": cluster_margin_weight,
        "cluster_separation_term": cluster_separation_weight,
        "cluster_balance_term": cluster_balance_weight,
        "peak_confidence_term": peak_confidence_weight,
    }
    penalties = {
        "summary_ratio_penalty_weight": summary_ratio_penalty_weight,
        "tiny_cluster_penalty_weight": tiny_cluster_penalty_weight,
    }
    has_negative_weight = any(weight < 0.0 for weight in weights.values())
    has_negative_penalty = any(weight < 0.0 for weight in penalties.values())
    if has_negative_weight or has_negative_penalty:
        raise ValueError("unsupervised_v3 weights must be non-negative")
    if sum(weights.values()) <= 0.0:
        raise ValueError("unsupervised_v3 requires at least one positive weight")

    def objective(record: EvaluationRecord) -> ObjectiveResult:
        valid, reason, details = basic_clustering_guard(
            record,
            min_clusters=min_clusters,
            min_labeled_points=min_labeled_points,
            max_noise_frac=max_noise_frac,
        )
        if not valid:
            return ObjectiveResult(
                name="unsupervised_v3",
                score=float("-inf"),
                valid=False,
                reason=reason,
                details=details,
            )

        ts = float(record.internal_metrics.get("ts", float("nan")))
        cluster_margin = float(record.diagnostics.get("cluster_margin", float("nan")))
        cluster_separation = float(record.diagnostics.get("cluster_separation", float("nan")))
        cluster_balance = float(record.diagnostics.get("cluster_size_balance", float("nan")))
        peak_confidence = float(record.diagnostics.get("peak_confidence", float("nan")))
        summary_ratio = float(record.diagnostics.get("summary_ratio", 0.0))
        tiny_cluster_point_frac = float(record.diagnostics.get("tiny_cluster_point_frac", 0.0))
        required_terms = (ts, cluster_margin, cluster_separation, cluster_balance)
        if any(math.isnan(value) for value in required_terms):
            return ObjectiveResult(
                name="unsupervised_v3",
                score=float("-inf"),
                valid=False,
                reason="missing unsupervised_v3 score term",
                details=details,
            )

        ts_term = _clip01((ts + 1.0) / 2.0)
        cluster_margin_term = _clip01(cluster_margin)
        cluster_separation_term = _clip01(cluster_separation)
        cluster_balance_term = _clip01(cluster_balance)
        peak_confidence_term = float("nan") if math.isnan(peak_confidence) else _clip01(peak_confidence)
        summary_penalty = summary_ratio_penalty_weight * _clip01(summary_ratio)
        tiny_cluster_penalty = tiny_cluster_penalty_weight * _clip01(tiny_cluster_point_frac)
        score_terms = {
            "ts_term": ts_term,
            "cluster_margin_term": cluster_margin_term,
            "cluster_separation_term": cluster_separation_term,
            "cluster_balance_term": cluster_balance_term,
            "peak_confidence_term": peak_confidence_term,
        }
        weighted_sum = sum(
            weights[name] * value
            for name, value in score_terms.items()
            if weights[name] > 0.0 and not math.isnan(value)
        )
        active_weight = sum(
            weights[name]
            for name, value in score_terms.items()
            if weights[name] > 0.0 and not math.isnan(value)
        )
        if active_weight <= 0.0:
            return ObjectiveResult(
                name="unsupervised_v3",
                score=float("-inf"),
                valid=False,
                reason="no active unsupervised_v3 score terms",
                details=details,
            )

        score = weighted_sum / active_weight - summary_penalty - tiny_cluster_penalty

        return ObjectiveResult(
            name="unsupervised_v3",
            score=score,
            valid=True,
            reason="ok",
            details={
                "ts_raw": ts,
                "ts_term": ts_term,
                "cluster_margin_term": cluster_margin_term,
                "cluster_separation_term": cluster_separation_term,
                "cluster_balance_term": cluster_balance_term,
                "peak_confidence_term": peak_confidence_term,
                "summary_penalty": summary_penalty,
                "tiny_cluster_penalty": tiny_cluster_penalty,
                "active_weight": active_weight,
                **details,
            },
        )

    return objective


def make_unsupervised_v4_objective(
    *,
    ts_weight: float = 0.15,
    cluster_margin_weight: float = 0.20,
    cluster_separation_weight: float = 0.25,
    cluster_balance_weight: float = 0.10,
    peak_confidence_weight: float = 0.05,
    summary_capacity_penalty_weight: float = 0.20,
    summary_shape_penalty_weight: float = 0.15,
    dominant_cluster_penalty_weight: float = 0.25,
    missed_split_penalty_weight: float = 0.20,
    fragmentation_penalty_weight: float = 0.15,
    tiny_cluster_penalty_weight: float = 0.10,
    summary_sqrt_low: float = 1.5,
    summary_sqrt_high: float = 12.0,
    summary_capacity_threshold: float = 0.85,
    dominant_cluster_threshold: float = 0.45,
    missed_split_threshold: float = 0.85,
    min_clusters: int = 2,
    min_labeled_points: int = 20,
    max_noise_frac: float = 0.98,
):
    """Rank candidates with stronger pressure against under-splitting and capacity artifacts."""

    weights = {
        "ts_term": ts_weight,
        "cluster_margin_term": cluster_margin_weight,
        "cluster_separation_term": cluster_separation_weight,
        "cluster_balance_term": cluster_balance_weight,
        "peak_confidence_term": peak_confidence_weight,
    }
    penalties = {
        "summary_capacity_penalty_weight": summary_capacity_penalty_weight,
        "summary_shape_penalty_weight": summary_shape_penalty_weight,
        "dominant_cluster_penalty_weight": dominant_cluster_penalty_weight,
        "missed_split_penalty_weight": missed_split_penalty_weight,
        "fragmentation_penalty_weight": fragmentation_penalty_weight,
        "tiny_cluster_penalty_weight": tiny_cluster_penalty_weight,
    }
    if any(weight < 0.0 for weight in weights.values()) or any(weight < 0.0 for weight in penalties.values()):
        raise ValueError("unsupervised_v4 weights must be non-negative")
    if sum(weights.values()) <= 0.0:
        raise ValueError("unsupervised_v4 requires at least one positive weight")
    if summary_sqrt_low <= 0.0 or summary_sqrt_high <= summary_sqrt_low:
        raise ValueError("unsupervised_v4 summary_sqrt bounds must satisfy 0 < low < high")

    def objective(record: EvaluationRecord) -> ObjectiveResult:
        valid, reason, details = basic_clustering_guard(
            record,
            min_clusters=min_clusters,
            min_labeled_points=min_labeled_points,
            max_noise_frac=max_noise_frac,
        )
        if not valid:
            return ObjectiveResult(
                name="unsupervised_v4",
                score=float("-inf"),
                valid=False,
                reason=reason,
                details=details,
            )

        ts = float(record.internal_metrics.get("ts", float("nan")))
        cluster_margin = float(record.diagnostics.get("cluster_margin", float("nan")))
        cluster_separation = float(record.diagnostics.get("cluster_separation", float("nan")))
        cluster_balance = float(record.diagnostics.get("cluster_size_balance", float("nan")))
        peak_confidence = float(record.diagnostics.get("peak_confidence", float("nan")))
        summary_capacity_frac = float(record.diagnostics.get("summary_capacity_frac", float("nan")))
        summary_sqrt_ratio = float(record.diagnostics.get("summary_sqrt_ratio", float("nan")))
        dominant_cluster_frac = float(record.diagnostics.get("dominant_cluster_frac", 0.0))
        missed_split_pressure = float(record.diagnostics.get("missed_split_pressure", 0.0))
        fragmentation_excess = float(record.diagnostics.get("fragmentation_excess", 0.0))
        tiny_cluster_point_frac = float(record.diagnostics.get("tiny_cluster_point_frac", 0.0))

        required_terms = (ts, cluster_margin, cluster_separation, cluster_balance)
        if any(math.isnan(value) for value in required_terms):
            return ObjectiveResult(
                name="unsupervised_v4",
                score=float("-inf"),
                valid=False,
                reason="missing unsupervised_v4 score term",
                details=details,
            )

        ts_term = _clip01((ts + 1.0) / 2.0)
        cluster_margin_term = _clip01(cluster_margin)
        cluster_separation_term = _clip01(cluster_separation)
        cluster_balance_term = _clip01(cluster_balance)
        peak_confidence_term = float("nan") if math.isnan(peak_confidence) else _clip01(peak_confidence)
        score_terms = {
            "ts_term": ts_term,
            "cluster_margin_term": cluster_margin_term,
            "cluster_separation_term": cluster_separation_term,
            "cluster_balance_term": cluster_balance_term,
            "peak_confidence_term": peak_confidence_term,
        }
        weighted_sum = sum(
            weights[name] * value
            for name, value in score_terms.items()
            if weights[name] > 0.0 and not math.isnan(value)
        )
        active_weight = sum(
            weights[name]
            for name, value in score_terms.items()
            if weights[name] > 0.0 and not math.isnan(value)
        )
        if active_weight <= 0.0:
            return ObjectiveResult(
                name="unsupervised_v4",
                score=float("-inf"),
                valid=False,
                reason="no active unsupervised_v4 score terms",
                details=details,
            )

        summary_capacity_pressure = (
            0.0
            if math.isnan(summary_capacity_frac)
            else _soft_excess(summary_capacity_frac, summary_capacity_threshold, 1.0)
        )
        summary_shape_pressure = 0.0
        if not math.isnan(summary_sqrt_ratio):
            low_pressure = _soft_shortfall(summary_sqrt_ratio, summary_sqrt_low)
            high_pressure = _soft_excess(summary_sqrt_ratio, summary_sqrt_high, 2.0 * summary_sqrt_high)
            summary_shape_pressure = max(low_pressure, high_pressure)
        dominant_cluster_pressure = _soft_excess(dominant_cluster_frac, dominant_cluster_threshold, 1.0)
        missed_split_pressure_term = _soft_excess(missed_split_pressure, missed_split_threshold, 1.0)

        summary_capacity_penalty = summary_capacity_penalty_weight * summary_capacity_pressure
        summary_shape_penalty = summary_shape_penalty_weight * summary_shape_pressure
        dominant_cluster_penalty = dominant_cluster_penalty_weight * dominant_cluster_pressure
        missed_split_penalty = missed_split_penalty_weight * missed_split_pressure_term
        fragmentation_penalty = fragmentation_penalty_weight * _clip01(fragmentation_excess)
        tiny_cluster_penalty = tiny_cluster_penalty_weight * _clip01(tiny_cluster_point_frac)

        score = (
            weighted_sum / active_weight
            - summary_capacity_penalty
            - summary_shape_penalty
            - dominant_cluster_penalty
            - missed_split_penalty
            - fragmentation_penalty
            - tiny_cluster_penalty
        )

        return ObjectiveResult(
            name="unsupervised_v4",
            score=score,
            valid=True,
            reason="ok",
            details={
                "ts_raw": ts,
                "ts_term": ts_term,
                "cluster_margin_term": cluster_margin_term,
                "cluster_separation_term": cluster_separation_term,
                "cluster_balance_term": cluster_balance_term,
                "peak_confidence_term": peak_confidence_term,
                "summary_capacity_pressure": summary_capacity_pressure,
                "summary_shape_pressure": summary_shape_pressure,
                "dominant_cluster_pressure": dominant_cluster_pressure,
                "missed_split_pressure_term": missed_split_pressure_term,
                "fragmentation_excess": _clip01(fragmentation_excess),
                "summary_capacity_penalty": summary_capacity_penalty,
                "summary_shape_penalty": summary_shape_penalty,
                "dominant_cluster_penalty": dominant_cluster_penalty,
                "missed_split_penalty": missed_split_penalty,
                "fragmentation_penalty": fragmentation_penalty,
                "tiny_cluster_penalty": tiny_cluster_penalty,
                "active_weight": active_weight,
                **details,
            },
        )

    return objective


def make_supervised_objective(
    *,
    metric_name: str = "cmm",
    count_penalty_weight: float = 0.5,
    min_clusters: int = 2,
    min_labeled_points: int = 20,
    max_noise_frac: float = 0.98,
):
    """Reproduce the label-aware selection logic, but keep it optional."""

    def objective(record: EvaluationRecord) -> ObjectiveResult:
        valid, reason, details = basic_clustering_guard(
            record,
            min_clusters=min_clusters,
            min_labeled_points=min_labeled_points,
            max_noise_frac=max_noise_frac,
        )
        if not valid:
            return ObjectiveResult(
                name="supervised",
                score=float("-inf"),
                valid=False,
                reason=reason,
                details=details,
            )

        base_metric = float(record.external_metrics.get(metric_name, float("nan")))
        true_k = int(record.external_metrics.get("true_k", 0))
        n_clusters = int(record.diagnostics.get("n_clusters", 0))
        if math.isnan(base_metric) or true_k <= 0:
            return ObjectiveResult(
                name="supervised",
                score=float("-inf"),
                valid=False,
                reason=f"missing external metric: {metric_name}",
                details=details,
            )

        count_penalty = count_penalty_weight * abs(n_clusters - true_k) / max(true_k, 1)
        score = base_metric - count_penalty
        return ObjectiveResult(
            name="supervised",
            score=score,
            valid=True,
            reason="ok",
            details={
                "base_metric": base_metric,
                "count_penalty": count_penalty,
                "true_k": float(true_k),
                **details,
            },
        )

    return objective


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _soft_excess(value: float, threshold: float, hard_limit: float) -> float:
    if hard_limit <= threshold:
        return 0.0
    return _clip01((value - threshold) / (hard_limit - threshold))


def _soft_shortfall(value: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 0.0
    return _clip01((threshold - value) / threshold)
