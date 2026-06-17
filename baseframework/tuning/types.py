"""Core dataclasses shared by tuning workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunArtifacts:
    """Large evaluation outputs that are useful for inspection but not ranking."""

    assignments: list[int]
    timestamps: list[int]
    summaries: dict[int, dict[str, Any]]
    node_labels: dict[int, int]
    point_labels: list[int]
    offline_summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """Metric and diagnostic outputs for one parameter configuration."""

    params: dict[str, Any]
    internal_metrics: dict[str, float] = field(default_factory=dict)
    external_metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: RunArtifacts | None = None


@dataclass(frozen=True)
class ObjectiveResult:
    """Objective-specific ranking outcome derived from an evaluation record."""

    name: str
    score: float
    valid: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Compact, serializable result used for ranking and reporting."""

    params: dict[str, Any]
    objective_name: str
    score: float
    valid: bool
    reason: str
    objective_details: dict[str, Any] = field(default_factory=dict)
    internal_metrics: dict[str, float] = field(default_factory=dict)
    external_metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_evaluation(
        cls,
        evaluation: EvaluationRecord,
        objective: ObjectiveResult,
    ) -> "SearchResult":
        return cls(
            params=dict(evaluation.params),
            objective_name=objective.name,
            score=float(objective.score),
            valid=bool(objective.valid),
            reason=objective.reason,
            objective_details=dict(objective.details),
            internal_metrics=dict(evaluation.internal_metrics),
            external_metrics=dict(evaluation.external_metrics),
            diagnostics=dict(evaluation.diagnostics),
        )

    @classmethod
    def from_failure(
        cls,
        *,
        params: dict[str, Any],
        reason: str,
        error: str | None = None,
    ) -> "SearchResult":
        details = {"error": error} if error is not None else {}
        return cls(
            params=dict(params),
            objective_name="error",
            score=float("-inf"),
            valid=False,
            reason=reason,
            objective_details=details,
        )
