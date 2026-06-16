"""Objective-agnostic tuning engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .types import EvaluationRecord, ObjectiveResult, SearchResult


EvaluatorFn = Callable[..., EvaluationRecord]
ObjectiveFn = Callable[[EvaluationRecord], ObjectiveResult]
ReporterFn = Callable[[SearchResult], None]


def evaluate_candidate(
    evaluator: EvaluatorFn,
    objective: ObjectiveFn,
    params: dict[str, Any],
    *,
    fixed: dict[str, Any] | None = None,
) -> SearchResult:
    """Run one evaluation and derive its ranking score."""

    merged = {**(fixed or {}), **params}
    evaluation = evaluator(**merged)
    objective_result = objective(evaluation)
    return SearchResult.from_evaluation(evaluation, objective_result)


def run_search(
    evaluator: EvaluatorFn,
    objective: ObjectiveFn,
    candidates: Iterable[dict[str, Any]],
    *,
    fixed: dict[str, Any] | None = None,
    reporter: ReporterFn | None = None,
) -> list[SearchResult]:
    """Evaluate, rank, and return all search results best-first."""

    results: list[SearchResult] = []
    for params in candidates:
        result = evaluate_candidate(evaluator, objective, params, fixed=fixed)
        results.append(result)
        if reporter is not None:
            reporter(result)
    results.sort(key=lambda r: (r.valid, r.score), reverse=True)
    return results
