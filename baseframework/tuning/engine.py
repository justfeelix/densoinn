"""Objective-agnostic tuning engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import traceback
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
    fail_fast: bool = False,
) -> SearchResult:
    """Run one evaluation and derive its ranking score."""

    merged = {**(fixed or {}), **params}
    try:
        evaluation = evaluator(**merged)
        objective_result = objective(evaluation)
        return SearchResult.from_evaluation(evaluation, objective_result)
    except Exception as exc:
        if fail_fast:
            raise
        return SearchResult.from_failure(
            params=merged,
            reason=f"{type(exc).__name__}: {exc}",
            error=traceback.format_exc(),
        )


def run_search(
    evaluator: EvaluatorFn,
    objective: ObjectiveFn,
    candidates: Iterable[dict[str, Any]],
    *,
    fixed: dict[str, Any] | None = None,
    reporter: ReporterFn | None = None,
    fail_fast: bool = False,
) -> list[SearchResult]:
    """Evaluate, rank, and return all search results best-first."""

    results: list[SearchResult] = []
    for params in candidates:
        result = evaluate_candidate(evaluator, objective, params, fixed=fixed, fail_fast=fail_fast)
        results.append(result)
        if reporter is not None:
            reporter(result)
    results.sort(key=lambda r: (r.valid, r.score), reverse=True)
    return results
