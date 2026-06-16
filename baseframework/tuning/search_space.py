"""Candidate-generation helpers shared by tuning entrypoints."""

from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np

from .types import SearchResult


ConstraintFn = Callable[[dict[str, Any]], bool]
SamplerFn = Callable[[np.random.Generator], Any]


def grid_candidates(
    param_grid: dict[str, list[Any]],
    constraint: ConstraintFn | None = None,
):
    """Yield every combination from a discrete parameter grid."""

    names = list(param_grid)
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(names, combo))
        if constraint is None or constraint(params):
            yield params


def random_candidates(
    param_space: dict[str, SamplerFn],
    n_iter: int,
    *,
    seed: int | None = None,
    constraint: ConstraintFn | None = None,
    max_attempt_factor: int = 100,
):
    """Yield up to ``n_iter`` constrained random parameter sets."""

    rng = np.random.default_rng(seed)
    produced = 0
    for _ in range(n_iter * max_attempt_factor):
        if produced >= n_iter:
            break
        params = {name: sample(rng) for name, sample in param_space.items()}
        if constraint is None or constraint(params):
            yield params
            produced += 1


def refined_grid_from_topk(
    results: list[SearchResult],
    *,
    top_k: int = 5,
    n_points: int = 3,
    int_params: set[str] | None = None,
) -> dict[str, list[Any]]:
    """Build a small grid spanning the observed min..max of the top-k results."""

    int_params = int_params or set()
    top = [r for r in results if r.valid][:top_k]
    if len(top) < 2:
        raise ValueError("need at least two valid top-k results to refine")

    grid: dict[str, list[Any]] = {}
    for name in top[0].params:
        values = sorted({r.params[name] for r in top})
        lo, hi = min(values), max(values)
        if lo == hi:
            grid[name] = [int(round(lo)) if name in int_params else float(lo)]
            continue
        pts = np.linspace(lo, hi, n_points)
        if name in int_params:
            grid[name] = sorted({int(round(v)) for v in pts})
        else:
            grid[name] = [round(float(v), 4) for v in pts]
    return grid
