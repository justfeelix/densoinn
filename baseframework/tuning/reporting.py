"""Shared reporting helpers for tuning runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .types import SearchResult


def result_value(result: SearchResult, metric: str) -> float:
    """Resolve a dotted metric path from a search result."""

    if metric == "score":
        return result.score
    if metric.startswith("internal."):
        return float(result.internal_metrics.get(metric.split(".", 1)[1], np.nan))
    if metric.startswith("external."):
        return float(result.external_metrics.get(metric.split(".", 1)[1], np.nan))
    if metric.startswith("diagnostics."):
        value = result.diagnostics.get(metric.split(".", 1)[1], np.nan)
        return float(value) if value is not None else np.nan
    if metric.startswith("objective."):
        return float(result.objective_details.get(metric.split(".", 1)[1], np.nan))
    raise KeyError(f"unsupported metric path: {metric}")


def format_search_result(
    result: SearchResult,
    *,
    extra_metrics: list[str] | None = None,
) -> str:
    """Create a compact, human-readable one-line summary."""

    params = "  ".join(f"{name}={value}" for name, value in sorted(result.params.items()))
    extras = []
    for metric in extra_metrics or []:
        value = result_value(result, metric)
        rendered = "n/a" if np.isnan(value) else f"{value:.4f}"
        extras.append(f"{metric}={rendered}")
    extras_text = f"  ({', '.join(extras)})" if extras else ""
    validity = "ok" if result.valid else result.reason
    return f"  {params}  ->  score={result.score:>9.4f}  [{validity}]{extras_text}"


def print_top(
    results: list[SearchResult],
    *,
    n: int = 5,
    extra_metrics: list[str] | None = None,
) -> None:
    """Print the top-n ranked search results."""

    print(f"\n=== Top {n} configurations ===")
    for rank, result in enumerate(results[:n], 1):
        print(f"{rank:>2d}.{format_search_result(result, extra_metrics=extra_metrics)}")
    if results:
        print(f"\nBest score: {results[0].score:.4f}  with {results[0].params}")


def grid_matrix(
    results: list[SearchResult],
    x_param: str,
    y_param: str,
    *,
    metric: str = "score",
):
    """Pivot 2-D grid results into ``(x_values, y_values, matrix)``."""

    xs = sorted({r.params[x_param] for r in results})
    ys = sorted({r.params[y_param] for r in results})
    lut = {(r.params[x_param], r.params[y_param]): r for r in results}
    matrix = np.full((len(ys), len(xs)), np.nan)
    for row, y_val in enumerate(ys):
        for col, x_val in enumerate(xs):
            result = lut.get((x_val, y_val))
            if result is not None:
                matrix[row, col] = result_value(result, metric)
    return xs, ys, matrix


def grid_matrix_marginalized(
    results: list[SearchResult],
    x_param: str,
    y_param: str,
    *,
    metric: str = "score",
    agg=max,
):
    """Project an N-D search onto two axes by aggregating all other dimensions."""

    xs = sorted({r.params[x_param] for r in results})
    ys = sorted({r.params[y_param] for r in results})
    buckets: dict[tuple[Any, Any], list[float]] = {}
    for result in results:
        key = (result.params[x_param], result.params[y_param])
        buckets.setdefault(key, []).append(result_value(result, metric))
    matrix = np.full((len(ys), len(xs)), np.nan)
    for row, y_val in enumerate(ys):
        for col, x_val in enumerate(xs):
            values = buckets.get((x_val, y_val))
            if values:
                matrix[row, col] = agg(values)
    return xs, ys, matrix


def print_matrix(
    xs: list[Any],
    ys: list[Any],
    matrix: np.ndarray,
    x_param: str,
    y_param: str,
    metric: str,
) -> None:
    """Render a text heatmap that does not depend on plotting libraries."""

    print(f"\n{metric} over {x_param} (cols) x {y_param} (rows):")
    print("        " + "".join(f"{x:>10}" for x in xs))
    for row, y_val in enumerate(ys):
        rendered = "".join(
            "       .  " if np.isnan(value) else f"{value:>10.3f}"
            for value in matrix[row]
        )
        print(f"{y_val:>7} {rendered}")


def save_heatmap(
    xs: list[Any],
    ys: list[Any],
    matrix: np.ndarray,
    x_param: str,
    y_param: str,
    metric: str,
    path: str | Path,
    *,
    title: str,
) -> Path | None:
    """Save a PNG heatmap when matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   [plot skipped] matplotlib not installed")
        return None

    path = Path(path)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_yticks(range(len(ys)), [str(y) for y in ys])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=metric)
    for row in range(len(ys)):
        for col in range(len(xs)):
            value = matrix[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"   heatmap saved to {path}")
    return path


def write_results_csv(results: list[SearchResult], path: str | Path) -> Path:
    """Write ranked results to CSV using flattened metric columns."""

    path = Path(path)
    rows = [flatten_result(result) for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def flatten_result(result: SearchResult) -> dict[str, Any]:
    """Flatten nested search-result dictionaries for tabular storage."""

    row: dict[str, Any] = {
        "score": result.score,
        "valid": result.valid,
        "reason": result.reason,
        "objective_name": result.objective_name,
    }
    for prefix, values in (
        ("param", result.params),
        ("internal", result.internal_metrics),
        ("external", result.external_metrics),
        ("diagnostics", result.diagnostics),
        ("objective", result.objective_details),
    ):
        for name, value in values.items():
            row[f"{prefix}_{name}"] = value
    return row
