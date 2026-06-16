"""
Programmatic parameter search for DenSOINN.

Supports two ranking modes:
  - unsupervised: tune on internal, label-free metrics only
  - supervised:   reproduce the old label-aware selection as a benchmark

Ground-truth labels stay out of unsupervised runs unless they are explicitly
requested for validation-only reporting.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import argparse
import os
from functools import partial

from baseframework.datahandler import load_data
from baseframework.tuning.engine import run_search
from baseframework.tuning.metrics import make_external_metric_bundle, make_internal_metric_bundle
from baseframework.tuning.objectives import make_supervised_objective, make_unsupervised_objective
from baseframework.tuning.reporting import (
    format_search_result,
    grid_matrix,
    grid_matrix_marginalized,
    print_matrix,
    print_top,
    save_heatmap,
    write_results_csv,
)
from baseframework.tuning.search_space import grid_candidates, random_candidates, refined_grid_from_topk
from src.eval.densoinn_eval import evaluate_params


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for DenSOINN")
    parser.add_argument("--dataset", default="complex9", help="Dataset name")
    parser.add_argument("--raw", action="store_true", help="Disable min-max scaling before streaming")
    parser.add_argument("--method", choices=["grid", "random", "refined"], default="grid")
    parser.add_argument("--objective", choices=["unsupervised", "supervised"], default="unsupervised")
    parser.add_argument("--n_iter", type=int, default=30, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random search")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--points_per_timestep", type=int, default=1)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--intrinsic_dim", type=int, default=2)
    parser.add_argument("--grid_gamma", type=float, default=0.25, help="Gamma held fixed during 2-D grid search")
    parser.add_argument("--grid_stream_speed", type=int, default=200, help="Stream speed held fixed during 2-D grid search")
    parser.add_argument("--count_penalty_weight", type=float, default=0.5)
    parser.add_argument("--noise_penalty_weight", type=float, default=0.25)
    parser.add_argument("--tiny_cluster_penalty_weight", type=float, default=0.25)
    parser.add_argument("--fragmentation_penalty_weight", type=float, default=0.5)
    parser.add_argument("--summary_penalty_weight", type=float, default=0.0)
    parser.add_argument("--ts_window", type=int, default=200)
    parser.add_argument("--ts_kn", type=int, default=200)
    parser.add_argument("--ts_c", type=float, default=1.0)
    parser.add_argument("--cmm_k", type=int, default=5)
    parser.add_argument(
        "--with_validation_metrics",
        action="store_true",
        help="Also compute external metrics during unsupervised search for reporting only",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Abort the search immediately if one candidate evaluation fails",
    )
    parser.add_argument("--plot_dir", default=None)
    args = parser.parse_args()

    if args.plot_dir is None:
        args.plot_dir = os.path.join("data", "results", "tuning", args.dataset, args.objective)
    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"Loading data '{args.dataset}'...")
    need_labels = args.objective == "supervised" or args.with_validation_metrics
    X, y = load_data(args.dataset, scale=not args.raw, include_labels=need_labels)

    internal_metric_fns = make_internal_metric_bundle(
        ts_window=args.ts_window,
        ts_kn=args.ts_kn,
        ts_c=args.ts_c,
    )
    external_metric_fns = make_external_metric_bundle(cmm_k=args.cmm_k) if need_labels else None
    evaluator = partial(
        evaluate_params,
        X,
        y,
        batch_size=args.batch_size,
        points_per_timestep=args.points_per_timestep,
        internal_metric_fns=internal_metric_fns,
        external_metric_fns=external_metric_fns,
        return_artifacts=False,
    )

    fixed = {
        "max_nodes": args.max_nodes,
        "intrinsic_dim": args.intrinsic_dim,
    }
    objective = _build_objective(args)
    reporter = _build_reporter(args.objective, include_external=need_labels)

    if args.method == "grid":
        param_grid = {
            "alpha": [2.0, 3.5, 5.0, 6.5, 8.0],
            "epsilon": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
        fixed.update({
            "gamma": args.grid_gamma,
            "stream_speed": args.grid_stream_speed,
        })
        candidates = list(grid_candidates(param_grid))
        print(
            f"\nGRID SEARCH on '{args.dataset}' ({args.objective} objective): "
            f"{len(candidates)} configs with gamma={args.grid_gamma} and "
            f"stream_speed={args.grid_stream_speed}\n"
        )
        results = run_search(
            evaluator,
            objective,
            candidates,
            fixed=fixed,
            reporter=reporter,
            fail_fast=args.fail_fast,
        )

        xs, ys, matrix = grid_matrix(results, "alpha", "epsilon", metric="score")
        print_matrix(xs, ys, matrix, "alpha", "epsilon", "score")
        save_heatmap(
            xs,
            ys,
            matrix,
            "alpha",
            "epsilon",
            "score",
            os.path.join(args.plot_dir, "heatmap_grid.png"),
            title=f"DenSOINN {args.objective} score",
        )
    elif args.method == "random":
        print(f"\nRANDOM SEARCH on '{args.dataset}' ({args.objective} objective): {args.n_iter} configs\n")
        candidates = random_candidates(_param_space(), args.n_iter, seed=args.seed)
        results = run_search(
            evaluator,
            objective,
            candidates,
            fixed=fixed,
            reporter=reporter,
            fail_fast=args.fail_fast,
        )
    else:
        print(
            f"\nREFINED step 1/2: RANDOM SEARCH on '{args.dataset}' "
            f"({args.objective} objective, {args.n_iter} configs, seed={args.seed})\n"
        )
        random_results = run_search(
            evaluator,
            objective,
            random_candidates(_param_space(), args.n_iter, seed=args.seed),
            fixed=fixed,
            reporter=reporter,
            fail_fast=args.fail_fast,
        )
        try:
            param_grid = refined_grid_from_topk(random_results, top_k=5, int_params={"stream_speed"})
        except ValueError:
            print("REFINED: fewer than two valid random configurations; nothing to refine.")
            return

        print(f"\nREFINED step 2/2: 4-D GRID on '{args.dataset}' with ranges {param_grid}\n")
        candidates = list(grid_candidates(param_grid))
        results = run_search(
            evaluator,
            objective,
            candidates,
            fixed=fixed,
            reporter=reporter,
            fail_fast=args.fail_fast,
        )

        xs, ys, matrix = grid_matrix_marginalized(results, "alpha", "epsilon", metric="score")
        print_matrix(xs, ys, matrix, "alpha", "epsilon", "best score (max over gamma, stream_speed)")
        save_heatmap(
            xs,
            ys,
            matrix,
            "alpha",
            "epsilon",
            "score",
            os.path.join(args.plot_dir, "heatmap_refined.png"),
            title=f"DenSOINN {args.objective} score",
        )

    csv_path = write_results_csv(results, os.path.join(args.plot_dir, "results.csv"))
    top_metrics = ["internal.ts", "diagnostics.n_clusters"]
    if need_labels:
        top_metrics.insert(1, "external.cmm")
    print_top(results, n=5, extra_metrics=top_metrics)
    print(f"\nSaved ranked results to {csv_path}")


def _param_space():
    return {
        "gamma": lambda rng: round(float(rng.uniform(0.10, 1.00)), 4),
        "stream_speed": lambda rng: int(rng.integers(50, 201)),
        "epsilon": lambda rng: round(float(rng.uniform(0.20, 1.00)), 4),
        "alpha": lambda rng: round(float(rng.uniform(2.0, 10.0)), 4),
    }


def _build_objective(args):
    if args.objective == "supervised":
        return make_supervised_objective(count_penalty_weight=args.count_penalty_weight)
    return make_unsupervised_objective(
        noise_penalty_weight=args.noise_penalty_weight,
        tiny_cluster_penalty_weight=args.tiny_cluster_penalty_weight,
        fragmentation_penalty_weight=args.fragmentation_penalty_weight,
        summary_penalty_weight=args.summary_penalty_weight,
    )


def _build_reporter(objective_name: str, *, include_external: bool):
    extra_metrics = ["internal.ts", "diagnostics.n_clusters", "diagnostics.noise_frac"]
    if objective_name == "supervised" or include_external:
        extra_metrics.append("external.cmm")

    def reporter(result):
        print(format_search_result(result, extra_metrics=extra_metrics))

    return reporter


if __name__ == "__main__":
    main()
