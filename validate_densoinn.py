"""Validate one DenSOINN configuration with internal and external metrics."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import argparse
import csv
from pathlib import Path

from baseframework.datahandler import load_data
from baseframework.tuning.metrics import make_external_metric_bundle, make_internal_metric_bundle
from src.eval.densoinn_eval import evaluate_params


PARAM_CASTERS = {
    "gamma": float,
    "stream_speed": int,
    "epsilon": float,
    "max_nodes": int,
    "alpha": float,
    "intrinsic_dim": int,
}


def main():
    parser = argparse.ArgumentParser(description="Validate one DenSOINN parameter configuration")
    parser.add_argument("--dataset", default="complex9", help="Dataset name")
    parser.add_argument("--raw", action="store_true", help="Disable min-max scaling before streaming")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--points_per_timestep", type=int, default=1)
    parser.add_argument("--label_mode", choices=["nearest_survivor", "summary_lookup"], default="nearest_survivor")
    parser.add_argument("--results_csv", default=None, help="Optional tuning CSV to load params from")
    parser.add_argument("--rank", type=int, default=1, help="1-based row rank when loading from results CSV")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--stream_speed", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--intrinsic_dim", type=int, default=2)
    parser.add_argument("--ts_window", type=int, default=200)
    parser.add_argument("--ts_kn", type=int, default=200)
    parser.add_argument("--ts_c", type=float, default=1.0)
    parser.add_argument("--cmm_k", type=int, default=5)
    args = parser.parse_args()

    params = (
        _load_params_from_csv(args.results_csv, args.rank)
        if args.results_csv is not None
        else {
            "gamma": args.gamma,
            "stream_speed": args.stream_speed,
            "epsilon": args.epsilon,
            "max_nodes": args.max_nodes,
            "alpha": args.alpha,
            "intrinsic_dim": args.intrinsic_dim,
        }
    )

    print(f"Loading data '{args.dataset}'...")
    X, y = load_data(args.dataset, scale=not args.raw)

    record = evaluate_params(
        X,
        y,
        batch_size=args.batch_size,
        points_per_timestep=args.points_per_timestep,
        label_mode=args.label_mode,
        internal_metric_fns=make_internal_metric_bundle(
            ts_window=args.ts_window,
            ts_kn=args.ts_kn,
            ts_c=args.ts_c,
        ),
        external_metric_fns=make_external_metric_bundle(cmm_k=args.cmm_k),
        return_artifacts=False,
        **params,
    )

    print("\nParameters")
    for name, value in params.items():
        print(f"  {name}: {value}")

    print("\nDiagnostics")
    for name, value in sorted(record.diagnostics.items()):
        print(f"  {name}: {value}")

    print("\nInternal Metrics")
    for name, value in sorted(record.internal_metrics.items()):
        print(f"  {name}: {value:.6f}")

    print("\nExternal Metrics")
    for name, value in sorted(record.external_metrics.items()):
        if name == "true_k":
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.6f}")


def _load_params_from_csv(path: str, rank: int) -> dict[str, float | int]:
    rows = list(csv.DictReader(Path(path).open()))
    if rank < 1 or rank > len(rows):
        raise ValueError(f"rank must be between 1 and {len(rows)}")
    row = rows[rank - 1]
    return {
        name: caster(row[f"param_{name}"])
        for name, caster in PARAM_CASTERS.items()
    }


if __name__ == "__main__":
    main()
