"""
Main entry point for running and experimenting with DenSOINN.

Drives the full online/offline pipeline on a dataset and evaluates the result
with CMM + Temporal Silhouette. This is the orchestration scaffold; the actual
algorithm lives in the src/ classes.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'baseframework/')

import argparse
import numpy as np
from src.densoinn import DenSOINN
from baseframework.datahandler import load_data
try:
    from baseframework.metrics.cmm import get_cmm
    from baseframework.metrics.TSindex import tempsil
except ImportError:
    print("Warning: Could not import metrics!")


def _evaluate(X_acc, y_acc, labels_acc, ts_acc):
    """Compute and print CMM and Temporal Silhouette on accumulated data."""
    labels_arr = np.array(labels_acc)
    unique_clusters = np.unique(labels_arr[labels_arr != -1])
    if len(unique_clusters) < 2:
        print("   [Eval] Skipped: fewer than 2 non-noise clusters found.")
        return

    w = np.ones(len(X_acc))
    cmm = get_cmm(np.array(X_acc), np.array(y_acc), labels_arr, w, k=5)
    _, _, ts = tempsil(np.array(ts_acc), np.array(X_acc), labels_arr)
    print(f"   [Eval] CMM={cmm:.4f}  |  TS={ts:.4f}  |  "
          f"n={len(labels_arr)}  k={len(unique_clusters)}")


def main():
    parser = argparse.ArgumentParser(description="Run DenSOINN Clustering Simulation")

    # Dataset / batching
    parser.add_argument('--dataset', type=str, default='complex9',
                        help='Dataset name: complex9, rbf3, har, stream_densired')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Number of instances per batch')
    parser.add_argument('--raw', action='store_true',
                        help='Disable min-max scaling and feed raw data. DenSOINN is '
                             'designed to handle this via its self-adaptive distance metric.')

    # DenSOINN online parameters (paper defaults)
    parser.add_argument('--gamma', type=float, default=0.25,
                        help='Edge weight decay speed (recommended [0.1, 1])')
    parser.add_argument('--stream_speed', type=int, default=200,
                        help='Patterns per time period; denoising runs every stream_speed inputs '
                             '(recommended [50, 200])')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Denoising intensity (recommended [0.2, 1])')
    parser.add_argument('--max_nodes', type=int, default=1000,
                        help='Maximum number of network nodes')

    # DenSOINN offline parameters
    parser.add_argument('--alpha', type=float, default=5.0,
                        help='Cluster-center sensitivity (recommended [2, 10], typically 5)')
    parser.add_argument('--intrinsic_dim', type=int, default=2,
                        help='Estimated intrinsic dimensionality for the density estimate')

    # Offline phase / evaluation
    parser.add_argument('--offline_interval', type=int, default=1,
                        help='Run offline clustering + evaluation every N batches (0 = only at end of stream)')

    args = parser.parse_args()

    print(f"Loading data '{args.dataset}'...")
    X, y = load_data(args.dataset, scale=not args.raw)

    densoinn = DenSOINN(
        gamma=args.gamma,
        stream_speed=args.stream_speed,
        epsilon=args.epsilon,
        max_nodes=args.max_nodes,
        alpha=args.alpha,
        intrinsic_dim=args.intrinsic_dim,
    )

    num_batches = len(X) // args.batch_size

    print(f"\n=======================================================")
    print(f"STARTING STREAM PROCESSING: {num_batches} batches of size {args.batch_size}")
    print(f"PARAMETERS: gamma={args.gamma}  stream_speed={args.stream_speed}  "
          f"epsilon={args.epsilon}  max_nodes={args.max_nodes}")
    print(f"            alpha={args.alpha}  intrinsic_dim={args.intrinsic_dim}  "
          f"scaling={'off (raw)' if args.raw else 'on'}")
    print(f"=======================================================\n")

    # Accumulators — store every point's authoritative summary_id (node id) from
    # the online phase, so the final offline pass needs no nearest-node approximation.
    all_X, all_y, all_assignments, all_ts = [], [], [], []

    for i in range(num_batches + 1):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(X))
        batch = X[start_idx:end_idx]

        if len(batch) == 0:
            break

        timestamps = np.arange(start_idx, end_idx)

        # ONLINE PHASE — each result is (summary_id, cluster_id, event_log)
        results = densoinn.learn(batch, timestamps)
        batch_assignments = [r[0] for r in results]

        all_X.extend(batch)
        all_y.extend(y[start_idx:end_idx])
        all_assignments.extend(batch_assignments)
        all_ts.extend(timestamps)

        print(f"   Batch {i:>4d} | points so far: {len(all_X):>6d}")

        # OFFLINE PHASE — only on demand
        run_offline = (
            args.offline_interval > 0 and (i + 1) % args.offline_interval == 0
        )
        if run_offline:
            labels = densoinn.predict_points(np.array(batch))
            num_clusters = len(np.unique(np.array(labels)[np.array(labels) != -1]))
            print(f"          [Offline] clusters this batch: {num_clusters}")
            _evaluate(list(batch), list(y[start_idx:end_idx]), labels, list(timestamps))

    print(f"\n--- Final offline clustering + evaluation ---")
    final_labels = densoinn.predict_points(np.array(all_X))
    final_arr = np.array(final_labels)
    unique_clusters = np.unique(final_arr[final_arr != -1])
    print(f"    Total points: {len(all_X)}  |  clusters: {len(unique_clusters)}")
    _evaluate(all_X, all_y, final_labels, all_ts)


if __name__ == '__main__':
    main()
