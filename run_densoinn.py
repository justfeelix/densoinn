"""
Main entry point for running and experimenting with DenSOINN.

Drives the full online/offline pipeline on a dataset and evaluates the result
using 10 randomized runs, reporting Mean ± Std Dev for CMM, TS, NMI, and ARI.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'baseframework/')

import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from src.densoinn import DenSOINN
from baseframework.datahandler import load_data

try:
    from baseframework.metrics.cmm import get_cmm
    from baseframework.metrics.TSindex import tempsil
except ImportError:
    print("Warning: Could not import CMM or TS metrics!")


def _evaluate(X_acc, y_acc, labels_arr, ts_acc):
    """Compute metrics for a single run."""
    unique_clusters = np.unique(labels_arr[labels_arr != -1])
    
    # Scikit-Learn Metrics (Matches the Paper)
    nmi = normalized_mutual_info_score(y_acc, labels_arr)
    ari = adjusted_rand_score(y_acc, labels_arr)

    # Baseframework Metrics
    cmm, ts = 0.0, 0.0
    if len(unique_clusters) >= 2:
        w = np.ones(len(X_acc))
        try:
            cmm = get_cmm(np.array(X_acc), np.array(y_acc), labels_arr, w, k=5)
            _, _, ts = tempsil(np.array(ts_acc), np.array(X_acc), labels_arr)
        except Exception as e:
            print(f"      [Metric Warning] Framework metric failed: {e}")

    return cmm, ts, nmi, ari, len(unique_clusters)


def main():
    parser = argparse.ArgumentParser(description="Run DenSOINN Clustering Simulation")

    # Dataset / batching / runs
    parser.add_argument('--dataset', type=str, default='complex9',
                        help='Dataset name: complex9, rbf3, har, spiral, stream_densired')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Number of instances per batch')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of randomized runs to average over (default: 10)')
    parser.add_argument('--raw', action='store_true',
                        help='Disable min-max scaling and feed raw data. Tests the self-adaptive metric.')

    # DenSOINN online parameters
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--stream_speed', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--max_nodes', type=int, default=1000)

    # DenSOINN offline parameters
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--intrinsic_dim', type=int, default=2)

    args = parser.parse_args()

    print(f"Loading data '{args.dataset}'...")
    X_original, y_original = load_data(args.dataset, scale=not args.raw)

    print(f"\n=======================================================")
    print(f"DATASET: {args.dataset} | Patterns: {len(X_original)} | Features: {X_original.shape[1]}")
    print(f"EVALUATION: {args.runs} Randomized Runs")
    print(f"PARAMETERS: gamma={args.gamma} | stream_speed={args.stream_speed} | epsilon={args.epsilon}")
    print(f"            alpha={args.alpha} | intrinsic_dim={args.intrinsic_dim} | scaling={'off (raw)' if args.raw else 'on'}")
    print(f"=======================================================\n")

    # Store results across all runs
    all_cmm, all_ts, all_nmi, all_ari, all_k = [], [], [], [], []

    for run in range(args.runs):
        print(f"--- Starting Run {run + 1}/{args.runs} ---")
        
        # 1. Randomize the input sequence for this run
        indices = np.random.permutation(len(X_original))
        X = X_original[indices]
        y = y_original[indices]

        # 2. Initialize a fresh DenSOINN instance
        densoinn = DenSOINN(
            gamma=args.gamma,
            stream_speed=args.stream_speed,
            epsilon=args.epsilon,
            max_nodes=args.max_nodes,
            alpha=args.alpha,
            intrinsic_dim=args.intrinsic_dim,
        )

        # 3. Online Phase (Streaming)
        num_batches = len(X) // args.batch_size
        timestamps = np.arange(len(X))
        
        for i in range(num_batches + 1):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(X))
            batch = X[start_idx:end_idx]
            batch_ts = timestamps[start_idx:end_idx]

            if len(batch) == 0:
                break
            
            # Feed batch to online phase
            densoinn.learn(batch, batch_ts)

        # 4. Final Offline Phase & Evaluation (Paper Protocol)
        # We use predict_points to assign each point to the nearest surviving node
        final_labels = densoinn.predict_points(X)
        
        cmm, ts, nmi, ari, k = _evaluate(X, y, final_labels, timestamps)
        
        print(f"    Result: k={k} | NMI={nmi:.4f} | ARI={ari:.4f} | CMM={cmm:.4f} | TS={ts:.4f}\n")
        
        all_cmm.append(cmm)
        all_ts.append(ts)
        all_nmi.append(nmi)
        all_ari.append(ari)
        all_k.append(k)

    # Print Final Averaged Results for the Table
    print(f"=======================================================")
    print(f"FINAL AVERAGED RESULTS OVER {args.runs} RUNS ({args.dataset})")
    print(f"=======================================================")
    print(f"Clusters (k): {np.mean(all_k):.1f} ± {np.std(all_k):.1f}")
    print(f"NMI:          {np.mean(all_nmi):.4f} ± {np.std(all_nmi):.4f}")
    print(f"ARI:          {np.mean(all_ari):.4f} ± {np.std(all_ari):.4f}")
    print(f"CMM:          {np.mean(all_cmm):.4f} ± {np.std(all_cmm):.4f}")
    print(f"TS:           {np.mean(all_ts):.4f} ± {np.std(all_ts):.4f}")
    print(f"=======================================================")

if __name__ == '__main__':
    main()