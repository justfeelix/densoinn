"""
DenSOINN: Density Based Self Organizing Incremental Neural Network.
Main orchestrator combining the online network training and the offline
density-based clustering of nodes.
"""
import numpy as np

from .online_phase import DenSOINNOnlinePhase
from .offline_phase import DenSOINNOfflinePhase


class DenSOINN:
    """
    Orchestrates both phases:
      - online:  grow a self-organizing network of micro-clusters
      - offline: cluster the network nodes by density on a clustering request

    Parameters
    ----------
    gamma : float
        Edge weight decay speed (online).
    stream_speed : int
        Patterns per time period; denoising frequency (online).
    epsilon : float
        Denoising intensity (online).
    max_nodes : int
        Maximum network capacity (online).
    alpha : float
        Cluster-center sensitivity (offline).
    intrinsic_dim : int
        Estimated intrinsic dimensionality for the density estimate (offline).
    """

    def __init__(self, gamma: float = 0.25, stream_speed: int = 200, epsilon: float = 0.5,
                 max_nodes: int = 1000, alpha: float = 5.0, intrinsic_dim: int = 2):
        self.online_phase = DenSOINNOnlinePhase(
            gamma=gamma, stream_speed=stream_speed, epsilon=epsilon, max_nodes=max_nodes
        )
        self.offline_phase = DenSOINNOfflinePhase(alpha=alpha, intrinsic_dim=intrinsic_dim)

    # for online
    def learn(self, xs, ts) -> list:
        """
        Feed a new batch of patterns into the online phase.
        Leave as is; the actual work happens in DenSOINNOnlinePhase.learn_one().
        """
        return self.online_phase.learn(xs, ts)

    # for offline
    def predict(self, assignments=None):
        """
        Trigger the offline phase to get the current clustering. Two cases:
          1. assignments is None -> cluster the network nodes themselves and
             return predict_summaries() ({node_id: label}).
          2. assignments given   -> map the per-point node assignments (the
             authoritative summary_ids produced by the online phase) to final
             macro-cluster labels via predict_assignment(), returning one label
             per point.
        """
        summaries = self.online_phase.get_summary()

        if assignments is None:
            return self.offline_phase.predict_summaries(summaries)
        else:
            return self.offline_phase.predict_assignment(assignments, summaries)

    def predict_points(self, X):
        """
        Label each raw point by its NEAREST surviving node, then map node ->
        macro-cluster label. This reproduces the paper's evaluation protocol
        (each pattern is labeled by its nearest exemplar in the final model)
        and is robust to nodes deleted during denoising, unlike using the
        stored online assignment ids.

        Returns one cluster label per point; -1 is used both for the offline
        noise label and as a fallback when the network has no nodes to label
        against.
        """
        X = np.asarray(X, dtype=float)
        n_points = X.shape[0]

        # Node centroids + topology, keyed by node id.
        summaries = self.online_phase.get_summary()
        if not summaries:
            # Network never grew a node -> nothing to label against.
            return [-1] * n_points

        # node id -> macro-cluster label (offline density clustering).
        node_labels = self.offline_phase.predict_summaries(summaries)

        metric = self.online_phase.metric
        if metric is None:
            return [-1] * n_points

        # Rebuild the self-adaptive metric A on the *surviving* nodes. The
        # offline phase rebuilds the same diagonal metric from these centroids
        # (Algorithm 1), so this keeps our nearest-node distances consistent
        # with the metric under which the labels were assigned. It also corrects
        # for the final denoising step, which can remove nodes after the last
        # in-stream metric update. Harmless to the stream: it is over by now,
        # and learn_one re-derives A on every pattern anyway.
        metric.update(self.online_phase.network.nodes)

        # One fixed ordering shared by the centroid matrix and the id lookup, so
        # an argmin index maps back to the correct node id.
        node_ids = list(summaries.keys())
        centroids = np.array([summaries[i]["centroid"] for i in node_ids], dtype=float)

        # Variant 1: Nearest-node assignment
        # Loop over points; distance_to_nodes is already vectorised over the
        # nodes, so each step is O(n_nodes) in time and memory. This reuses the
        # exact distance path of the online competition step, keeping labels
        # consistent, and never materialises a large matrix.
        #
        # Variant 2 (we switch to this ONLY if the loop becomes a bottleneck on
        # whole-stream evaluation): vectorise over points AND nodes at once.
        # With the diagonal metric exposed as `metric.a_diag`:
        #     s  = np.sqrt(metric.a_diag)
        #     Xs = X * s
        #     Cs = centroids * s
        #     d2 = (Xs**2).sum(1)[:, None] + (Cs**2).sum(1)[None, :] - 2 * Xs @ Cs.T
        #     nearest = d2.argmin(axis=1)
        # Faster, but materialises an (n_points x n_nodes) distance matrix
        # (~800 MB at 100k points x 1000 nodes), so chunk X into blocks of a few
        # thousand rows if you adopt it.
        labels = []
        for x in X:
            dists = metric.distance_to_nodes(x, centroids)
            nearest_id = node_ids[int(np.argmin(dists))]
            labels.append(node_labels.get(nearest_id, -1))
        return labels
