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
        TODO:
        Label each raw point by its NEAREST surviving node, then map node ->
        macro-cluster label. This reproduces the paper's evaluation protocol
        (each pattern is labeled by its nearest exemplar in the final model)
        and is robust to nodes deleted during denoising, unlike using the
        stored online assignment ids.

        Steps to implement:
          - summaries  = self.online_phase.get_summary()
          - node_labels = self.offline_phase.predict_summaries(summaries)   # {node_id: label}
          - for each x in X: find the nearest node centroid (use the online
            phase's distance metric for consistency) and take its label,
            falling back to -1 if no node exists.
          - return one label per point.

        Switch run_densoinn.py's final evaluation to this method once the
        algorithm is implemented and you want to match the paper's numbers.
        """
        raise NotImplementedError("predict_points: implement nearest-node labeling for paper reproduction")
