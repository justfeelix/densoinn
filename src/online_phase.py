"""
Online network training of DenSOINN.
"""
import numpy as np
from baseframework.stream_clustering import OnlinePhase_StreamClusterer

from .network import DenSOINNNetwork
from .distance_metric import SelfAdaptiveDistanceMetric


class DenSOINNOnlinePhase(OnlinePhase_StreamClusterer):
    """
    Grows the DenSOINN network one pattern at a time.

    Parameters
    ----------
    gamma : float
        Edge weight decay speed. Recommended range [0.1, 1].
    stream_speed : int
        Number of patterns per time period; denoising runs every stream_speed
        inputs. Recommended range [50, 200].
    epsilon : float
        Denoising intensity. Recommended range [0.2, 1].
    max_nodes : int
        Maximum number of nodes; above this the network stops creating new ones
        and patterns always join their nearest node.
    """

    def __init__(self, gamma: float, stream_speed: int, epsilon: float, max_nodes: int):
        super().__init__()
        self.gamma = gamma
        self.stream_speed = stream_speed
        self.epsilon = epsilon
        self.max_nodes = max_nodes

        self.network = DenSOINNNetwork(gamma)
        # NOTE: dimensionality is unknown until the first point arrives, so the
        # metric is created lazily inside learn_one (or here once n_dims is known).
        self.metric: SelfAdaptiveDistanceMetric | None = None

        self.n_seen = 0        # number of patterns processed (for denoising trigger)
        self.current_time = 0

    def learn_one(self, x: np.ndarray, t: int) -> tuple[int, int, str]:
        """
        TODO: process a single pattern.
          1. (lazily) create the metric on the first point
          2. if |V| < 2: create a new node and return early
          3. adjust the metric A from the current nodes
          4. find winner s1 and runner-up s2
          5. compute activation thresholds t_s1, t_s2
          6. if x cannot activate s1 or s2 (and |V| < max_nodes):
                create a new node r and the edge (r, s1)
             else:
                absorb x into s1, decay edges of s1,
                prune weak edges, reinforce edge (s1, s2)
          7. every stream_speed patterns: call _denoise()

        Return (summary_id, summary_id, event_log) following the framework
        signature; summary_id is the id of the node x was assigned to.
        """
        self.current_time = t
        self.n_seen += 1
        event_log: list[str] = []
        summary_id = -1


        return (summary_id, summary_id, " | ".join(event_log))

    def _denoise(self, event_log: list) -> None:
        """
        TODO: trigger the network's denoising step and log which nodes
        were removed. Called every stream_speed patterns.
        """
        pass

    def get_summary(self) -> dict:
        """
        TODO: return the current network state as a summary dict for the
        offline phase, mapping node_id -> summary representation.

        Unlike DenStream, the DenSOINN offline phase needs the graph
        topology (connected components), so each summary must also expose the
        node's neighbors / edges (not only centroid + weight). One option is to
        return the Node objects together with the network, or to embed the
        adjacency in the representation.
        """
        pass

    # this can be left as it is: it just calls learn_one on multiple instances
    def learn(self, xs, ts) -> list[tuple[int, int, str]]:
        return super().learn(xs, ts)
