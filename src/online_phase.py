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
        Process a single pattern (Algorithm 2, one iteration).

        Returns (summary_id, cluster_id, event_log) following the framework
        signature. DenSOINN has no online cluster ids (clustering is an offline
        step), so summary_id is the id of the node x was assigned to and is
        returned for both fields. The offline phase later maps these node ids to
        macro-cluster labels.
        """
        x = np.asarray(x, dtype=float)
        self.current_time = t
        self.n_seen += 1
        event_log: list[str] = []

        # 1. Lazily create the metric once the dimensionality is known.
        if self.metric is None:
            self.metric = SelfAdaptiveDistanceMetric(n_dims=x.shape[0])

        # 2. Bootstrap: with fewer than two nodes, x simply forms a new node.
        #    (Algorithm 2 "goto step 2" -> skip metric/competition/denoising.)
        if self.network.num_nodes() < 2:
            summary_id = self.network.create_node(x)
            event_log.append(f"bootstrap node {summary_id}")
            return (summary_id, summary_id, " | ".join(event_log))

        # 3. Adjust the self-adaptive metric A from the current node positions.
        self.metric.update(self.network.nodes)

        # 4. Competition: winner s1 and runner-up s2.
        s1, s2 = self.network.find_winners(x, self.metric)

        # 5. Activation thresholds (Eqs. 3/4).
        t_s1 = self.network.activation_threshold(s1, self.metric)
        t_s2 = self.network.activation_threshold(s2, self.metric)

        d_s1 = self.metric.distance(x, self.network.nodes[s1].w)
        d_s2 = self.metric.distance(x, self.network.nodes[s2].w)

        # 6. Insert-or-update decision.
        if self.network.num_nodes() < self.max_nodes and (d_s1 > t_s1 or d_s2 > t_s2):
            # x activates neither winner -> it starts its own micro-cluster.
            r = self.network.create_node(x)
            self.network.reinforce_edge(r, s1)   # directed edge (r -> s1), weight 1
            summary_id = r
            event_log.append(f"new node {r} (edge ->{s1})")
        else:
            # x is similar enough -> absorb into the winner s1.
            self.network.nodes[s1].absorb(x)
            # Edge bookkeeping. Order: decay -> prune -> reinforce (Algorithm 2,
            # lines 14-20). This protects the fresh s1<->s2 Hebbian edge from
            # being pruned by a threshold computed on stale weights. The paper
            # prose (Eq. 8 paragraph) gives the opposite order; see
            # DenSOINN-reference.md S11.1 for this documented design decision.
            self.network.decay_edges(s1)
            self.network.prune_weak_edges(s1)
            self.network.reinforce_edge(s1, s2)
            summary_id = s1
            event_log.append(f"absorb into {s1} (edge {s1}<->{s2})")

        # 7. Periodic denoising every stream_speed patterns.
        if self.n_seen % self.stream_speed == 0:
            self._denoise(event_log)

        return (summary_id, summary_id, " | ".join(event_log))

    def _denoise(self, event_log: list) -> None:
        """
        Trigger the network's denoising step (Eq. 9) and log removed nodes.
        Called every stream_speed patterns.
        """
        removed = self.network.denoise(self.epsilon)
        if removed:
            event_log.append(f"denoised {len(removed)} nodes: {removed}")

    def get_summary(self) -> dict:
        """
        Return the current network state as a summary dict for the offline
        phase, mapping node_id -> {centroid, weight, neighbors}.

        Unlike DenStream, the DenSOINN offline phase needs the graph topology
        (connected components, density-peak neighbours), so each summary exposes
        the node's neighbour ids in addition to its centroid and weight. The
        offline phase can rebuild the same self-adaptive diagonal metric from
        these centroids (Algorithm 1) so node-to-node distances stay consistent
        with the online phase.
        """
        return {
            node_id: {
                "centroid": node.w.copy(),
                "weight": node.m,
                "neighbors": self.network.neighbors(node_id),
            }
            for node_id, node in self.network.nodes.items()
        }

    # this can be left as it is: it just calls learn_one on multiple instances
    def learn(self, xs, ts) -> list[tuple[int, int, str]]:
        return super().learn(xs, ts)
