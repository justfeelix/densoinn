"""
The DenSOINN topological network G = <V, E>.
Mirrors the role of the AdjacencyGraph from CEDGM, but here edges
are weighted and directed and are built/decayed by competitive Hebbian learning.
"""
from collections import defaultdict
import numpy as np

from .node import Node


class DenSOINNNetwork:
    """
    Maintains the set of neural nodes V and the set of weighted directed
    edges E learned online.

    Edges are stored as an adjacency map:
        self.edges = { i: {j: m_ij, k: m_ik, ...}, ... }
    where m_ij is the weight of the directed edge (i, j). For neighbor queries
    and connected components the graph is treated as weakly connected
    (i.e. direction is ignored).
    """

    def __init__(self, gamma: float):
        self.nodes: dict[int, Node] = {}            # node_id -> Node
        self.edges: dict[int, dict[int, float]] = defaultdict(dict)
        self.gamma = gamma                          # edge decay speed (Eq. 8)
        self._id_counter = 0                        # generator for unique node ids

    # ------------------------------------------------------------------ #
    # Node management
    # ------------------------------------------------------------------ #
    def create_node(self, w: np.ndarray) -> int:
        """
        TODO: create a new Node with centroid w, add it to V and return its id.
        Use self._id_counter to hand out unique ids.
        """
        pass

    def remove_node(self, node_id: int) -> None:
        """
        TODO: remove the node from V and delete every edge incident to it
        (both outgoing self.edges[node_id] and incoming edges from others).
        """
        pass

    def num_nodes(self) -> int:
        """Return |V|."""
        return len(self.nodes)

    # ------------------------------------------------------------------ #
    # Competition
    # ------------------------------------------------------------------ #
    def find_winners(self, x: np.ndarray, metric) -> tuple[int, int]:
        """
        TODO: find the winner s1 (nearest node) and the runner-up
        s2 (second nearest node) for input pattern x under the given metric.
        Return their node ids. Use metric.distance_to_nodes for efficiency.
        """
        pass

    # ------------------------------------------------------------------ #
    # Activation threshold
    # ------------------------------------------------------------------ #
    def activation_threshold(self, node_id: int, metric) -> float:
        """
        TODO: compute the activation threshold t_i of a node.
        - if the node has topological neighbors N_i: t_i = max distance to a
          neighbor (Eq. 3)
        - if N_i is empty: t_i = min distance to any other node in V (Eq. 4)
        """
        pass

    # ------------------------------------------------------------------ #
    # Edge management
    # ------------------------------------------------------------------ #
    def reinforce_edge(self, i: int, j: int) -> None:
        """
        TODO: if edge (i, j) does not exist, create it with weight 1;
        otherwise increase its weight by 1. (Hebbian reinforcement of s1, s2.)
        """
        pass

    def decay_edges(self, i: int) -> None:
        """
        TODO: multiply the weight of every edge starting from i by
        2^(-gamma). Called whenever the position of node i changes.
        """
        pass

    def prune_weak_edges(self, i: int) -> None:
        """
        TODO: remove every edge (i, j) whose weight drops below 10% of the
        summed weights of all edges starting from i.
        """
        pass

    def neighbors(self, node_id: int) -> set:
        """
        TODO: return the set N_i of topological neighbors of node_id,
        considering edges in both directions (weakly connected view).
        """
        pass

    # ------------------------------------------------------------------ #
    # Denoising
    # ------------------------------------------------------------------ #
    def denoise(self, epsilon: float) -> list[int]:
        """
        TODO: remove noise nodes and their edges.
        A node i is noise if  m_i < (10^(1 - |N_i|) / epsilon) * (1/|V|) * sum_j m_j.
        Return the list of removed node ids (useful for the event log).
        """
        pass

    # ------------------------------------------------------------------ #
    # Offline support
    # ------------------------------------------------------------------ #
    def connected_components(self) -> list[list[int]]:
        """
        TODO: split the (weakly connected) graph into connected components.
        Return a list of components, each a list of node ids. Used by the
        offline phase before finding density peaks per component.
        Keep this lightweight (e.g. BFS/union-find), it may be called often.
        """
        pass
