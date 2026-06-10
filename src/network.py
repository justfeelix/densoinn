"""
The DenSOINN topological network G = <V, E>.
Mirrors the role of the AdjacencyGraph from CEDGM, but here edges
are weighted and directed and are built/decayed by competitive Hebbian learning.
"""
from collections import defaultdict, deque
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
        """Create a new Node with centroid w, add it to V and return its id."""
        node_id = self._id_counter
        self._id_counter += 1
        self.nodes[node_id] = Node(node_id, np.array(w, dtype=float))
        return node_id

    def remove_node(self, node_id: int) -> None:
        """
        Remove the node from V and delete every edge incident to it, both
        outgoing (self.edges[node_id]) and incoming (edges from others to it).
        """
        self.nodes.pop(node_id, None)
        self.edges.pop(node_id, None)
        for src in self.edges:
            self.edges[src].pop(node_id, None)

    def num_nodes(self) -> int:
        """Return |V|."""
        return len(self.nodes)

    # ------------------------------------------------------------------ #
    # Competition
    # ------------------------------------------------------------------ #
    def _stacked_centroids(self) -> tuple[list[int], np.ndarray]:
        """Return (ordered node ids, matrix of their centroids) for vectorised ops."""
        ids = list(self.nodes.keys())
        centroids = np.array([self.nodes[i].w for i in ids], dtype=float)
        return ids, centroids

    def find_winners(self, x: np.ndarray, metric) -> tuple[int, int]:
        """
        Find the winner s1 (nearest node) and runner-up s2 (second nearest) for
        input pattern x under the given metric. Returns their node ids.
        Assumes |V| >= 2.
        """
        ids, centroids = self._stacked_centroids()
        dists = metric.distance_to_nodes(x, centroids)
        order = np.argpartition(dists, 1)[:2]
        # argpartition does not order the two smallest among themselves
        if dists[order[0]] <= dists[order[1]]:
            s1, s2 = order[0], order[1]
        else:
            s1, s2 = order[1], order[0]
        return ids[s1], ids[s2]


    # ------------------------------------------------------------------ #
    def activation_threshold(self, node_id: int, metric) -> float:
        """
        Compute the activation threshold t_i of a node.
        - if the node has topological neighbors N_i: t_i = max distance to a
          neighbor (Eq. 3)
        - if N_i is empty: t_i = min distance to any other node in V (Eq. 4)
        """
        w_i = self.nodes[node_id].w
        nbrs = self.neighbors(node_id)
        if nbrs:
            return max(metric.distance(w_i, self.nodes[j].w) for j in nbrs)
        # No neighbors: nearest other node in V.
        others = [j for j in self.nodes if j != node_id]
        return min(metric.distance(w_i, self.nodes[j].w) for j in others)


    def reinforce_edge(self, i: int, j: int) -> None:
        """
        If edge (i, j) does not exist, create it with weight 1; otherwise
        increase its weight by 1 (Hebbian reinforcement of s1, s2).
        """
        self.edges[i][j] = self.edges[i].get(j, 0.0) + 1.0

    def decay_edges(self, i: int) -> None:
        """
        Multiply the weight of every edge starting from i by 2^(-gamma).
        Called whenever the position of node i changes (Eq. 8).
        """
        factor = 2.0 ** (-self.gamma)
        for j in self.edges[i]:
            self.edges[i][j] *= factor

    def prune_weak_edges(self, i: int) -> None:
        """
        Remove every edge (i, j) whose weight drops below 10% of the summed
        weights of all edges starting from i.
        """
        out = self.edges[i]
        if not out:
            return
        threshold = 0.10 * sum(out.values())
        for j in [j for j, w in out.items() if w < threshold]:
            del out[j]

    def neighbors(self, node_id: int) -> set:
        """
        Return the set N_i of topological neighbors of node_id, considering
        edges in both directions (weakly connected view).
        """
        nbrs = set(self.edges.get(node_id, {}).keys())
        for src, targets in self.edges.items():
            if node_id in targets:
                nbrs.add(src)
        nbrs.discard(node_id)
        return nbrs


    # ------------------------------------------------------------------ #
    def denoise(self, epsilon: float) -> list[int]:
        """
        Remove noise nodes and their edges (Eq. 9). A node i is noise if
            m_i < (10^(1 - |N_i|) * epsilon / |V|) * sum_j m_j.
        Fewer neighbors -> higher threshold -> more easily judged noise.

        Snapshot-then-remove: thresholds use the network state *before* any
        deletion, so removing nodes cannot corrupt |V| or |N_i| mid-pass.
        Returns the list of removed node ids.
        """
        n = len(self.nodes)
        if n == 0:
            return []
        total_weight = sum(node.m for node in self.nodes.values())

        noise = []
        for node_id, node in self.nodes.items():
            deg = len(self.neighbors(node_id))
            threshold = (10.0 ** (1 - deg)) * epsilon / n * total_weight
            if node.m < threshold:
                noise.append(node_id)

        for node_id in noise:
            self.remove_node(node_id)
        return noise

    # ------------------------------------------------------------------ #
    def connected_components(self) -> list[list[int]]:
        """
        Split the (weakly connected) graph into connected components.
        Returns a list of components, each a list of node ids. Used by the
        offline phase before finding density peaks per component.
        """
        # Build an undirected adjacency view once (cheaper than calling
        # neighbors() per node, which scans all edges each time).
        undirected: dict[int, set] = {i: set() for i in self.nodes}
        for src, targets in self.edges.items():
            if src not in undirected:
                continue
            for tgt in targets:
                if tgt in undirected:
                    undirected[src].add(tgt)
                    undirected[tgt].add(src)

        visited: set[int] = set()
        components: list[list[int]] = []
        for start in self.nodes:
            if start in visited:
                continue
            comp = []
            queue = deque([start])
            visited.add(start)
            while queue:
                node_id = queue.popleft()
                comp.append(node_id)
                for nbr in undirected[node_id]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(comp)
        return components
