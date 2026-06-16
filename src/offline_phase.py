"""
Offline clustering of DenSOINN nodes by density.
"""
import numpy as np
from baseframework.offline_clustering.density_peak_clustering import DensityPeakClustering
from baseframework.stream_clustering import OfflinePhase_Clusterer


class DenSOINNOfflinePhase(OfflinePhase_Clusterer):
    """
    Density-based offline clustering of the learned network nodes.

    Combines "density connectivity" (weakly connected components of the network)
    with "density peaks": the graph is cut into connected components, density
    peaks within each component become cluster centers, and the remaining nodes
    are grouped to the nearest node of higher density.

    Parameters
    ----------
    alpha : float
        Cluster-center sensitivity. A node is a center if its peak
        score h_i >= alpha * h_mean. Reasonable range [2, 10], typically ~5.
    intrinsic_dim : int
        Estimated intrinsic dimensionality n' used in the density estimate.
        A small value such as 2 usually works.

    """

    def __init__(self, alpha: float = 5.0, intrinsic_dim: int = 2):
        super().__init__()
        self.alpha = alpha
        self.intrinsic_dim = intrinsic_dim

    def predict_summaries(self, summaries: dict) -> dict:
        """
        Clusters the network nodes and returns {node_id: macro_cluster_label}
        """
        # input:  dict of node summaries (centroid, weight, neighbors)
        # output: dict mapping node_id -> final cluster label
        labels, _ = self.predict_summaries_with_centers(summaries)
        return labels

    def predict_summaries_with_centers(self, summaries: dict) -> tuple[dict, list]:
        """
        Same clustering as predict_summaries(), but also returns the cluster
        centers (the density peaks chosen as macro-cluster seeds).

        Returns
        -------
        (labels, centers)
            labels  : {node_id: macro_cluster_label}
            centers : list of node_ids that were chosen as cluster centers,
                      one per macro cluster. These are the highlighted nodes in
                      step (c) "Finding cluster centers" of the paper; the
                      remaining nodes are then labelled by their nearest
                      higher-density center, giving step (d).
        """
        if not summaries:
            return {}, []

        # 1. Compute densities
        densities = self._node_density(summaries)

        # 2. Split into connected components
        components = self._connected_components(summaries)

        # 3. Find peaks per component and offset global labels
        final_labels = {}
        centers: list = []
        global_label_offset = 0

        for comp in components:
            local_labels, local_centers = self._find_cluster_centers(comp, densities, summaries)

            # Map local labels to global labels so clusters from different components don't share IDs
            for node_id, loc_label in local_labels.items():
                final_labels[node_id] = global_label_offset + loc_label

            centers.extend(local_centers)

            if local_labels:
                global_label_offset += len(set(local_labels.values()))

        return final_labels, centers

    def predict_components(self, summaries: dict) -> dict:
        """
        Just the "cut the network into connected components" step.

        Returns {node_id: component_id}, a component_id per weakly connected
        subgraph. This is a *coarser* partition than predict_summaries(): the
        density-peak step may later split one component into several clusters,
        but a cluster never spans two components.
        """
        comps = self._connected_components(summaries)
        out: dict = {}
        for comp_id, comp in enumerate(comps):
            for node_id in comp:
                out[node_id] = comp_id
        return out

    def _connected_components(self, summaries: dict) -> list:
        """Weakly connected components over the node-neighbour graph (BFS)."""
        visited = set()
        components = []
        for node in summaries:
            if node not in visited:
                comp = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    # Topological neighbours provided in the summary dict.
                    for nbr in summaries[curr]["neighbors"]:
                        if nbr not in visited and nbr in summaries:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(comp)
        return components

    def predict_assignment(self, assignments: list[int], summaries: dict) -> list[int]:
        """
        map a list of node assignments (one node id per raw point) to
        final macro-cluster labels, using predict_summaries as a lookup table.
        The super class already implements this generic mapping; override
        if DenSOINN needs different behaviour.
        """
        return super().predict_assignment(assignments, summaries)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _node_density(self, summaries: dict) -> dict:
        """
        estimate the density p_i of every node.
        p_i ∝ m_i / (1 + d_i^2), where m_i is the node weight and d_i is the
        node's neighborhood radius (e.g. mean distance to its neighbors).
        Return {node_id: p_i}.
        """
        densities = {}
        for node_id, data in summaries.items():
            m_i = data["weight"]
            w_i = data["centroid"]
            neighbors = data["neighbors"]
            
            if not neighbors:
                # Isolated node
                p_i = float(m_i)
            else:
                # 1. Calculate the mean distance to topological neighbors (d_i)
                dists = [np.linalg.norm(w_i - summaries[nbr]["centroid"]) for nbr in neighbors]
                d_i = np.mean(dists)
                
                # 2. Apply intrinsic dimensionality (n') from the paper
                p_i = m_i / (1.0 + (d_i ** self.intrinsic_dim))
                
            densities[node_id] = p_i
            
        return densities

    def _find_cluster_centers(self, component: list[int], densities: dict, summaries: dict) -> tuple[dict, list]:
        """
        Finds density peaks within a component and cascades labels downhill.

        Returns (local_labels, local_centers): the per-node label map and the
        list of node ids that were chosen as cluster centers (density peaks).
        """
        if not component: 
            return {}, []
        if len(component) == 1: 
            return {component[0]: 0}, [component[0]]  # single node is its own center

        # 1. Sort nodes by density descending
        sorted_nodes = sorted(component, key=lambda nid: densities[nid], reverse=True)

        q = {} # q[i] = nearest node of higher density
        h = {} # h[i] = peak score

        # 2. Find nearest higher-density node and calculate distances
        for k in range(1, len(sorted_nodes)):
            i = sorted_nodes[k]
            w_i = summaries[i]["centroid"]

            # Candidates are nodes with higher density (indices 0 to k-1)
            candidates = sorted_nodes[:k]
            dists = [np.linalg.norm(w_i - summaries[c]["centroid"]) for c in candidates]

            min_idx = np.argmin(dists)
            q_i = candidates[min_idx]
            d_min = dists[min_idx]

            q[i] = q_i
            h[i] = densities[i] * d_min

        # 3. Handle the absolute highest density node (Global Peak of component)
        peak_node = sorted_nodes[0]
        # Its distance is the maximum distance to any other node in the component
        max_dist = max(np.linalg.norm(summaries[peak_node]["centroid"] - summaries[c]["centroid"]) for c in sorted_nodes[1:])
        h[peak_node] = densities[peak_node] * max_dist

        # 4. Calculate threshold and assign centers
        h_mean = np.mean(list(h.values()))
        
        local_labels = {}
        local_centers: list = []
        current_label = 0

        for node in sorted_nodes:
            # If peak score is high enough, it is a cluster center
            if h[node] >= self.alpha * h_mean:
                local_labels[node] = current_label
                local_centers.append(node)
                current_label += 1

        # Fallback: if alpha is too strict and no centers pass, force the peak to be a center
        if not local_labels:
            local_labels[peak_node] = 0
            local_centers.append(peak_node)
            current_label = 1

        # 5. Cascade labels downhill
        for node in sorted_nodes:
            if node not in local_labels:
                # Inherit the label from the nearest higher density node (moving uphill)
                local_labels[node] = local_labels[q[node]]

        return local_labels, local_centers