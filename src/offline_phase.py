"""
Offline clustering of DenSOINN nodes by density.
"""
import numpy as np
from baseframework.stream_clustering import OfflinePhase_Clusterer
from .distance_metric import SelfAdaptiveDistanceMetric



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
        if not summaries:
            return {}
        
        # 0. Rebuild the self-adaptive metric (Algorithm 1) from the final node centroids, so offline distances stay consistent with the online phase's notion of d(x, y) 
        # instead of falling back to plain Euclidean distance on raw, unnormalized features
        metric = self.build_metric(summaries)

        # 1. Compute densities
        densities = self._node_density(summaries, metric)

        # 2. Split network into Connected Components (BFS)
        components = self._get_components_from_summaries(summaries)

        # 3. Find peaks per component and offset global labels
        final_labels = {}
        global_label_offset = 0

        for comp in components:
            local_labels = self._find_cluster_centers(comp, densities, summaries, metric)
            
            # Map local labels to global labels so clusters from different components don't share IDs
            for node_id, loc_label in local_labels.items():
                final_labels[node_id] = global_label_offset + loc_label

            if local_labels:
                global_label_offset += len(set(local_labels.values()))

        return final_labels


    def predict_assignment(self, assignments: list[int], summaries: dict) -> list[int]:
        """
        map a list of node assignments (one node id per raw point) to
        final macro-cluster labels, using predict_summaries as a lookup table.
        The super class already implements this generic mapping; override
        if DenSOINN needs different behaviour.
        """
        return super().predict_assignment(assignments, summaries)
    

    def predict_summaries_with_stages(self, summaries: dict) -> dict:
        """
        Same clustering as predict_summaries(), but also returns the
        intermediate stages (components, per-component centers) for
        visualization. predict_summaries() remains the lean, label-only
        entry point used by the rest of the pipeline.
        """
        if not summaries:
            return {"components": [], "centers": set(), "labels": {}}

        metric = self.build_metric(summaries)
        densities = self._node_density(summaries, metric)
        components = self._get_components_from_summaries(summaries)

        final_labels = {}
        all_centers = set()
        global_label_offset = 0

        for comp in components:
            local_labels, comp_centers = self._find_cluster_centers_plus(comp, densities, summaries, metric)
            all_centers |= comp_centers

            for node_id, loc_label in local_labels.items():
                final_labels[node_id] = global_label_offset + loc_label

            if local_labels:
                global_label_offset += len(set(local_labels.values()))

        return {"components": components, "centers": all_centers, "labels": final_labels}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def build_metric(self, summaries: dict) -> SelfAdaptiveDistanceMetric:
        """
        Rebuild the same self-adaptive diagonal metric used online (Algorithm 1),
        but from the final, frozen node centroids at clustering time.
        """
        centroids = np.array([data["centroid"] for data in summaries.values()], dtype=float)
        metric = SelfAdaptiveDistanceMetric(n_dims=centroids.shape[1])
        metric.update_from_centroids(centroids)
        return metric

    def _get_components_from_summaries(self, summaries):
        """
        Finds a list of components (connected nodes) from the summaries
        each component is a list of nodes which are connected by an edge in the network. done via BFS with the neighbours of each node.
        """
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
                    
                    # Iterate through the topological neighbors provided in the summary dict
                    for nbr in summaries[curr]["neighbors"]:
                        if nbr not in visited and nbr in summaries:
                            visited.add(nbr)
                            queue.append(nbr)
                            
                components.append(comp)
        return components

    def _node_density(self, summaries: dict, metric: SelfAdaptiveDistanceMetric) -> dict:
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
                dists = [metric.distance(w_i, summaries[nbr]["centroid"]) for nbr in neighbors]
                d_i = np.mean(dists)
                
                # 2. Apply intrinsic dimensionality (n') from the paper
                p_i = m_i / (1.0 + (d_i ** self.intrinsic_dim))
                
            densities[node_id] = p_i
            
        return densities

    def _find_cluster_centers(self, component: list[int], densities: dict, summaries: dict, metric: SelfAdaptiveDistanceMetric) -> dict:
        """
        Finds density peaks within a component and cascades labels downhill.
        """
        if not component: 
            return {}
        if len(component) == 1: 
            return {component[0]: 0} # Only one node, it's its own center

        # 1. Sort nodes by density descending
        sorted_nodes = sorted(component, key=lambda nid: densities[nid], reverse=True)

        q = {} # q[i] = nearest node of higher density
        h = {} # h[i] = peak score

        # 2. Find nearest higher-density node and calculate distances
        for k in range(1, len(sorted_nodes)):
            i = sorted_nodes[k]
            w_i = summaries[i]["centroid"]

            # Candidates are nodes with higher density (indices 0 to k-1)
            candidates = [c for c in sorted_nodes[:k] if densities[c] > densities[i]]
            if not candidates:
                # all higher-ranked nodes are exact density ties with i;
                # fall back to the full higher-or-equal set
                candidates = sorted_nodes[:k]
            dists = [metric.distance(w_i, summaries[c]["centroid"]) for c in candidates]

            min_idx = np.argmin(dists)
            q_i = candidates[min_idx]
            d_min = dists[min_idx]

            q[i] = q_i
            h[i] = densities[i] * d_min

        # 3. Handle the absolute highest density node (Global Peak of component)
        # Peak node: per the paper, h_peak = inf, so it is always a cluster center,
        # and h_mean is computed over the remaining nodes only
        peak_node = sorted_nodes[0]
        non_peak_nodes = sorted_nodes[1:]

        # we give the global peak its center cluster label directly
        local_labels = {peak_node: 0}
        current_label = 1
        # and we exlude it form the threshold calculation
        h_mean = np.mean([h[node] for node in non_peak_nodes])

        for node in non_peak_nodes:
            # If peak score is high enough, it is a cluster center
            if h[node] >= self.alpha * h_mean:
                local_labels[node] = current_label
                current_label += 1

        # 5. Cascade labels downhill
        for node in sorted_nodes:
            if node not in local_labels:
                # Inherit the label from the nearest higher density node (moving uphill)
                local_labels[node] = local_labels[q[node]]

        return local_labels


    def _find_cluster_centers_plus(self, component: list[int], densities: dict, summaries: dict, metric) -> tuple[dict, set]:
        """
        Finds density peaks within a component and cascades labels downhill.
        Returns (local_labels, centers) where centers is the set of node ids
        chosen as cluster centers in this component (before cascading).
        """
        if not component:
            return {}, set()
        if len(component) == 1:
            return {component[0]: 0}, {component[0]}

        sorted_nodes = sorted(component, key=lambda nid: densities[nid], reverse=True)

        q = {}
        h = {}

        for k in range(1, len(sorted_nodes)):
            i = sorted_nodes[k]
            w_i = summaries[i]["centroid"]
            candidates = [c for c in sorted_nodes[:k] if densities[c] > densities[i]]
            if not candidates:
                candidates = sorted_nodes[:k]
            dists = [metric.distance(w_i, summaries[c]["centroid"]) for c in candidates]
            min_idx = np.argmin(dists)
            q_i = candidates[min_idx]
            d_min = dists[min_idx]
            q[i] = q_i
            h[i] = densities[i] * d_min

        peak_node = sorted_nodes[0]
        non_peak_nodes = sorted_nodes[1:]

        local_labels = {peak_node: 0}
        current_label = 1
        h_mean = np.mean([h[node] for node in non_peak_nodes])

        for node in non_peak_nodes:
            if h[node] >= self.alpha * h_mean:
                local_labels[node] = current_label
                current_label += 1

        centers = set(local_labels.keys())  # NEW: capture centers before cascading

        for node in sorted_nodes:
            if node not in local_labels:
                local_labels[node] = local_labels[q[node]]

        return local_labels, centers
