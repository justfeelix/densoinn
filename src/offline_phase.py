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
        return self.summarize_clustering(summaries)["labels"]


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
        summary = self.summarize_clustering(summaries)
        return {
            "components": summary["components"],
            "centers": summary["centers"],
            "labels": summary["labels"],
        }

    def summarize_clustering(self, summaries: dict) -> dict:
        """
        Run the full offline clustering pass and expose lightweight decision
        diagnostics alongside the final summary labels.
        """
        if not summaries:
            return {
                "components": [],
                "centers": set(),
                "labels": {},
                "selected_center_margins": [],
                "unselected_center_ratios": [],
            }

        # Rebuild the final frozen metric from the surviving summaries so the
        # offline analysis stays consistent with the online distance geometry.
        metric = self.build_metric(summaries)
        densities = self._node_density(summaries, metric)
        components = self._get_components_from_summaries(summaries)

        final_labels = {}
        all_centers = set()
        selected_center_margins: list[float] = []
        unselected_center_ratios: list[float] = []
        global_label_offset = 0

        for comp in components:
            stats = self._find_cluster_centers_stats(comp, densities, summaries, metric)
            local_labels = stats["labels"]
            all_centers |= stats["centers"]
            selected_center_margins.extend(stats["selected_center_margins"])
            unselected_center_ratios.extend(stats["unselected_center_ratios"])

            for node_id, loc_label in local_labels.items():
                final_labels[node_id] = global_label_offset + loc_label

            if local_labels:
                global_label_offset += len(set(local_labels.values()))

        return {
            "components": components,
            "centers": all_centers,
            "labels": final_labels,
            "metric": metric,
            "densities": densities,
            "selected_center_margins": selected_center_margins,
            "unselected_center_ratios": unselected_center_ratios,
        }

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
        return self._find_cluster_centers_stats(component, densities, summaries, metric)["labels"]


    def _find_cluster_centers_plus(self, component: list[int], densities: dict, summaries: dict, metric) -> tuple[dict, set]:
        """
        Finds density peaks within a component and cascades labels downhill.
        Returns (local_labels, centers) where centers is the set of node ids
        chosen as cluster centers in this component (before cascading).
        """
        stats = self._find_cluster_centers_stats(component, densities, summaries, metric)
        return stats["labels"], stats["centers"]

    def _find_cluster_centers_stats(
        self,
        component: list[int],
        densities: dict,
        summaries: dict,
        metric: SelfAdaptiveDistanceMetric,
    ) -> dict:
        """
        Find density peaks and expose how confidently extra centers clear the
        alpha * mean(h) threshold inside their component.
        """
        if not component:
            return {
                "labels": {},
                "centers": set(),
                "selected_center_margins": [],
                "unselected_center_ratios": [],
            }
        if len(component) == 1:
            return {
                "labels": {component[0]: 0},
                "centers": {component[0]},
                "selected_center_margins": [],
                "unselected_center_ratios": [],
            }

        sorted_nodes = sorted(component, key=lambda nid: densities[nid], reverse=True)

        q = {}
        h = {}

        for k in range(1, len(sorted_nodes)):
            node_id = sorted_nodes[k]
            w_i = summaries[node_id]["centroid"]

            candidates = [c for c in sorted_nodes[:k] if densities[c] > densities[node_id]]
            if not candidates:
                candidates = sorted_nodes[:k]
            dists = [metric.distance(w_i, summaries[c]["centroid"]) for c in candidates]
            min_idx = int(np.argmin(dists))
            q_i = candidates[min_idx]
            d_min = float(dists[min_idx])

            q[node_id] = q_i
            h[node_id] = float(densities[node_id] * d_min)

        peak_node = sorted_nodes[0]
        non_peak_nodes = sorted_nodes[1:]
        threshold = float(self.alpha * np.mean([h[node] for node in non_peak_nodes]))

        local_labels = {peak_node: 0}
        centers = {peak_node}
        selected_center_margins: list[float] = []
        unselected_center_ratios: list[float] = []
        current_label = 1

        for node_id in non_peak_nodes:
            if h[node_id] >= threshold:
                local_labels[node_id] = current_label
                centers.add(node_id)
                current_label += 1
                # Normalize the center-threshold gap so the score stays in [0, 1]
                # and remains readable across components with different scales.
                margin = (h[node_id] - threshold) / max(abs(h[node_id]), 1e-12)
                selected_center_margins.append(float(np.clip(margin, 0.0, 1.0)))
            else:
                ratio = h[node_id] / max(abs(threshold), 1e-12)
                unselected_center_ratios.append(float(np.clip(ratio, 0.0, 1.0)))

        for node_id in sorted_nodes:
            if node_id not in local_labels:
                local_labels[node_id] = local_labels[q[node_id]]

        return {
            "labels": local_labels,
            "centers": centers,
            "selected_center_margins": selected_center_margins,
            "unselected_center_ratios": unselected_center_ratios,
        }
