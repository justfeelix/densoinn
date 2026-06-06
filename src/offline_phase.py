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

    NOTE: this phase needs the network topology, not just isolated centroids.
    The connected components come from the graph edges (see
    DenSOINNNetwork.connected_components). The provided DensityPeakClustering
    helper can optionally be reused per component for the peak logic.
    """

    def __init__(self, alpha: float = 5.0, intrinsic_dim: int = 2):
        super().__init__()
        self.alpha = alpha
        self.intrinsic_dim = intrinsic_dim

    def predict_summaries(self, summaries: dict) -> dict:
        """
        TODO: cluster the nodes and return a dict
        {node_id: macro_cluster_label} (noise label is -1).
          1. compute node density p_i for every node
          2. split the network into connected components
          3. for each component:
               - sort nodes by descending density
               - for each node, find nearest node of higher density q_i
                 and its peak score h_i = p_i * d(w_i, w_qi)
               - compute mean peak score h_mean
               - a node is a center if h_i >= alpha * h_mean
                 otherwise it joins the cluster of q_i
        """
        # input:  dict of node summaries (centroid, weight, neighbors)
        # output: dict mapping node_id -> final cluster label
        pass

    def predict_assignment(self, assignments: list[int], summaries: dict) -> list[int]:
        """
        TODO: map a list of node assignments (one node id per raw point) to
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
        TODO: estimate the density p_i of every node.
        p_i ∝ m_i / (1 + d_i^2), where m_i is the node weight and d_i is the
        node's neighborhood radius (e.g. mean distance to its neighbors).
        Return {node_id: p_i}.
        """
        pass

    def _find_cluster_centers(self, component: list[int], densities: dict, summaries: dict) -> dict:
        """
        TODO: given one connected component, find density
        peaks (q_i, h_i), apply the alpha * h_mean rule, and return the local
        labeling for the nodes in this component.
        """
        pass
