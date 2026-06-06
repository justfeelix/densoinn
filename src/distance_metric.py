"""
Self-adaptive distance metric for DenSOINN
"""
import numpy as np


class SelfAdaptiveDistanceMetric:
    """
    Self-adaptive Mahalanobis distance metric.

    The metric uses a positive semi-definite parameter matrix A so that
    d_A(x, y) = sqrt((x - y)^T A (x - y))

    A is recomputed at the beginning of every learning round from the current
    node centroids. A is a diagonal matrix that approximates the
    effect of min-max normalization.
    This lets the algorithm run on raw (unnormalized) data.
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        # TODO: initialize A (e.g. diagonal matrix of ones until enough nodes exist)
        self.A = None

    def update(self, nodes) -> None:
        """
        TODO (Algorithm 1): recompute the diagonal matrix A from the
        per-dimension min/max of all node centroids in `nodes`.
        - for each dimension i: find max_i and min_i over all node.w
        - A_ii = 1 / (max_i - min_i)^2, or 0 if max_i == min_i
        """
        pass

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        TODO: return the Mahalanobis distance d_A(x, y) between two
        feature vectors under the current matrix A.
        """
        pass

    def distance_to_nodes(self, x: np.ndarray, nodes) -> np.ndarray:
        """
        TODO (efficiency helper): vectorised distance from x to all node
        centroids at once. Used by the competition step instead of
        calling distance() in a Python loop.
        """
        pass
