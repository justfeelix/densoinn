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
    effect of min-max normalization (Theorem 1):
        A_ii = 1 / (max_i - min_i)^2
    where max_i / min_i are taken over the node centroids (not raw inputs,
    which are far more sensitive to outliers).

    For efficiency only the diagonal is stored, as a vector ``a_diag``. With a
    diagonal A the Mahalanobis distance reduces to a weighted Euclidean
    distance,  d^2 = sum_i a_i (x_i - y_i)^2,  which stays fully vectorised.
    This lets the algorithm run on raw (unnormalized) data.
    """

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self.a_diag = np.ones(n_dims, dtype=float)

    @property
    def A(self) -> np.ndarray:
        """The full diagonal matrix form of the metric (for inspection)."""
        return np.diag(self.a_diag)

    def update(self, nodes) -> None:
        """
        Algorithm 1: recompute the diagonal matrix A from the per-dimension
        min/max of all node centroids in ``nodes`` (a {id: Node} mapping or any
        iterable of Nodes).

        For each dimension i:  A_ii = 1 / (max_i - min_i)^2, or 0 if
        max_i == min_i (that feature does not vary across nodes, so it
        contributes nothing to the distance).

        Degeneracy guard: if *every* dimension collapses (e.g. all centroids
        identical), A would be all zeros and winner selection would be
        arbitrary; in that case we fall back to the identity metric for this
        round.
        """
        node_iter = nodes.values() if isinstance(nodes, dict) else nodes
        W = np.array([n.w for n in node_iter], dtype=float)
        if W.shape[0] == 0:
            return

        spread = W.max(axis=0) - W.min(axis=0)
        a_diag = np.zeros(self.n_dims, dtype=float)
        nonzero = spread > 0
        a_diag[nonzero] = 1.0 / (spread[nonzero] ** 2)

        if not nonzero.any():
            a_diag = np.ones(self.n_dims, dtype=float)

        self.a_diag = a_diag

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Mahalanobis distance d_A(x, y) under the current diagonal metric.
        """
        diff = x - y
        return float(np.sqrt(np.dot(self.a_diag * diff, diff)))

    def distance_to_nodes(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Vectorised distance from x to every centroid at once (used by the
        competition step instead of a Python loop).

        Parameters
        ----------
        x : np.ndarray, shape (n_dims,)
        centroids : np.ndarray, shape (n_nodes, n_dims)
            Stacked node centroids, row order defined by the caller.

        Returns
        -------
        np.ndarray, shape (n_nodes,) of distances, aligned with ``centroids``.
        """
        diff = centroids - x
        return np.sqrt((diff * diff) @ self.a_diag)
