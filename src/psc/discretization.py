"""
Pseudo-spectral collocation utilities.

The functions in this module create Legendre-Gauss-Lobatto (LGL) meshes, the
associated quadrature weights, and the differentiation matrix that is required
for orthogonal collocation.  The helper hides the math so policy code can focus
on defining dynamics and costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.polynomial.legendre import legder, legroots, legval


@dataclass(frozen=True)
class CollocationMesh:
    """
    Container for nodes, quadrature weights and the differentiation matrix.

    Attributes
    ----------
    nodes : np.ndarray
        Collocation nodes in [-1, 1].
    weights : np.ndarray
        Quadrature weights matching ``nodes``.
    diff_matrix : np.ndarray
        Differentiation matrix ``D`` such that D @ x approximates dx/dt on
        the LGL grid.
    """

    nodes: np.ndarray
    weights: np.ndarray
    diff_matrix: np.ndarray

    def scaled(self, t0: float, tf: float) -> "CollocationMesh":
        """
        Returns a mesh scaled to [t0, tf] while reusing the differentiation
        matrix.  Only the nodes and weights change during this operation.
        """
        scaled_nodes = 0.5 * (tf - t0) * self.nodes + 0.5 * (tf + t0)
        scaled_weights = 0.5 * (tf - t0) * self.weights
        return CollocationMesh(
            nodes=scaled_nodes, weights=scaled_weights, diff_matrix=self.diff_matrix
        )


def legendre_gauss_lobatto_mesh(order: int) -> CollocationMesh:
    """
    Creates an LGL mesh of ``order`` + 1 nodes.

    Parameters
    ----------
    order : int
        Polynomial order of the collocation.  The mesh contains ``order + 1``
        nodes (including -1 and 1).
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    if order == 1:
        nodes = np.array([-1.0, 1.0])
        weights = np.array([1.0, 1.0])
    else:
        coeffs = np.zeros(order + 1)
        coeffs[-1] = 1.0
        dcoeffs = legder(coeffs)
        interior = legroots(dcoeffs)
        nodes = np.concatenate(([-1.0], interior, [1.0]))
        weights = _lgl_weights(nodes, order)

    diff_matrix = _lgl_diff_matrix(nodes)
    return CollocationMesh(nodes=nodes, weights=weights, diff_matrix=diff_matrix)


def _lgl_weights(nodes: np.ndarray, order: int) -> np.ndarray:
    weights = np.zeros_like(nodes)
    coeffs = np.zeros(order + 1)
    coeffs[-1] = 1.0
    for i, node in enumerate(nodes):
        Pn = legval(node, coeffs)
        weights[i] = 2.0 / (order * (order + 1) * Pn * Pn)
    return weights


def _lgl_diff_matrix(nodes: np.ndarray) -> np.ndarray:
    N = len(nodes) - 1
    D = np.zeros((N + 1, N + 1))
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0

    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                continue
            factor = c[i] / c[j]
            D[i, j] = factor / (nodes[i] - nodes[j])
    D[np.diag_indices_from(D)] = -np.sum(D, axis=1)
    return D


def enforce_state_bounds(
    values: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Projects ``values`` into the provided bounds.  Projection is a simple clip
    operation; the helper is handy when crafting warm-start trajectories.
    """
    lower, upper = bounds
    return np.clip(values, lower, upper)
