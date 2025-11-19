# src/psc/nodes.py

import numpy as np


def chebyshev_nodes(N: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1]."""
    k = np.arange(N + 1)
    tau = np.cos(np.pi * k / N)
    return tau


def chebyshev_diff_matrix(tau: np.ndarray) -> np.ndarray:
    """Differentiation matrix D for Chebyshev nodes tau."""
    N = len(tau) - 1
    D = np.zeros((N + 1, N + 1))

    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** np.arange(N + 1)

    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) / (tau[i] - tau[j])
        D[i, i] = -np.sum(D[i, :])

    return D
