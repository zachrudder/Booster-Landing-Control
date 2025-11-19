# src/tvlqr.py

import numpy as np
import casadi as ca
from typing import Optional

from mpc.rocket_model import export_rocket_ode_model


def _make_linearized_dynamics():
    model = export_rocket_ode_model()
    x = model.x
    u = model.u
    f_expl = model.f_expl_expr

    f = ca.Function("f", [x, u], [f_expl])
    A_sym = ca.jacobian(f_expl, x)
    B_sym = ca.jacobian(f_expl, u)

    A_fun = ca.Function("A_fun", [x, u], [A_sym])
    B_fun = ca.Function("B_fun", [x, u], [B_sym])

    nx = int(x.size()[0])
    nu = int(u.size()[0])

    return f, A_fun, B_fun, nx, nu


def compute_tvlqr_gains(
    ts: np.ndarray,
    xs: np.ndarray,
    us: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """
    Compute discrete-time TVLQR gains along a given trajectory.
    """
    _, A_fun, B_fun, nx, nu = _make_linearized_dynamics()

    N = len(ts) - 1
    dt = np.diff(ts)

    if Qf is None:
        Qf = Q

    A_list = []
    B_list = []

    for k in range(N):
        xk = xs[k, :]
        uk = us[k, :]
        A_k = A_fun(xk, uk).full()
        B_k = B_fun(xk, uk).full()

        Ak_disc = np.eye(nx) + dt[k] * A_k
        Bk_disc = dt[k] * B_k

        A_list.append(Ak_disc)
        B_list.append(Bk_disc)

    P = Qf.copy()
    Ks = [None] * N

    for k in reversed(range(N)):
        A = A_list[k]
        B = B_list[k]

        S = R + B.T @ P @ B
        K = -np.linalg.solve(S, B.T @ P @ A)
        Ks[k] = K

        P = Q + A.T @ P @ (A + B @ K)

    return Ks
