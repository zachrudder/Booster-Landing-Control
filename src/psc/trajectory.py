# src/psc/trajectory.py

import numpy as np
import casadi as ca

from mpc.rocket_model import export_rocket_ode_model
from .nodes import chebyshev_nodes, chebyshev_diff_matrix


def compute_psc_trajectory(
    x0: np.ndarray,
    Tf: float = 3.0,
    N: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a pseudo-spectral trajectory for the booster using Chebyshev
    collocation and the same dynamics as the MPC model.

    Returns:
        ts : (N+1,)   time grid in [0, Tf]
        xs : (N+1,nx) state trajectory
        us : (N+1,nu) control trajectory
    """

    # ----- Dynamics from existing MPC model -----
    model = export_rocket_ode_model()
    x_sym = model.x
    u_sym = model.u
    f_expl = model.f_expl_expr

    nx = int(x_sym.size()[0])
    nu = int(u_sym.size()[0])

    f = ca.Function("f", [x_sym, u_sym], [f_expl])

    # ----- PSC grid -----
    tau = chebyshev_nodes(N)                 # in [-1, 1]
    D = chebyshev_diff_matrix(tau)           # (N+1, N+1)
    ts = (tau + 1.0) * (Tf / 2.0)            # map [-1,1] -> [0,Tf]
    dt_scale = Tf / 2.0

    # ----- Decision variables -----
    X = ca.MX.sym("X", nx, N + 1)  # states at nodes
    U = ca.MX.sym("U", nu, N + 1)  # controls at nodes

    # ----- Cost weights (based on MPC weights) -----
    weight_diag = np.array(model.weight_diag).reshape(-1)
    Q = ca.diag(ca.DM(weight_diag))
    Qf = Q
    R = ca.diag(ca.DM.ones(nu) * 100.0)

    # reference: upright at target altitude (same as MPC)
    x_ref = np.zeros(nx)
    x_ref[0] = 1.0    # q0 = 1
    x_ref[9] = 2.42   # altitude

    # ----- Constraints -----
    g_list = []
    lbg_list = []
    ubg_list = []

    # Initial: x(0) = x0
    g_list.append(X[:, 0] - x0)
    lbg_list.append(np.zeros(nx))
    ubg_list.append(np.zeros(nx))

    # Desired hover configuration for terminal penalty
    x_final = np.zeros(nx)
    x_final[0] = 1.0
    x_final[9] = 2.42

    cost = 0

    for i in range(N + 1):
        # spectral derivative
        xdot_ps = X @ D[i, :].T

        # dynamics
        f_i = f(X[:, i], U[:, i])

        # PSC dynamics constraint
        g_list.append(xdot_ps - dt_scale * f_i)
        lbg_list.append(np.zeros(nx))
        ubg_list.append(np.zeros(nx))

        # running cost
        dx_i = X[:, i] - x_ref
        cost += ca.mtimes([dx_i.T, Q, dx_i]) \
              + ca.mtimes([U[:, i].T, R, U[:, i]])
        
        # quaternion norm + attitude penalties
        q = X[0:4, i]
        q_norm_sq = ca.dot(q, q)
        cost += 1000.0 * (q_norm_sq - 1.0)**2
        cost += 200.0 * ca.sumsqr(X[1:4, i])

    # terminal cost
    dx_f = X[:, -1] - x_ref
    cost += ca.mtimes([dx_f.T, Qf, dx_f])
    cost += 1000.0 * ca.sumsqr(X[:, -1] - x_final)

    # stack constraints
    g = ca.vertcat(*[gi for gi in g_list])
    lbg = np.concatenate(lbg_list)
    ubg = np.concatenate(ubg_list)

    # decision vector
    w = ca.vertcat(
        ca.reshape(X, nx * (N + 1), 1),
        ca.reshape(U, nu * (N + 1), 1),
    )

    # initial guess: interpolate x0 → x_ref, zero u
    w0 = np.zeros(w.shape[0])
    for i in range(N + 1):
        alpha = i / N
        xi = (1 - alpha) * x0 + alpha * x_ref
        xi[13:16] = x0[13:16]
        w0[i * nx:(i + 1) * nx] = xi

    # bounds on w
    lbw = -1e9 * np.ones(w.shape[0])
    ubw =  1e9 * np.ones(w.shape[0])

    THRUST_MAX_N = 1800.0
    THRUST_MAX_ANGLE = np.deg2rad(10.0)
    MAX_RATE = np.deg2rad(20.0)
    for i in range(N + 1):
        offset = i * nx
        lbw[offset + 4: offset + 7] = -MAX_RATE
        ubw[offset + 4: offset + 7] =  MAX_RATE
        lbw[offset + 13] = 0.0
        ubw[offset + 13] = THRUST_MAX_N
        lbw[offset + 14] = -THRUST_MAX_ANGLE
        ubw[offset + 14] = THRUST_MAX_ANGLE
        lbw[offset + 15] = -THRUST_MAX_ANGLE
        ubw[offset + 15] = THRUST_MAX_ANGLE

    # input bounds: SAME as MPC
    lbu = np.array([0.20, -1.0, -1.0, -1.0, -1.0])
    ubu = np.array([1.00,  1.0,  1.0,  1.0,  1.0])

    offset_u = nx * (N + 1)
    for i in range(N + 1):
        start = offset_u + i * nu
        end = start + nu
        lbw[start:end] = lbu
        ubw[start:end] = ubu

    # NLP
    nlp = {"x": w, "f": cost, "g": g}
    opts = {
        "ipopt.print_level": 0,
        "ipopt.max_iter": 200,
        "print_time": 0,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    sol = solver(x0=w0, lbg=lbg, ubg=ubg, lbx=lbw, ubx=ubw)
    w_opt = sol["x"].full().squeeze()

    # unpack
    X_opt = w_opt[: nx * (N + 1)].reshape(nx, N + 1)
    U_opt = w_opt[nx * (N + 1):].reshape(nu, N + 1)

    xs = X_opt.T  # (N+1, nx)
    us = U_opt.T  # (N+1, nu)
    return ts, xs, us
