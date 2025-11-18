"""
Helper utilities to build the PSC acados OCP for offline code generation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, Function, reshape, vertcat

from mpc.rocket_model import export_rocket_ode_model
from .psc_layout import PSCDecisionLayout


def build_psc_ocp(
    mesh,
    horizon_sec: float,
    state_cost_diag: np.ndarray,
    control_cost_diag: np.ndarray,
    state_lower: np.ndarray,
    state_upper: np.ndarray,
    control_bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[AcadosOcp, PSCDecisionLayout]:
    layout = PSCDecisionLayout(state_dim=16, control_dim=5, num_nodes=mesh.nodes.size)

    model = _build_model(mesh, horizon_sec, layout)
    ocp = _build_ocp(
        model=model,
        layout=layout,
        mesh=mesh,
        state_cost_diag=state_cost_diag,
        control_cost_diag=control_cost_diag,
        state_lower=state_lower,
        state_upper=state_upper,
        control_bounds=control_bounds,
    )
    return ocp, layout


def _build_model(mesh, horizon_sec: float, layout: PSCDecisionLayout) -> AcadosModel:
    decision = SX.sym("z", layout.decision_dim, 1)
    decision_dot = SX.sym("zdot", layout.decision_dim, 1)
    u_dummy = SX.sym("u_dummy", 0, 1)
    p = SX.sym("p_psc", 2 * layout.state_dim, 1)

    states = reshape(
        decision[0 : layout.state_block], layout.state_dim, layout.num_nodes
    ).T
    controls = reshape(
        decision[layout.state_block : layout.decision_dim],
        layout.control_dim,
        layout.num_nodes,
    ).T

    model_ct = export_rocket_ode_model()
    dyn_fun = Function("psc_dynamics", [model_ct.x, model_ct.u], [model_ct.f_expl_expr])

    diff_matrix = mesh.diff_matrix
    scale = 2.0 / horizon_sec
    h_terms = []
    for i in range(layout.num_nodes):
        colloc = SX.zeros(layout.state_dim, 1)
        for j in range(layout.num_nodes):
            colloc += diff_matrix[i, j] * states[j, :].T
        x_sub = states[i, :].T
        u_sub = controls[i, :].T
        dyn = dyn_fun(x_sub, u_sub)
        residual = scale * colloc - dyn
        h_terms.append(residual)

    h_terms.append(states[0, :].T - p[0 : layout.state_dim])
    h_terms.append(states[-1, :].T - p[layout.state_dim :])
    con_h_expr = vertcat(*h_terms)

    model = AcadosModel()
    model.x = decision
    model.xdot = decision_dot
    model.u = u_dummy
    model.p = p
    model.name = "psc_collocation"
    model.f_expl_expr = SX.zeros(layout.decision_dim, 1)
    model.f_impl_expr = decision_dot
    model.con_h_expr = con_h_expr
    return model


def _build_ocp(
    model: AcadosModel,
    layout: PSCDecisionLayout,
    mesh,
    state_cost_diag: np.ndarray,
    control_cost_diag: np.ndarray,
    state_lower: np.ndarray,
    state_upper: np.ndarray,
    control_bounds: Tuple[np.ndarray, np.ndarray],
) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = 1
    ocp.dims.nx = layout.decision_dim
    ocp.dims.nu = 0
    ocp.dims.ny = layout.decision_dim
    ocp.dims.ny_e = 0
    ocp.parameter_values = np.zeros(2 * layout.state_dim)

    state_weights = _expand_weights(mesh.weights, state_cost_diag)
    control_weights = _expand_weights(mesh.weights, control_cost_diag)
    W = np.diag(np.concatenate([state_weights, control_weights]))

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.W = W
    ocp.cost.Vx = np.eye(layout.decision_dim)
    ocp.cost.Vu = np.zeros((layout.decision_dim, 0))
    ocp.cost.yref = np.zeros(layout.decision_dim)

    constraint_dim = model.con_h_expr.shape[0]
    ocp.constraints.constr_type = "BGH"
    ocp.dims.nh = constraint_dim
    ocp.constraints.lh = np.zeros(constraint_dim)
    ocp.constraints.uh = np.zeros(constraint_dim)

    lower, upper = _expand_bounds(
        layout.num_nodes, state_lower, state_upper, control_bounds
    )
    ocp.constraints.lbx = lower
    ocp.constraints.ubx = upper
    ocp.constraints.idxbx = np.arange(layout.decision_dim)

    ocp.solver_options.tf = 1.0
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = "SQP"
    return ocp


def _expand_bounds(
    num_nodes: int,
    state_lower: np.ndarray,
    state_upper: np.ndarray,
    control_bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for _ in range(num_nodes):
        lower.extend(state_lower)
        upper.extend(state_upper)
    u_lower, u_upper = control_bounds
    for _ in range(num_nodes):
        lower.extend(u_lower)
        upper.extend(u_upper)
    return np.array(lower), np.array(upper)


def _expand_weights(weights: np.ndarray, diag: np.ndarray) -> np.ndarray:
    expanded = []
    for i in range(weights.size):
        expanded.extend(weights[i] * diag)
    return np.array(expanded)
