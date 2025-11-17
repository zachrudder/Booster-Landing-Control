"""
Pseudo-spectral collocation trajectory optimizer.

The optimizer discretizes the rocket dynamics with a Legendre-Gauss-Lobatto
mesh and solves the resulting nonlinear program with SciPy's SLSQP backend.
It is intentionally self-contained so it can run anywhere NumPy/SciPy run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .discretization import legendre_gauss_lobatto_mesh
from .rocket_dynamics import RocketDynamics


@dataclass
class PSCTrajectoryConfig:
    horizon_sec: float = 3.0
    polynomial_order: int = 12
    max_iterations: int = 200
    state_cost_diag: np.ndarray = field(
        default_factory=lambda: np.concatenate([np.ones(3) * 10.0, np.ones(13)])
    )
    control_cost_diag: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 1.0, 1.0, 0.5, 0.5])
    )
    terminal_cost_diag: np.ndarray = field(
        default_factory=lambda: np.concatenate([np.ones(3) * 50.0, np.ones(13) * 10.0])
    )
    state_lower: Optional[np.ndarray] = None
    state_upper: Optional[np.ndarray] = None
    target_state: Optional[np.ndarray] = None
    dynamics_penalty: float = 25.0


@dataclass
class TrajectoryOptimizationResult:
    success: bool
    message: str
    cost: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray
    time_grid: np.ndarray


class PSCTrajectoryOptimizer:
    """
    Solves the collocated optimal control problem and exposes the solution to
    the policy.  Warm-starts and mesh customization are handled inside this
    class so the policy can focus on triggering replans.
    """

    def __init__(self, dynamics: RocketDynamics, config: PSCTrajectoryConfig):
        self.dynamics = dynamics
        self.config = config
        self.mesh = legendre_gauss_lobatto_mesh(config.polynomial_order)
        self.mesh_scaled = self.mesh.scaled(0.0, config.horizon_sec)
        self.num_nodes = self.mesh.nodes.size
        self.state_dim = dynamics.state_dim
        self.control_dim = dynamics.control_dim
        self.scale = 2.0 / config.horizon_sec
        self.Q = np.diag(config.state_cost_diag)
        self.R = np.diag(config.control_cost_diag)
        self.Qf = np.diag(config.terminal_cost_diag)
        self.Q_sqrt = np.sqrt(self.Q)
        self.R_sqrt = np.sqrt(self.R)
        self.Qf_sqrt = np.sqrt(self.Qf)
        self.control_bounds = dynamics.control_bounds

    def solve(
        self,
        initial_state: np.ndarray,
        reference_state: Optional[np.ndarray] = None,
        reference_control: Optional[np.ndarray] = None,
        warm_start: Optional[TrajectoryOptimizationResult] = None,
    ) -> TrajectoryOptimizationResult:
        target = (
            reference_state
            if reference_state is not None
            else self.config.target_state
            if self.config.target_state is not None
            else self.dynamics.default_terminal_state()
        )
        u_ref = reference_control if reference_control is not None else np.zeros(
            self.control_dim
        )

        decision0 = (
            self._warm_start_from_solution(warm_start, initial_state, target)
            if warm_start is not None
            else self._generate_initial_guess(initial_state, target)
        )

        lower, upper = self._build_bounds()

        residual_fn = self._residual_function(initial_state, target, u_ref)

        result = least_squares(
            fun=residual_fn,
            x0=decision0,
            bounds=(lower, upper),
            max_nfev=self.config.max_iterations,
            verbose=0,
        )

        states, controls = self._split_decision(result.x)
        return TrajectoryOptimizationResult(
            success=result.success,
            message=result.message,
            cost=result.fun,
            state_trajectory=states,
            control_trajectory=controls,
            time_grid=self.mesh_scaled.nodes,
        )

    def _generate_initial_guess(self, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
        """
        Creates a dynamically consistent warm-start by integrating the dynamics
        with a slowly decaying thrust command.  The final state is forced onto
        the requested terminal condition so the boundary constraint is satisfied
        right away, but intermediate points stem from actual model rollouts.
        """
        states = np.zeros((self.num_nodes, self.state_dim))
        controls = np.zeros((self.num_nodes, self.control_dim))
        state = x0.copy()
        dt = self.config.horizon_sec / max(self.num_nodes - 1, 1)
        u_lower, u_upper = self.control_bounds
        for i in range(self.num_nodes):
            alpha = i / max(self.num_nodes - 1, 1)
            control = np.zeros(self.control_dim)
            hover = 0.7 - 0.4 * alpha
            control[0] = np.clip(hover, u_lower[0], u_upper[0])
            control[1:] = 0.0
            control = np.clip(control, u_lower, u_upper)
            states[i] = state
            controls[i] = control
            if i < self.num_nodes - 1:
                derivative = self.dynamics.state_derivative(state, control)
                state = state + dt * derivative
                state[0:4] /= np.linalg.norm(state[0:4]) + 1e-9
        states[-1] = xf
        return self._pack_decision(states, controls)

    def _warm_start_from_solution(
        self,
        previous: TrajectoryOptimizationResult,
        new_initial_state: np.ndarray,
        target_state: np.ndarray,
    ) -> np.ndarray:
        states = previous.state_trajectory.copy()
        controls = previous.control_trajectory.copy()
        states = np.vstack((new_initial_state, states[:-1]))
        controls = np.vstack((controls[1:], controls[-1]))
        states[-1] = target_state
        return self._pack_decision(states, controls)

    def _pack_decision(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return np.concatenate([states.flatten(), controls.flatten()])

    def _split_decision(self, decision: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.state_dim * self.num_nodes
        states_flat = decision[:n]
        controls_flat = decision[n:]
        states = states_flat.reshape(self.num_nodes, self.state_dim)
        controls = controls_flat.reshape(self.num_nodes, self.control_dim)
        return states, controls

    def _dynamics_constraint(self, decision: np.ndarray) -> np.ndarray:
        states, controls = self._split_decision(decision)
        derivatives = np.zeros_like(states)
        for i in range(self.num_nodes):
            derivatives[i] = self.dynamics.state_derivative(states[i], controls[i])
        residual = self.scale * (self.mesh.diff_matrix @ states) - derivatives
        return residual.flatten()

    def _residual_function(
        self, x0: np.ndarray, xf: np.ndarray, u_ref: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        def residual(decision: np.ndarray) -> np.ndarray:
            states, controls = self._split_decision(decision)
            refs = self._reference_trajectory(x0, xf)
            res = []
            dynamics_res = self._dynamics_constraint(decision)
            res.append(np.sqrt(self.config.dynamics_penalty) * dynamics_res)
            res.append(states[0] - x0)
            res.append(states[-1] - xf)
            for i in range(self.num_nodes):
                x_err = states[i] - refs[i]
                u_err = controls[i] - u_ref
                res.append(self.Q_sqrt @ x_err)
                res.append(self.R_sqrt @ u_err)
            terminal_err = states[-1] - xf
            res.append(self.Qf_sqrt @ terminal_err)
            return np.concatenate(res)

        return residual

    def _reference_trajectory(self, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
        alphas = np.linspace(0.0, 1.0, self.num_nodes)
        return ((1 - alphas)[:, None] * x0[None, :]) + (alphas[:, None] * xf[None, :])

    def _build_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        state_lower = (
            self.config.state_lower
            if self.config.state_lower is not None
            else np.full(self.state_dim, -np.inf)
        )
        state_upper = (
            self.config.state_upper
            if self.config.state_upper is not None
            else np.full(self.state_dim, np.inf)
        )
        u_lower, u_upper = self.control_bounds
        lower_list = []
        upper_list = []
        for _ in range(self.num_nodes):
            lower_list.extend(state_lower)
            upper_list.extend(state_upper)
        for _ in range(self.num_nodes):
            lower_list.extend(u_lower)
            upper_list.extend(u_upper)
        return np.array(lower_list), np.array(upper_list)
