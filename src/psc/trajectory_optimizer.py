"""
High-level PSC trajectory optimizer that wraps the acados solver.

The acados solver treats the entire collocation trajectory as its state vector.
We provide helper methods to generate reference trajectories, warm-starts, and
post-process the optimization result so the policy can consume the same API as
before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .acados_psc_solver import PSCAcadosSolver
from .discretization import CollocationMesh, legendre_gauss_lobatto_mesh
from .rocket_dynamics import RocketDynamics


@dataclass
class PSCTrajectoryConfig:
    horizon_sec: float = 2.2
    polynomial_order: int = 10
    state_cost_diag: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                6.0,
                6.0,
                6.0,
                6.0,
                80.1,
                80.1,
                0.5,
                1.0,
                1.0,
                3.0,
                5.1,
                5.1,
                25.0,
                10.0,
                10.0,
                10.0,
            ]
        )
    )
    control_cost_diag: np.ndarray = field(
        default_factory=lambda: np.array([200.0, 200.0, 200.0, 600.0, 600.0])
    )
    state_lower: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -1.0,
                -0.2,
                -0.2,
                -0.2,
                -0.35,
                -0.35,
                -0.35,
                -200.0,
                -200.0,
                -200.0,
                -20.0,
                -20.0,
                -20.0,
                0.0,
                -0.25,
                -0.25,
            ]
        )
    )
    state_upper: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                1.0,
                0.2,
                0.2,
                0.2,
                0.35,
                0.35,
                0.35,
                200.0,
                200.0,
                200.0,
                20.0,
                20.0,
                20.0,
                2000.0,
                0.25,
                0.25,
            ]
        )
    )


@dataclass
class TrajectoryOptimizationResult:
    success: bool
    message: str
    cost: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray
    time_grid: np.ndarray
    raw_decision: np.ndarray


class PSCTrajectoryOptimizer:
    def __init__(self, dynamics: RocketDynamics, config: PSCTrajectoryConfig):
        self.dynamics = dynamics
        self.config = config
        self.mesh = legendre_gauss_lobatto_mesh(config.polynomial_order)
        self.mesh_scaled = self.mesh.scaled(0.0, config.horizon_sec)
        self.num_nodes = self.mesh.nodes.size
        self.state_dim = dynamics.state_dim
        self.control_dim = dynamics.control_dim
        self.control_bounds = dynamics.control_bounds
        self.solver = PSCAcadosSolver(
            mesh=self.mesh_scaled,
            horizon_sec=config.horizon_sec,
            state_cost_diag=config.state_cost_diag,
            control_cost_diag=config.control_cost_diag,
            state_lower=config.state_lower,
            state_upper=config.state_upper,
            control_bounds=self.control_bounds,
            json_file="acados_ocp_psc_collocation.json",
        )
        self._warm_solution: Optional[np.ndarray] = None

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
            else self.dynamics.default_terminal_state()
        )
        ref_traj = self._reference_vector(initial_state, target)
        guess = (
            warm_start.raw_decision
            if warm_start is not None
            else self._initial_guess(initial_state, target)
        )

        decision, status = self.solver.solve(
            x0=initial_state,
            xf=target,
            reference_trajectory=ref_traj,
            initial_guess=guess,
        )
        states, controls = self._split_decision(decision)
        success = status == 0
        message = "SQP converged" if success else f"acados error code {status}"
        result = TrajectoryOptimizationResult(
            success=success,
            message=message,
            cost=0.0,
            state_trajectory=states,
            control_trajectory=controls,
            time_grid=self.mesh_scaled.nodes,
            raw_decision=decision,
        )
        return result

    def _initial_guess(self, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
        states = self._reference_trajectory(x0, xf)
        controls = np.tile(
            0.5 * (self.control_bounds[0] + self.control_bounds[1]),
            (self.num_nodes, 1),
        )
        return np.concatenate([states.flatten(), controls.flatten()])

    def _split_decision(self, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.state_dim * self.num_nodes
        states = decision[:n].reshape(self.num_nodes, self.state_dim)
        controls = decision[n:].reshape(self.num_nodes, self.control_dim)
        return states, controls

    def _reference_trajectory(self, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
        alphas = np.linspace(0.0, 1.0, self.num_nodes)
        return ((1 - alphas)[:, None] * x0[None, :]) + (alphas[:, None] * xf[None, :])

    def _reference_vector(self, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
        states = self._reference_trajectory(x0, xf).flatten()
        controls = np.tile(0.5, self.control_dim * self.num_nodes)
        return np.concatenate([states, controls])
