"""
Time-varying LQR tracker for the pseudo-spectral trajectories.

The optimized trajectory is open-loop.  A disturbance (e.g. random drag
patches) will immediately kick the rocket off the path, so we add a feedback
layer.  The tracker linearizes the nonlinear model along the trajectory and
runs a backward Riccati recursion to produce gains ``K_k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.linalg import inv

from .rocket_dynamics import RocketDynamics


@dataclass
class LQRTrackerConfig:
    dt_sec: float = 1.0 / 20.0
    state_cost_diag: np.ndarray = field(
        default_factory=lambda: np.concatenate(
            [
                np.ones(4) * 25.0,
                np.ones(3) * 5.0,
                np.ones(3) * 1.0,
                np.ones(3) * 4.0,
                np.ones(3),
            ]
        )
    )
    control_cost_diag: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 2.0, 2.0, 0.5, 0.5])
    )
    terminal_cost_diag: np.ndarray = field(
        default_factory=lambda: np.concatenate(
            [
                np.ones(4) * 60.0,
                np.ones(3) * 10.0,
                np.ones(3) * 3.0,
                np.ones(3) * 8.0,
                np.ones(3) * 3.0,
            ]
        )
    )


class LQRTracker:
    def __init__(self, dynamics: RocketDynamics, config: LQRTrackerConfig):
        self.dynamics = dynamics
        self.config = config
        self.Q = np.diag(config.state_cost_diag)
        self.R = np.diag(config.control_cost_diag)
        self.Qf = np.diag(config.terminal_cost_diag)
        self._reference_states: np.ndarray = np.zeros((1, dynamics.state_dim))
        self._reference_controls: np.ndarray = np.zeros((1, dynamics.control_dim))
        self._gains: List[np.ndarray] = []
        self._feedforward: List[np.ndarray] = []

    def update_schedule(
        self, states: np.ndarray, controls: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """
        Pre-compute the gain schedule for the provided reference trajectory.
        """
        assert states.shape[0] == controls.shape[0]
        assert states.shape[0] == time_grid.shape[0]
        self._reference_states = states
        self._reference_controls = controls
        A_seq: List[np.ndarray] = []
        B_seq: List[np.ndarray] = []
        dt_seq = np.diff(time_grid, prepend=time_grid[0])
        for i in range(states.shape[0]):
            A_c, B_c = self.dynamics.linearize(states[i], controls[i], time_grid[i])
            dt = max(dt_seq[i], 1e-3)
            A_d = np.eye(self.dynamics.state_dim) + A_c * dt
            B_d = B_c * dt
            A_seq.append(A_d)
            B_seq.append(B_d)

        self._gains, self._feedforward = self._backward_riccati(A_seq, B_seq)

    def control(self, state: np.ndarray, step_idx: int) -> np.ndarray:
        """
        Returns the feedback corrected control command for the provided state.
        """
        step = min(step_idx, len(self._gains) - 1)
        K = self._gains[step]
        u_ff = self._feedforward[step]
        x_ref = self._reference_states[min(step, self._reference_states.shape[0] - 1)]
        control = u_ff - K @ (state - x_ref)
        lower, upper = self.dynamics.control_bounds
        return np.clip(control, lower, upper)

    def _backward_riccati(
        self, A_seq: List[np.ndarray], B_seq: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        N = len(A_seq)
        P = self.Qf.copy()
        gains: List[np.ndarray] = [
            np.zeros((self.dynamics.control_dim, self.dynamics.state_dim)) for _ in range(N)
        ]
        feedforward: List[np.ndarray] = [
            np.zeros(self.dynamics.control_dim) for _ in range(N)
        ]
        for k in reversed(range(N)):
            A = A_seq[k]
            B = B_seq[k]
            S = self.R + B.T @ P @ B
            K = inv(S) @ (B.T @ P @ A)
            P = self.Q + A.T @ P @ (A - B @ K)
            gains[k] = K
            idx = min(k, self._reference_controls.shape[0] - 1)
            feedforward[k] = self._reference_controls[idx]
        return gains, feedforward
