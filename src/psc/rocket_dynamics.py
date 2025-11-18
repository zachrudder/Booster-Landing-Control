"""
Numerical rocket dynamics used by the pseudo-spectral controller.

The MPC policy relies on a CasADi model definition.  Reusing it directly is
hard in a custom trajectory optimizer, so this module re-implements the same
equations with NumPy.  The model mirrors the acados version closely enough for
trajectory planning while keeping the implementation approachable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


StateVector = np.ndarray
ControlVector = np.ndarray
ForceTorque = Tuple[np.ndarray, np.ndarray]


def _default_disturbance(_: float, __: StateVector, ___: ControlVector) -> ForceTorque:
    zero = np.zeros(3)
    return zero, zero


@dataclass
class RocketPhysicalParams:
    gravity: float = 9.81
    mass_kg: float = 91.0
    J_diag: Tuple[float, float, float] = (372.6, 372.6, 1.55)
    thrust_max: float = 1800.0
    thrust_tau: float = 2.5
    thrust_vector_tau: float = 0.3
    thrust_max_angle: float = np.deg2rad(10.0)
    att_max_thrust: float = 50.0
    booster_pos: Tuple[float, float, float] = (0.0, 0.0, -2.0)
    att_thruster_pos: Tuple[float, float, float] = (0.0, 0.0, 2.0)

    def inertia_matrix(self) -> np.ndarray:
        return np.diag(self.J_diag)


def _skew(vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]
    )


@dataclass
class RocketDynamics:
    """
    Lightweight dynamics model suitable for direct collocation methods.

    Parameters
    ----------
    params : RocketPhysicalParams
        Physical constants.  Keep them aligned with ``mpc/rocket_model.py``.
    disturbance_fn : Callable
        Optional function that injects aerodynamic disturbances as
        ``force, torque`` tuples.  This enables Monte Carlo experiments where
        drag acts at random points on the rocket.
    """

    params: RocketPhysicalParams = field(default_factory=RocketPhysicalParams)
    disturbance_fn: Callable[[float, StateVector, ControlVector], ForceTorque] = (
        _default_disturbance
    )

    def __post_init__(self) -> None:
        self._J = self.params.inertia_matrix()
        self._J_inv = np.linalg.inv(self._J)
        self._booster_cross = _skew(np.asarray(self.params.booster_pos))
        self._att_cross = _skew(np.asarray(self.params.att_thruster_pos))

    @property
    def state_dim(self) -> int:
        return 16

    @property
    def control_dim(self) -> int:
        return 5

    @property
    def control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([0.2, -1.0, -1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        return lower, upper

    def default_terminal_state(self, altitude_m: float = 2.5) -> np.ndarray:
        target = np.zeros(self.state_dim)
        target[0] = 1.0
        target[9] = altitude_m
        target[13] = 0.0
        target[14] = 0.0
        target[15] = 0.0
        return target

    def state_derivative(
        self, state: StateVector, control: ControlVector, time_sec: float = 0.0
    ) -> StateVector:
        assert state.shape[0] == self.state_dim
        assert control.shape[0] == self.control_dim

        q = self._normalize_quaternion(state[0:4])
        omega = state[4:7]
        vel = state[10:13]
        thrust = state[13]
        t_alpha = state[14]
        t_beta = state[15]

        thrust_cmd = control[0]
        alpha_cmd = control[1]
        beta_cmd = control[2]
        att_x = control[3]
        att_y = control[4]

        thrust_dot = (self.params.thrust_max * thrust_cmd - thrust) / self.params.thrust_tau
        alpha_dot = (self.params.thrust_max_angle * alpha_cmd - t_alpha) / self.params.thrust_vector_tau
        beta_dot = (self.params.thrust_max_angle * beta_cmd - t_beta) / self.params.thrust_vector_tau

        thrust_body = np.array([thrust * t_alpha, thrust * t_beta, thrust])

        att_thrust = np.array(
            [
                att_x * self.params.att_max_thrust,
                att_y * self.params.att_max_thrust,
                0.0,
            ]
        )

        torque = self._booster_cross @ thrust_body + self._att_cross @ att_thrust

        gravity_n = np.array([0.0, 0.0, -self.params.gravity])
        R_b_to_n = rotation_matrix_from_quaternion(q)

        thrust_n = R_b_to_n @ thrust_body

        disturbance_force, disturbance_torque = self.disturbance_fn(
            time_sec, state, control
        )
        vel_dot = (
            thrust_n / self.params.mass_kg
            + gravity_n
            + disturbance_force / max(self.params.mass_kg, 1e-3)
        )
        omega_dot = self._J_inv @ (
            torque + disturbance_torque - np.cross(omega, self._J @ omega)
        )

        q_dot = 0.5 * np.array(
            [
                -omega[0] * q[1] - omega[1] * q[2] - omega[2] * q[3],
                omega[0] * q[0] + omega[2] * q[2] - omega[1] * q[3],
                omega[1] * q[0] - omega[2] * q[1] + omega[0] * q[3],
                omega[2] * q[0] + omega[1] * q[1] - omega[0] * q[2],
            ]
        )

        pos_dot = vel
        state_dot = np.zeros_like(state)
        state_dot[0:4] = q_dot
        state_dot[4:7] = omega_dot
        state_dot[7:10] = pos_dot
        state_dot[10:13] = vel_dot
        state_dot[13] = thrust_dot
        state_dot[14] = alpha_dot
        state_dot[15] = beta_dot
        return state_dot

    def linearize(
        self,
        state: StateVector,
        control: ControlVector,
        time_sec: float = 0.0,
        epsilon: float = 1e-5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numerically linearizes the dynamics around ``state, control``.
        """
        n = self.state_dim
        m = self.control_dim
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        base = self.state_derivative(state, control, time_sec)

        for i in range(n):
            perturb = np.zeros(n)
            perturb[i] = epsilon
            derivative = self.state_derivative(state + perturb, control, time_sec)
            A[:, i] = (derivative - base) / epsilon

        for j in range(m):
            perturb = np.zeros(m)
            perturb[j] = epsilon
            derivative = self.state_derivative(state, control + perturb, time_sec)
            B[:, j] = (derivative - base) / epsilon

        return A, B

    @staticmethod
    def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
