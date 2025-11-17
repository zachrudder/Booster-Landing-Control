"""
Pseudo-spectral trajectory optimization + LQR tracking policy.

Usage
-----
>>> from pscpolicy import PSCPolicy
>>> policy = PSCPolicy(initial_state)
>>> action, prediction = policy.next(state_vector)

The policy keeps the original interface of ``MPCPolicy`` so it can be dropped
into ``rocketcraft.py`` by changing a single import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from basecontrol import BaseControl
from psc import (
    LQRTracker,
    LQRTrackerConfig,
    PSCTrajectoryConfig,
    PSCTrajectoryOptimizer,
    TrajectoryOptimizationResult,
    RocketDynamics,
    RocketPhysicalParams,
)


DisturbanceCallback = Callable[[float, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class PSCPolicyConfig:
    """
    Configuration wrapper combining optimizer, tracker, and policy options.
    """

    horizon_sec: float = 4.0
    polynomial_order: int = 20
    replan_interval_steps: int = 5
    max_replan_failures: int = 5
    optimizer_config: Optional[PSCTrajectoryConfig] = None
    tracker_config: Optional[LQRTrackerConfig] = None
    disturbance_callback: Optional[DisturbanceCallback] = None


class PSCPolicy(BaseControl):
    """
    Implements a two-layer guidance architecture:

    1. A pseudo-spectral collocation optimizer plans a minimum-effort glide
       trajectory from the current state down to the landing target.
    2. A time-varying LQR tracker clamps the plan to the physical rocket,
       compensating for disturbances such as ad-hoc drag patches.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        config: Optional[PSCPolicyConfig] = None,
        physical_params: Optional[RocketPhysicalParams] = None,
    ):
        super().__init__()
        self.config = config or PSCPolicyConfig()
        params = physical_params or RocketPhysicalParams()

        dynamics = RocketDynamics(
            params,
            disturbance_fn=self.config.disturbance_callback
            if self.config.disturbance_callback is not None
            else lambda t, x, u: (np.zeros(3), np.zeros(3)),
        )

        optimizer_cfg = self.config.optimizer_config or PSCTrajectoryConfig(
            horizon_sec=self.config.horizon_sec,
            polynomial_order=self.config.polynomial_order,
        )
        tracker_cfg = self.config.tracker_config or LQRTrackerConfig()

        self.dynamics = dynamics
        self.optimizer = PSCTrajectoryOptimizer(dynamics, optimizer_cfg)
        self.tracker = LQRTracker(dynamics, tracker_cfg)
        self._active_plan: Optional[TrajectoryOptimizationResult] = None
        self._plan_step = 0
        self._replan_failures = 0
        self._last_control = np.zeros(dynamics.control_dim)
        self._target_state = (
            optimizer_cfg.target_state
            if optimizer_cfg.target_state is not None
            else dynamics.default_terminal_state()
        )
        self._plan_from_state(initial_state, warm_start=None)

    def get_name(self) -> str:
        return "PSC-LQR"

    def next(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        need_replan = (
            self._active_plan is None
            or self._plan_step >= self._active_plan.state_trajectory.shape[0] - 2
            or self._plan_step % self.config.replan_interval_steps == 0
        )
        if need_replan:
            warm_start = self._active_plan if self._active_plan and self._active_plan.success else None
            self._plan_from_state(observation, warm_start)

        if self._active_plan is None or self._replan_failures >= self.config.max_replan_failures:
            control = self._safe_fallback_control(observation)
            predicted = np.tile(observation, (5, 1))
        else:
            control = self.tracker.control(observation, self._plan_step)
            predicted = self._sample_prediction(self._active_plan.state_trajectory)
            self._plan_step += 1

        self._last_control = control
        return control, predicted

    def _plan_from_state(
        self,
        initial_state: np.ndarray,
        warm_start: Optional[TrajectoryOptimizationResult],
    ) -> None:
        result = self.optimizer.solve(
            initial_state=initial_state,
            reference_state=self._target_state,
            warm_start=warm_start,
        )
        if result.success:
            self._active_plan = result
            self.tracker.update_schedule(
                result.state_trajectory, result.control_trajectory, result.time_grid
            )
            self._plan_step = 0
            self._replan_failures = 0
        else:
            self._replan_failures += 1
            print(f"[PSCPolicy] Trajectory solve failed: {result.message}")

    def _sample_prediction(self, trajectory: np.ndarray, samples: int = 5) -> np.ndarray:
        step = max(trajectory.shape[0] // samples, 1)
        return trajectory[::step][:samples]

    def _safe_fallback_control(self, observation: np.ndarray) -> np.ndarray:
        """
        Emits a conservative control command (close to hover) if no plan is
        available.  This keeps the simulator running even if SLSQP fails.
        """
        hover = np.array([0.6, 0.0, 0.0, 0.0, 0.0])
        lower, upper = self.dynamics.control_bounds
        candidate = self._last_control if np.linalg.norm(self._last_control) > 0 else hover
        return np.clip(candidate, lower, upper)
