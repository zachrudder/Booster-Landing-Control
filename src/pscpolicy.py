# src/psc_tvlqr_policy.py

import numpy as np

from basecontrol import BaseControl
from psc.trajectory import compute_psc_trajectory
from psc.tvlqr import compute_tvlqr_gains
from psc.plotting import plot_trajectory
from mpc.rocket_model import export_rocket_ode_model


class PSCTVLQRPolicy(BaseControl):
    """
    Pseudo-spectral trajectory + time-varying LQR tracking policy.

    Usage in rocketcraft.py: just replace
        policy = MPCPolicy(initial_state)
    with
        policy = PSCTVLQRPolicy(initial_state)
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        Tf: float = 3.0,
        N: int = 40,
        replan_interval_steps: int = 80,
        show_plot: bool = True,
    ):
        super().__init__()

        self.initial_state = np.array(initial_state).copy()
        self.Tf = Tf
        self.N = N
        self.replan_interval = max(1, replan_interval_steps)
        self.show_plot = show_plot

        # Get dimensions / cost weights from existing model
        self.model = export_rocket_ode_model()
        self.nx = int(self.model.x.size()[0])
        self.nu = int(self.model.u.size()[0])

        self._plan_from_state(self.initial_state, plot=show_plot)
        self.replan_counter = 0

    def get_name(self) -> str:
        return "PSC + TVLQR"

    def _current_index(self) -> int:
        k = self.step_counter
        if k > self.max_index - 1:
            k = self.max_index - 1
        return max(k, 0)

    def next(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.array(observation).reshape(-1)
        self.step_counter += 1
        self.replan_counter += 1

        if self.replan_counter >= self.replan_interval:
            self._plan_from_state(x, plot=self.show_plot)
            self.replan_counter = 0

        k = self._current_index()
        x_ref = self.xs[k, :]
        u_ref = self.us[k, :]

        K = self.Ks[k]  # (nu, nx)

        dx = x - x_ref
        u = u_ref + K @ dx  # (nu,)

        # Clamp to same bounds as MPC & PSC OCP
        lbu = np.array([0.20, -1.0, -1.0, -1.0, -1.0])
        ubu = np.array([1.00,  1.0,  1.0,  1.0,  1.0])
        u = np.clip(u, lbu, ubu)

        # predicted segment like MPC
        NUM_PRED_EPOCHS = 5
        step_size = max(1, self.max_index // NUM_PRED_EPOCHS)
        predictedX = np.zeros((NUM_PRED_EPOCHS, self.nx))
        for i in range(NUM_PRED_EPOCHS):
            idx = min(self.max_index, k + i * step_size)
            predictedX[i, :] = self.xs[idx, :]

        return u, predictedX

    def _plan_from_state(self, state: np.ndarray, plot: bool = False) -> None:
        ts, xs, us = compute_psc_trajectory(state, Tf=self.Tf, N=self.N)
        self.ts = ts
        self.xs = xs
        self.us = us

        weight_diag = np.array(self.model.weight_diag).reshape(-1)
        Q = np.diag(weight_diag)
        Qf = np.diag(weight_diag)
        R = np.diag(np.ones(self.nu) * 100.0)

        self.Ks = compute_tvlqr_gains(ts, xs, us, Q=Q, R=R, Qf=Qf)

        self.step_counter = 0
        self.max_index = len(ts) - 1

        if plot:
            plot_trajectory(ts, xs, us)
