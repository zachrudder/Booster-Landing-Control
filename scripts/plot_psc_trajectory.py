#!/usr/bin/env python3
"""
Standalone PSC trajectory visualizer.

Generates a pseudo-spectral trajectory from an initial state sampled from
SimRocketEnv (without launching the full simulation) and plots the 3D path
plus thrust vectors.
"""

from __future__ import annotations

import numpy as np

from psc.trajectory import compute_psc_trajectory
from psc.plotting import plot_trajectory
from simrocketenv import SimRocketEnv


def sample_initial_state(seed: int = 14) -> np.ndarray:
    env = SimRocketEnv(interactive=False)
    state, _ = env.reset(seed=seed)
    return np.array(state)


def main() -> None:
    x0 = sample_initial_state()
    ts, xs, us = compute_psc_trajectory(x0, Tf=3.0, N=40)
    plot_trajectory(ts, xs, us)


if __name__ == "__main__":
    main()
