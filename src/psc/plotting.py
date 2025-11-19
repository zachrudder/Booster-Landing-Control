"""3D visualization utilities for PSC trajectories."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot_trajectory(ts: np.ndarray, xs: np.ndarray, us: np.ndarray) -> None:
    """Plot position and thrust vectors along the PSC trajectory."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ts = np.asarray(ts)
    xs = np.asarray(xs)
    us = np.asarray(us)
    t_uniform = np.linspace(ts[0], ts[-1], 200)

    east = interp1d(ts, xs[:, 7], kind="cubic")(t_uniform)
    north = interp1d(ts, xs[:, 8], kind="cubic")(t_uniform)
    up = interp1d(ts, xs[:, 9], kind="cubic")(t_uniform)
    east -= east[-1]
    north -= north[-1]
    up -= up[-1]
    ax.plot(east, north, up, label="Trajectory")

    step = max(1, len(ts) // 30)
    # Thrust vectors disabled for clarity; re-enable if needed.

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.legend()
    ax.set_title("PSC Trajectory")
    plt.tight_layout()
    plt.show()
