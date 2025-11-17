"""
Building blocks for pseudo-spectral collocation based trajectory optimization.

The package exposes utilities to discretize a horizon with Legendre-Gauss-Lobatto
nodes, a light-weight rocket dynamics model that mirrors the MPC version, a
nonlinear program transcription, and an LQR tracker that can be used to follow
the optimized trajectory in closed loop.
"""

from .discretization import CollocationMesh, legendre_gauss_lobatto_mesh
from .rocket_dynamics import RocketDynamics, RocketPhysicalParams
from .trajectory_optimizer import (
    PSCTrajectoryConfig,
    TrajectoryOptimizationResult,
    PSCTrajectoryOptimizer,
)
from .lqr_tracker import LQRTracker, LQRTrackerConfig

__all__ = [
    "CollocationMesh",
    "legendre_gauss_lobatto_mesh",
    "RocketDynamics",
    "RocketPhysicalParams",
    "PSCTrajectoryConfig",
    "TrajectoryOptimizationResult",
    "PSCTrajectoryOptimizer",
    "LQRTracker",
    "LQRTrackerConfig",
]
