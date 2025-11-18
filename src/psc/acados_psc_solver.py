"""
Wrapper that loads the pre-generated acados PSC solver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from acados_template import AcadosOcpSolver

from .acados_psc_builder import build_psc_ocp
from .psc_layout import PSCDecisionLayout


class PSCAcadosSolver:
    def __init__(
        self,
        mesh,
        horizon_sec: float,
        state_cost_diag: np.ndarray,
        control_cost_diag: np.ndarray,
        state_lower: np.ndarray,
        state_upper: np.ndarray,
        control_bounds: Tuple[np.ndarray, np.ndarray],
        json_file: str = "acados_ocp_psc_collocation.json",
    ):
        if not Path(json_file).exists():
            raise FileNotFoundError(
                f"{json_file} not found. Run scripts/build_psc_solver.py to generate it."
            )

        ocp, layout = build_psc_ocp(
            mesh=mesh,
            horizon_sec=horizon_sec,
            state_cost_diag=state_cost_diag,
            control_cost_diag=control_cost_diag,
            state_lower=state_lower,
            state_upper=state_upper,
            control_bounds=control_bounds,
        )
        self.mesh = mesh
        self.layout = layout
        self.solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
            build=False,
            generate=False,
            verbose=False,
        )

    def solve(
        self,
        x0: np.ndarray,
        xf: np.ndarray,
        reference_trajectory: np.ndarray,
        initial_guess: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        params = np.concatenate([x0, xf])
        self.solver.set(0, "p", params)
        self.solver.set(0, "yref", reference_trajectory)
        self.solver.set(0, "x", initial_guess)
        status = self.solver.solve()
        solution = self.solver.get(0, "x")
        return solution, status
