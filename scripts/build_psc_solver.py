#!/usr/bin/env python3
"""
Pre-generates the acados solver for the PSC controller.

Run once (with the acados environment sourced) to emit
acados_ocp_psc_collocation.json and the accompanying C code under
c_generated_code/.  Subsequent runs of the simulator can then load the solver
without compiling at startup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from acados_template import AcadosOcpSolver

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psc.discretization import legendre_gauss_lobatto_mesh
from psc.rocket_dynamics import RocketDynamics
from psc.trajectory_optimizer import PSCTrajectoryConfig
from psc.acados_psc_builder import build_psc_ocp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PSC acados solver.")
    parser.add_argument(
        "--output-json",
        default="acados_ocp_psc_collocation.json",
        help="Target JSON filename for the generated solver.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PSCTrajectoryConfig()
    mesh = legendre_gauss_lobatto_mesh(config.polynomial_order).scaled(
        0.0, config.horizon_sec
    )
    dynamics = RocketDynamics()
    ocp, layout = build_psc_ocp(
        mesh=mesh,
        horizon_sec=config.horizon_sec,
        state_cost_diag=config.state_cost_diag,
        control_cost_diag=config.control_cost_diag,
        state_lower=config.state_lower,
        state_upper=config.state_upper,
        control_bounds=dynamics.control_bounds,
    )

    output = Path(args.output_json)
    AcadosOcpSolver(ocp, json_file=str(output))
    print(f"Generated PSC solver artifacts at {output} and c_generated_code/.")


if __name__ == "__main__":
    main()
