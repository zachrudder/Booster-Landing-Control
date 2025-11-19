"""
Extract rocket dynamics from the existing Acados model and return a clean CasADi function f(x,u).
"""

import casadi as ca
from mpc.rocket_model import export_rocket_ode_model

def get_dynamics_function():
    model = export_rocket_ode_model()

    x = model.x
    u = model.u
    f_expl = model.f_expl_expr

    # Wrap the CasADi expression into a callable CasADi Function
    f = ca.Function("f", [x, u], [f_expl])

    nx = x.size()[0]
    nu = u.size()[0]

    return f, nx, nu
