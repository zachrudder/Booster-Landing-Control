# pscpolicy.py
#
# Pseudo-Spectral Collocation (PSC) trajectory optimizer for the rocket. LQR 
# controller to track the trajectory.
#
#   - We approximate x(t), u(t) with global polynomials on [-1, 1]
#   - Use Chebyshev–Lobatto nodes as collocation points
#   - Enforce dynamics via a differentiation matrix D
#   - Build an NLP: min J(x,u) subject to collocation + boundary constraints

import numpy as np
import time
from casadi import MX, vertcat, Function, nlpsol, jacobian

from basecontrol import BaseControl
from mpc.rocket_model import export_rocket_ode_model


def chebyshev_lobatto_nodes_and_D(N: int):
    """
    Compute Chebyshev-Gauss-Lobatto nodes and differentiation matrix.

    N: polynomial degree, gives N+1 collocation nodes.
       Indices: i = 0,...,N

    Returns:
      tau: (N+1,) array of nodes in [-1, 1]
      D:   (N+1, N+1) differentiation matrix such that
           [d/dτ x(τ_i)] ≈ sum_j D[i,j] * x(τ_j)

    Math:
      - Nodes: τ_j = cos(pi * j / N), j = 0..N
      - Differentiation matrix entries (for x_j = τ_j):
          c_j = 2 if j=0 or j=N else 1
          For i ≠ j:
            D_ij = (c_i / c_j) * (-1)^(i+j) / (x_i - x_j)
          For diagonal:
            D_ii = -x_i / (2*(1 - x_i^2)), i=1..N-1
            D_00 = (2*N^2 + 1)/6
            D_NN = -D_00
    """
    # 1) Collocation nodes τ_j in [-1,1]
    k = np.arange(0, N + 1)
    tau = np.cos(np.pi * k / N)  # shape: (N+1,)

    # 2) Differentiation matrix D
    x = tau.copy()
    D = np.zeros((N + 1, N + 1))
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0

    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j)) / (x[i] - x[j])

    # Diagonal entries
    for i in range(1, N):
        D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
    D[0, 0] = (2.0 * N ** 2 + 1.0) / 6.0
    D[N, N] = -D[0, 0]

    return tau, D


def trapezoidal_weights_on_nodes(tau: np.ndarray):
    """
    Very simple quadrature weights on arbitrary nodes using the
    trapezoidal rule in τ-space.

    tau: (N+1,) nodes in [-1,1] (not necessarily uniform, here Chebyshev)

    Returns:
      w: (N+1,) weights such that
         ∫_{-1}^1 g(τ) dτ ≈ sum_i w[i] * g(τ_i)

    NOTE: This is NOT spectrally accurate like true LGL/Clenshaw-Curtis
    weights, but it’s simple and works as a first implementation.
    You can later replace this with more accurate quadrature if desired.
    """
    Np1 = tau.shape[0]
    w = np.zeros(Np1)
    # Sort indices just in case (Chebyshev already monotonic in k).
    # Here we assume tau is in descending order (cos), but differences
    # handle this correctly.
    for i in range(Np1):
        if i == 0:
            # first node: half-interval to next
            w[i] = 0.5 * (tau[0] - tau[1])
        elif i == Np1 - 1:
            # last node: half-interval from previous
            w[i] = 0.5 * (tau[Np1 - 2] - tau[Np1 - 1])
        else:
            # middle nodes: half from left + half from right
            w[i] = 0.5 * (tau[i - 1] - tau[i + 1])
    # Absolute value to keep positive orientation (integral from -1 to 1)
    return np.abs(w)


class PSCTVLQRPolicy(BaseControl):
    """
    PSC trajectory policy (no LQR yet).

    __init__:
      - Builds a PSC optimal control problem using the rocket dynamics
      - Solves for a single open-loop trajectory x(τ_i), u(τ_i)

    next(observation):
      - Returns u at the current time index from the precomputed trajectory.
      - For now, ignores 'observation' (pure open-loop execution).
    """

    def __init__(self, initial_state, time_horizon=20.0, N_nodes=50, hover=False, use_tvlqr=False):
        super().__init__()

        # flag to turn on LQR
        self.use_tvlqr = use_tvlqr

        # keep correct time
        self.ctrl_dt = 1.0 / 120.0  # must match CTRL_DT_SEC in rocket_craft
        self.t_elapsed = 0.0

        # -------------
        # Problem setup
        # -------------
        self.model = export_rocket_ode_model()
        self.nx = int(self.model.x.size()[0])  # number of states
        self.nu = int(self.model.u.size()[0])  # number of controls (u)
        self.Tf = float(time_horizon)          # final time [s] (fixed for now)

        # Number of collocation intervals:
        #   N_nodes is the polynomial degree N; there are N+1 nodes.
        self.N = int(N_nodes)
        self.Np1 = self.N + 1

        # ---------------------------------
        # PSC ingredients: nodes, D, quad w
        # ---------------------------------
        # tau: (N+1,) collocation nodes in [-1,1]
        # D:   (N+1, N+1) differentiation matrix in τ
        self.tau, self.D = chebyshev_lobatto_nodes_and_D(self.N)

        # reverse so tau is [-1, 1] and D is increasing
        self.tau = self.tau[::-1]
        self.D   = self.D[::-1, ::-1]


        # quadrature weights for ∫_{-1}^1 g(τ) dτ
        self.w = trapezoidal_weights_on_nodes(self.tau)

        # map τ in [-1,1] to real time t ∈ [0, Tf]
        # t = (Tf/2)*(τ + 1)
        self.t_grid = (self.Tf / 2.0) * (self.tau + 1.0)  # shape: (N+1,)

        # ----------------------------------------------
        # Build CasADi function for the rocket dynamics:
        #   ẋ = f(x,u)
        # ----------------------------------------------
        x_sym = self.model.x   # (nx,1) SX
        u_sym = self.model.u   # (nu,1) SX
        f_expl = self.model.f_expl_expr  # (nx,1) SX

        # CasADi Function: f(x,u) -> xdot
        self.f_fun = Function('f_fun', [x_sym, u_sym], [f_expl])

        # ----------------------------------------------------------
        # Linearization: A = ∂f/∂x, B = ∂f/∂u  (continuous-time)
        # ----------------------------------------------------------
        A_sym = jacobian(f_expl, x_sym)   # (nx, nx)
        B_sym = jacobian(f_expl, u_sym)   # (nx, nu)
        self.A_fun = Function('A_fun', [x_sym, u_sym], [A_sym])
        self.B_fun = Function('B_fun', [x_sym, u_sym], [B_sym])


        # ---------------------------------
        # Reference / setpoint (like MPC)
        # ---------------------------------
        # We'll use the same style as in MPCPolicy:
        # - q0 = 1 (upright orientation)
        # - altitude (pos_z) = 2.42
        # You can tweak this for landing at 0, etc.
        self.x0 = np.asarray(initial_state).flatten()
        if not hover:
            x_ref = np.zeros(self.nx)
            x_ref[0] = 1.0   # q0
            x_ref[9] = 0.5  # altitude index in state (see rocket_model)
            self.x_ref = x_ref
        else:
            # Hover test
            self.x_ref = self.x0.copy()
            self.x_ref[0] = 1.0 

        # ------------------------------------------------------------------
        # Debug: print initial state and reference / goal state for PSC
        # ------------------------------------------------------------------
        print("\n[PSC] Initial state from ENV (x0):")
        print("  q      =", self.x0[0:4])
        print("  omega  =", self.x0[4:7])
        print("  pos    =", self.x0[7:10])
        print("  vel    =", self.x0[10:13])
        print("  thrust =", self.x0[13])
        print("  t_alpha, t_beta =", self.x0[14], self.x0[15])

        print("\n[PSC] Reference / goal state (x_ref):")
        print("  q_ref      =", self.x_ref[0:4])
        print("  omega_ref  =", self.x_ref[4:7])
        print("  pos_ref    =", self.x_ref[7:10])
        print("  vel_ref    =", self.x_ref[10:13])
        print("  thrust_ref =", self.x_ref[13] if self.x_ref.shape[0] > 13 else None)
        print("  t_alpha_ref, t_beta_ref =",
              self.x_ref[14] if self.x_ref.shape[0] > 14 else None,
              self.x_ref[15] if self.x_ref.shape[0] > 15 else None)
        print("-----------------------------------------------------\n")


        # For controls, we'll just encourage them to stay small around 0.
        if not hover:
            self.u_ref = np.zeros(self.nu)
            # self.R = np.diag(np.ones(self.nu) * 10.0)         # (nu,nu) smaller than MPC for now
        else:
            # approximate hover thrust fraction
            THRUST_MAX_N = 1800.0   # same as in rocket_model
            MASS_KG = 91.0          # same as in rocket_model
            GRAVITY = 9.81

            u_hover = (MASS_KG * GRAVITY) / THRUST_MAX_N   # ≈ 0.49

            self.u_ref = np.zeros(self.nu)
            self.u_ref[0] = u_hover    # we want thrust around hover

        # self.R = np.diag([10.0, 500.0, 500.0, 1000.0, 1000.0])
        self.R = np.diag([90.0, 750.0, 750.0, 100.0, 100.0])

        # Cost weights (same as MPC for state, simple R for control)
        # self.Q = np.diag(self.model.weight_diag)          # (nx,nx)

        self.Q = np.eye(self.nx) * 1e-6

        self.Q[0, 0] = 6.0              # quaternion
        self.Q[1, 1] = 6.0
        self.Q[2, 2] = 6.0
        self.Q[3, 3] = 6.0

        self.Q[4, 4] = 80.1             # angular X
        self.Q[5, 5] = 80.1             # angular Y
        self.Q[6, 6] = 0.5              # angular Z

        self.Q[7, 7] = 1.0            # pos E
        self.Q[8, 8] = 1.0            # pos N
        self.Q[9, 9] = 3.0            # pos U

        self.Q[10, 10] = 5.1          # vel E
        self.Q[11, 11] = 5.1          # vel N
        self.Q[12, 12] = 40.0          # vel U

        # self.Q[13, 13] = 0.0            # thrust
        # self.Q[14, 14] = 10.0           # thrust alpha
        # self.Q[15, 15] = 10.0           # thrust beta
        


        # Build the PSC NLP in CasADi
        self._build_psc_nlp(initial_state)

        # Solve NLP (offline)
        self._solve_nlp()

        # Internal index for playback of open-loop trajectory
        self.current_step = 0

        # build TVLQR around the trajectory
        self.K_seq = None
        if self.use_tvlqr:
            self._build_tvlqr()


    def get_name(self):
        return "PSC+TVLQR" if self.use_tvlqr else "PSC"

    def _build_psc_nlp(self, initial_state):
        """
        Construct the nonlinear program (NLP) for PSC.

        Decision variables:
          X: (nx, N+1)   states at each node
          U: (nu, N+1)   controls at each node

        Then flatten into one big vector z for CasADi:
          z = [X(:); U(:)]  with length nz = nx*(N+1) + nu*(N+1)
        """
        nx = self.nx
        nu = self.nu
        Np1 = self.Np1
        Tf = self.Tf

        # CasADi decision variables:
        # X[i] = state at node i in τ, shape: (nx, N+1)
        # U[i] = control at node i, shape: (nu, N+1)
        X = MX.sym('X', nx, Np1)
        U = MX.sym('U', nu, Np1)

        # Flatten into a single vector z for the NLP solver:
        # We use column-major ordering: first all X, then all U.
        z = vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        # Constraint vector g will contain:
        #   - Dynamics collocation constraints at each node     -> (nx*(N+1))
        #   - Initial condition: X[:,0] == initial_state       -> (nx)
        #   - Final condition:   X[:,N] == x_ref                -> (nx)
        g_list = []

        # 1) Dynamics collocation constraints
        #
        # For each node i, we enforce:
        #
        #   sum_j D[i,j] * X_j  - (Tf/2) * f(X_i, U_i) = 0
        #
        # where:
        #   D:   (N+1, N+1) differentiation matrix in τ
        #   f:   ẋ = f(x,u) in real time t
        #   Tf:  final time (scaling ∂x/∂τ to ∂x/∂t)
        #
        for i in range(Np1):
            # X_i: (nx,1) state at node i
            # U_i: (nu,1) control at node i
            X_i = X[:, i]
            U_i = U[:, i]

            # Polynomial derivative at node i: dX/dτ(τ_i)
            #   dX/dτ(τ_i) = sum_j D[i,j] * X_j
            dX_dtau_i = 0
            for j in range(Np1):
                dX_dtau_i = dX_dtau_i + self.D[i, j] * X[:, j]

            # Dynamics in real time: ẋ = f(X_i, U_i)
            f_i = self.f_fun(X_i, U_i)  # (nx,1)

            # Time scaling: dX/dτ = (Tf/2) * f(X(τ), U(τ))
            dyn_constr_i = dX_dtau_i - (Tf / 2.0) * f_i

            # Add nx constraints for node i
            g_list.append(dyn_constr_i)

        # 2) Initial condition constraints: X[:,0] = initial_state
        #    -> X[:,0] - x0 = 0
        x0 = np.asarray(initial_state).flatten()
        assert x0.shape[0] == nx, "Initial state dimension mismatch"
        g_init = X[:, 0] - x0
        g_list.append(g_init)

        # 3) Final condition constraints: X[:,N] = x_ref
        #    -> X[:,N] - x_ref = 0
        # g_final = X[:, self.N] - self.x_ref
        # g_list.append(g_final)

        # Stack all constraints into one vector g
        g = vertcat(*g_list)

        # All of these constraints are equalities, so lbg = ubg = 0
        n_g = int(g.size()[0])
        self.lbg = np.zeros(n_g)
        self.ubg = np.zeros(n_g)

        # -------------------------
        # Objective (cost function)
        # -------------------------
        #
        # We approximate the integral:
        #
        #   J = ∫_0^Tf L(x(t), u(t)) dt
        #     = (Tf/2) ∫_{-1}^1 L(x(τ), u(τ)) dτ
        #     ≈ (Tf/2) * sum_i w[i] * L(X_i, U_i)
        #
        # Where L(x,u) = (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)
        #
        J = 0
        for i in range(Np1):
            X_i = X[:, i]
            U_i = U[:, i]

            # Deviation from reference
            dx = X_i - self.x_ref
            du = U_i - self.u_ref

            # Quadratic running cost at node i
            Li = dx.T @ self.Q @ dx + du.T @ self.R @ du  # scalar MX

            # Accumulate with quadrature weight w[i]
            J = J + self.w[i] * Li

        # Scale by (Tf/2) due to time mapping
        J = (self.Tf / 2.0) * J

        # --- ADD: terminal cost on altitude and vertical speed ---
        # Final node index
        # iN = Np1 - 1
        # X_final = X[:, iN]

        # # State indices:
        # #   pos_U (altitude) = 9
        # #   vel_U (vertical velocity) = 12
        # z_final  = X_final[9]
        # vz_final = X_final[12]

        # # You already set x_ref[9] = 2.45 in __init__
        # z_target  = self.x_ref[9]
        # vz_target = 0.0

        # # Weights for terminal cost
        # w_z_term  = 50.0    # cost on (z(T) - z_target)
        # w_vz_term = 200.0   # strong cost on vertical speed at final time

        # J_terminal = w_z_term  * (z_final - z_target)**2 \
        #            + w_vz_term * (vz_final - vz_target)**2

        # Add terminal cost (no scaling by Tf/2 — it's a pure endpoint cost)
        # J = J + J_terminal


        # ----------------------
        # Variable bounds (lbx, ubx)
        # ----------------------
        #
        # We only bound the control inputs U based on the MPC constraints:
        #   u[0] (thrust) in [0.20, 1.00]
        #   u[1..4] in [-1.0, 1.0]
        #
        # The state X is left unbounded here. You can add state bounds later.
        #
        nz = int(z.size()[0])
        lbz = -np.inf * np.ones(nz)
        ubz = +np.inf * np.ones(nz)

        # Indices in z:
        #   First nx*(N+1) entries: X(:)
        #   Next  nu*(N+1) entries: U(:)
        X_size = nx * self.Np1
        U_size = nu * self.Np1

        # For U, index offset in z:
        u_offset = X_size  # where U starts in z

        # Bounds per control component
        # u_min = np.array([0.20, -1.0, -1.0, -1.0, -1.0])
        # u_max = np.array([1.00,  1.0,  1.0,  1.0,  1.0])

        # Disable attitude thrusters u[3] and u[4]
        u_min = np.array([0.20, -1.0, -1.0, 0.0, 0.0])
        u_max = np.array([1.00,  1.0,  1.0, 0.0, 0.0])

        # limit range of bounds
        # u_min = np.array([0.20, -0.5, -0.5, -0.5, -0.5])
        # u_max = np.array([1.00,  0.5,  0.5,  0.5,  0.5])


        for i in range(self.Np1):
            for j in range(nu):
                idx = u_offset + i * nu + j
                lbz[idx] = u_min[j]
                ubz[idx] = u_max[j]

        self.z = z
        self.g = g
        self.J = J
        self.lbx = lbz
        self.ubx = ubz

        # ----------------
        # Create NLP solver
        # ----------------
        nlp = {"x": self.z, "f": self.J, "g": self.g}

        # Use IPOPT via CasADi
        self.solver = nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "ipopt.print_level": 3,
                "ipopt.max_iter": 1000,
                "print_time": False,
            },
        )

    def _solve_nlp(self):
        """
        Solve the PSC NLP once at initialization.

        Stores:
          self.X_opt: (nx, N+1) numpy array of optimal states
          self.U_opt: (nu, N+1) numpy array of optimal controls
        """
        nz = int(self.z.size()[0])

        # Initial guess for decision variables z:
        # - X: start from linear interpolation between x0 and x_ref
        # - U: zeros within bounds (then solver refines)
        z0 = np.zeros(nz)
        x0 = self.x0

        nx = self.nx
        nu = self.nu
        Np1 = self.Np1

        X_size = nx * Np1
        u_offset = X_size

        # Build simple initial guess for X
        for i in range(Np1):
            alpha = i / self.N  # from 0 to 1
            x_guess = (1 - alpha) * x0 + alpha * self.x_ref  # trivial: all x_ref
            # If you want better: blend from initial_state to x_ref over nodes.
            z0[i * nx:(i + 1) * nx] = x_guess

        # U initial guess: all zeros (or mid-range thrust)
        for i in range(Np1):
            for j in range(nu):
                idx = u_offset + i * nu + j
                if j == 0:
                    # thrust around mid-range 0.6
                    z0[idx] = 0.6
                else:
                    z0[idx] = 0.0

        # Solve NLP
        sol = self.solver(x0=z0, lbg=self.lbg, ubg=self.ubg,
                          lbx=self.lbx, ubx=self.ubx)
        
        # get the status from the solver
        stats = self.solver.stats()
        status = stats.get("return_status", "")
        print("[PSC] IPOPT status:", status)

        # Accept only clearly successful outcomes
        ok_statuses = ["Solve_Succeeded", "Optimal Solution Found", "Solved_To_Acceptable_Level"]
        if not any(s in status for s in ok_statuses):
            print("[PSC] WARNING: NLP did NOT converge to a feasible optimum.")
            # Fallback: trivial hover-like trajectory
            self.X_opt = np.tile(self.x0.reshape(-1, 1), (1, self.Np1))
            self.U_opt = np.zeros((self.nu, self.Np1))
            self.U_opt[0, :] = 0.6   # mid thrust
            return

        z_opt = np.array(sol["x"]).flatten()

        # Extract X_opt and U_opt from z_opt
        X_flat = z_opt[:X_size]
        U_flat = z_opt[u_offset:u_offset + nu * Np1]

        self.X_opt = X_flat.reshape((self.nx, Np1), order='F')
        self.U_opt = U_flat.reshape((self.nu, Np1), order='F')

        # for plotting
        self.last_X_traj = self.X_opt
        self.last_U_traj = self.U_opt

        # Optional: print basic info
        print("[PSC] NLP solved. Final cost J =", float(sol["f"]))

        # print other useful info
        print("[PSC] First node state:")
        print("  q =", self.X_opt[0:4, 0])
        print("  omega =", self.X_opt[4:7, 0])
        print("  pos =", self.X_opt[7:10, 0])
        print("  vel =", self.X_opt[10:13, 0])
        print("  thrust =", self.X_opt[13, 0])
        print("  t_alpha, t_beta =", self.X_opt[14, 0], self.X_opt[15, 0])

        print("[PSC] Last node state:")
        print("  q =", self.X_opt[0:4, -1])
        print("  omega =", self.X_opt[4:7, -1])
        print("  pos =", self.X_opt[7:10, -1])
        print("  vel =", self.X_opt[10:13, -1])
        print("  thrust =", self.X_opt[13, -1])
        print("  t_alpha, t_beta =", self.X_opt[14, -1], self.X_opt[15, -1])

        print("[PSC] Example controls (first, mid, last):")
        print("  u0 =", self.U_opt[:, 0])
        print("  u_mid =", self.U_opt[:, self.N // 2])
        print("  uN =", self.U_opt[:, -1])

    
    def _build_tvlqr(self):
        """
        Build a discrete-time time-varying LQR (TVLQR) along the PSC trajectory.

        We linearize the continuous dynamics around each nominal (x_i, u_i):
            xdot = f(x,u)
            A_i = ∂f/∂x |_(x_i,u_i)
            B_i = ∂f/∂u |_(x_i,u_i)

        Then approximate a discrete system over step h = Tf/N:
            x_{k+1} ≈ A_d x_k + B_d u_k
            A_d = I + A_i * h
            B_d = B_i * h

        and run a backward Riccati recursion with (Q, R) to get K_k.
        """

        if not hasattr(self, "X_opt") or not hasattr(self, "U_opt"):
            print("[TVLQR] No nominal trajectory available; skipping TVLQR.")
            self.K_seq = None
            return

        nx = self.nx
        nu = self.nu
        Np1 = self.Np1   # number of nodes = N+1
        N = self.N

        # Time step used for discrete approximation in LQR
        h = self.Tf / float(self.N)

        # Stage cost matrices
        Q_tvlqr = np.eye(nx)

        Q_tvlqr[0, 0] = 10.0              # quaternion
        Q_tvlqr[1, 1] = 10.0
        Q_tvlqr[2, 2] = 10.0
        Q_tvlqr[3, 3] = 10.0

        Q_tvlqr[4, 4] = 10.0            # angular X
        Q_tvlqr[5, 5] = 10.0             # angular Y
        Q_tvlqr[6, 6] = 10.0             # angular Z

        Q_tvlqr[7, 7] = 30.0            # pos E
        Q_tvlqr[8, 8] = 30.0            # pos N
        Q_tvlqr[9, 9] = 30.0            # pos U

        Q_tvlqr[10, 10] = 30.0          # vel E
        Q_tvlqr[11, 11] = 30.0          # vel N
        Q_tvlqr[12, 12] = 30.0          # vel U

        Q_tvlqr[13, 13] = 0.0            # thrust
        Q_tvlqr[14, 14] = 0.0           # thrust alpha
        Q_tvlqr[15, 15] = 0.0           # thrust beta

        R_tvlqr = np.eye(nu) * 15000.0

        # Precompute linearizations A_i, B_i at each node
        A_seq = []
        B_seq = []
        for i in range(Np1):
            x_i = self.X_opt[:, i]
            u_i = self.U_opt[:, i]

            A_i = np.array(self.A_fun(x_i, u_i)).astype(float)
            B_i = np.array(self.B_fun(x_i, u_i)).astype(float)

            A_seq.append(A_i)
            B_seq.append(B_i)

        # Backward Riccati recursion
        K_seq = [np.zeros((nu, nx)) for _ in range(Np1)]

        # Terminal cost: you can tweak this (e.g. 10*Q)
        P = Q_tvlqr.copy()

        for i in reversed(range(N)):
            A = A_seq[i]
            B = B_seq[i]

            # Discretize
            A_d = np.eye(nx) + A * h
            B_d = B * h

            # Riccati step
            S = R_tvlqr + B_d.T @ P @ B_d

            # K_i: (nu, nx)
            K_i = np.linalg.solve(S, B_d.T @ P @ A_d)
            # K_i[3:, :] = 0.0    # disable attitude thrusters
            K_seq[i] = K_i

            # P update
            P = Q_tvlqr + A_d.T @ (P - P @ B_d @ np.linalg.solve(S, B_d.T @ P)) @ A_d

        # For the last node, just reuse K_{N-1}
        K_seq[N] = K_seq[N - 1].copy()

        self.K_seq = K_seq
        print(f"[TVLQR] Built time-varying LQR gains for {Np1} nodes.")


    def next(self, observation):
        """
        Return control from the PSC trajectory, with optional TVLQR tracking.

        Parameters
        ----------
        observation : np.ndarray
            Current state (16,) from the environment.

        Returns
        -------
        u : np.ndarray  shape = (nu,)
            Control at current time step.
        predictedX : np.ndarray shape = (5, nx)
            Simple preview of the next 5 nominal states on the trajectory.
        """

        # Advance time
        # self.t_elapsed += self.ctrl_dt
        # if self.t_elapsed > self.Tf:
        #     self.t_elapsed = self.Tf

        # only advance the time when sim actually updates
        # if not hasattr(self, "_last_x_obs"):
        #     self._last_x_obs = np.asarray(observation).copy()

        # if np.linalg.norm(np.asarray(observation) - self._last_x_obs) > 1e-6:
        #     # sim actually advanced
        #     self.t_elapsed += self.ctrl_dt
        #     self._last_x_obs = np.asarray(observation).copy()

        # another try for fixing time
        now = time.time()
        if not hasattr(self, "_last_call"):
            self._last_call = now
            dt = 0.0
        else:
            dt = now - self._last_call
            self._last_call = now

        # Clamp dt so we don't jump far along the trajectory after a stall
        # (sim thread also ignores huge dt_sec > 0.1)
        if dt > 0.1:
            dt = 0.0

        # advance PSC time using ACTUAL dt
        self.t_elapsed += dt

        # Don't run past the end of the PSC horizon
        if self.t_elapsed > self.Tf:
            self.t_elapsed = self.Tf


        # Map current time to a node index
        # alpha = self.t_elapsed / self.Tf
        # i_float = alpha * self.N
        # idx = int(np.clip(np.round(i_float), 0, self.N))

        # alternate method
        idx = np.searchsorted(self.t_grid, self.t_elapsed, side="left")
        idx = int(np.clip(idx, 0, self.N))

        # debug
        # print("t_grid[0], t_grid[-1] =", self.t_grid[0], self.t_grid[-1])

        # Nominal control and state from PSC
        u_nom = np.array(self.U_opt[:, idx]).flatten()
        x_nom = np.array(self.X_opt[:, idx]).flatten()

        # Default: open-loop PSC
        u = u_nom.copy()

        # If TVLQR is enabled, use it as a tracking law
        if self.use_tvlqr and (self.K_seq is not None):
            x = np.asarray(observation).flatten()
            K = self.K_seq[idx]          # (nu, nx)
            x_err = x - x_nom            # state deviation

            delta_u = -K @ x_err         # (nu,)
            u = u_nom + delta_u
            
            # Debug: first few timesteps
            # if self.t_elapsed < 0.05:
            #     print("[STATE DEBUG] idx", idx)
            #     print("  x_obs  =", x)
            #     print("  x_nom  =", x_nom)
            #     print("  x_err  =", x_err)

            if self.t_elapsed < 0.1:
                print("[TVLQR] ACTIVE at idx", idx)
                print("  ||x_err|| =", np.linalg.norm(x_err))
                print("  u_nom     =", u_nom)
                print("  delta_u   =", delta_u)
                print("  u_total   =", u)

        else:
            if self.t_elapsed < 0.1:
                print("[TVLQR] NOT active. use_tvlqr=",
                    getattr(self, "use_tvlqr", None),
                    "K_seq is None" if not hasattr(self, "K_seq") or self.K_seq is None else "K_seq set")

        # Safety: enforce the same input bounds as the NLP
        # Main thrust in [0.20, 1.00], others in [-1, 1]
        u[0] = np.clip(u[0], 0.20, 1.0)
        u[1:] = np.clip(u[1:], -1.0, 1.0)

        # Build preview of nominal states (unchanged)
        NUM_PRED_EPOCHS = 5
        predictedX = np.zeros((NUM_PRED_EPOCHS, self.nx))
        for k in range(NUM_PRED_EPOCHS):
            j = int(np.clip(idx + k, 0, self.N))
            predictedX[k, :] = np.array(self.X_opt[:, j]).flatten()

        return u, predictedX

    
    def debug_plot_collocation_nodes(self):
        """
        Visualize the Chebyshev–Lobatto collocation nodes and optional
        positions along the PSC trajectory (if solved).

        - self.tau: nodes in [-1, 1]
        - self.t_grid: corresponding physical times in [0, Tf]
        - self.X_opt (if available): states at those nodes
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 1) Plot τ nodes (Chebyshev–Lobatto on [-1, 1])
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        ax = axs[0]
        ax.plot(self.tau, 'o-')
        ax.set_xlabel("Node index i")
        ax.set_ylabel(r"$\tau_i$ (dimensionless)")
        ax.set_title("Chebyshev–Lobatto nodes in $[-1, 1]$")
        ax.grid(True)

        # 2) Plot corresponding time grid t in [0, Tf]
        ax = axs[1]
        ax.plot(self.t_grid, 'o-')
        ax.set_xlabel("Node index i")
        ax.set_ylabel("t_i [s]")
        ax.set_title(f"Collocation nodes in time [0, {self.Tf:.2f}]")
        ax.grid(True)

        plt.tight_layout()
        plt.show()

        # 3) Optional: if we have a solved trajectory, plot altitude vs time
        if hasattr(self, "X_opt"):
            pos_E = self.X_opt[7, :]   # East
            pos_N = self.X_opt[8, :]   # North
            pos_U = self.X_opt[9, :]   # Up

            fig = plt.figure(figsize=(6, 5))
            ax3 = fig.add_subplot(111, projection="3d")
            sc = ax3.scatter(pos_E, pos_N, pos_U, c=self.t_grid,
                            cmap="viridis", s=40)

            ax3.set_xlabel("East (m)")
            ax3.set_ylabel("North (m)")
            ax3.set_zlabel("Up (m)")
            ax3.set_title("Trajectory at collocation nodes\n(color = time)")
            fig.colorbar(sc, ax=ax3, label="t [s]")

            plt.tight_layout()
            plt.show()


