# pscpolicy.py
#
# Created by Cody Hopkins, Zach Rudder, and Finn Maniscalco
#
# Pseudo-Spectral Collocation (PSC) trajectory optimizer for the rocket. LQR 
# controller to track the trajectory.

import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple

from basecontrol import BaseControl
from casadi import MX, Function, jacobian, nlpsol, vertcat
from psc.rocket_model import export_rocket_ode_model
from mpl_toolkits.mplot3d import Axes3D


def chebyshev_lobatto_nodes_and_D(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Chebyshev-Gauss-Lobatto nodes and differentiation matrix.

    Args: 
        N: polynomial degree

    Returns:
      tau: (N+1,) array of nodes in [-1, 1]
      D:   (N+1, N+1) differentiation matrix

    """
    # Collocation nodes
    k = np.arange(0, N + 1)
    tau = np.cos(np.pi * k / N)

    # Differentiation matrix
    x = tau.copy()
    D = np.zeros((N + 1, N + 1))
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0

    # Off-diagonal terms
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j)) / (x[i] - x[j])
    
    # Diagonal terms
    for i in range(1, N):
        D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
    D[0, 0] = (2.0 * N ** 2 + 1.0) / 6.0
    D[N, N] = -D[0, 0]

    return tau, D


def clenshaw_curtis_weights(N: int) -> np.ndarray:
    """
    Compute Clenshaw-Curtis quadrature weights for CGL nodes.
    Spectrally accurate quadrature on Chebyshev-Lobatto grid.

    Args: 
        N: polynomial degree

    Returns:
      w: (N+1,) weights that integrate polynomials of degree <= N exactly
      
    """
    theta = np.pi * np.arange(N+1) / N
    w = np.zeros(N+1)

    J = np.arange(0, N+1, 2)
    w = np.zeros(N+1)

    for k in range(N+1):
        theta_k = theta[k]
        s = 0.0
        for j in J:
            if j == 0:
                s += 1.0
            else:
                s += (2 / (1 - j**2)) * np.cos(j * theta_k)
        w[k] = (2 / N) * s

    return w


class PSCPolicy(BaseControl):
    """
    PSC trajectory policy with optional TVLQR tracking.

    Builds a pseudo-spectral collocation plan using the rocket dynamics and
    optionally linearizes the trajectory to compute time-varying LQR gains for
    feedback tracking during execution.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        time_horizon: float = 30.0,
        N_nodes: int = 30,
        use_tvlqr: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.use_tvlqr = use_tvlqr
        self.debug = debug

        # Keep correct time
        self.ctrl_dt = 1.0 / 120.0  # match CTRL_DT_SEC in rocket_craft
        self.t_elapsed = 0.0

        # Problem setup
        self.model = export_rocket_ode_model()
        self.nx = int(self.model.x.size()[0])  # number of states
        self.nu = int(self.model.u.size()[0])  # number of control inputs
        self.Tf = float(time_horizon)          # final time in seconds

        # N knot points, N+1 intervals
        self.N = int(N_nodes)
        self.Np1 = self.N + 1

        # Compute tau and D
        self.tau, self.D = chebyshev_lobatto_nodes_and_D(self.N)
        self.w = clenshaw_curtis_weights(self.N)

        # reverse so tau is [-1, 1] and D is increasing
        self.tau = self.tau[::-1]
        self.D   = self.D[::-1, ::-1]

        # Map tau to real time
        self.t_grid = (self.Tf / 2.0) * (self.tau + 1.0)

        # Build Casadi function for the rocket dynamics
        x_sym = self.model.x 
        u_sym = self.model.u 
        f_expl = self.model.f_expl_expr 

        self.f_fun = Function('f_fun', [x_sym, u_sym], [f_expl])

        # Linearize function
        A_sym = jacobian(f_expl, x_sym)
        B_sym = jacobian(f_expl, u_sym)
        self.A_fun = Function('A_fun', [x_sym, u_sym], [A_sym])
        self.B_fun = Function('B_fun', [x_sym, u_sym], [B_sym])


        # Set the reference state (goal)
        self.x0 = np.asarray(initial_state).flatten()
        x_ref = np.zeros(self.nx)
        x_ref[0] = 1.0  # upright
        x_ref[9] = 0.5  # altitude  
        self.x_ref = x_ref

        if self.debug:
            # Print initial state and reference / goal state for PSC
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


        # Set control reference
        self.u_ref = np.zeros(self.nu)
        
        # Control input cost weights
        self.R = np.diag([90.0, 750.0, 750.0, 1000.0, 1000.0])

        # State cost weights
        self.Q = np.eye(self.nx) * 1e-6

        self.Q[0, 0] = 6.0              # quaternion
        self.Q[1, 1] = 6.0
        self.Q[2, 2] = 6.0
        self.Q[3, 3] = 6.0

        self.Q[4, 4] = 80.1             # angular X
        self.Q[5, 5] = 80.1             # angular Y
        self.Q[6, 6] = 0.5              # angular Z

        self.Q[7, 7] = 1.0              # pos E
        self.Q[8, 8] = 1.0              # pos N
        self.Q[9, 9] = 3.0              # pos U

        self.Q[10, 10] = 5.1            # vel E
        self.Q[11, 11] = 5.1            # vel N
        self.Q[12, 12] = 33.0           # vel U

        # self.Q[13, 13] = 0.0           # thrust
        # self.Q[14, 14] = 0.0           # thrust alpha
        # self.Q[15, 15] = 0.0           # thrust beta

        # --- Logging for analysis ---
        self.log_t = []              # time stamps
        self.log_idx = []            # PSC node indices
        self.log_x_actual = []       # actual state from env
        self.log_x_nom = []          # nominal state from PSC
        self.log_u_total = []        # control actually sent to env
        self.log_u_nom = []          # nominal PSC control
        self.log_x_err_norm = []     # ||x - x_nom||
        self.log_delta_u_norm = []   # ||u - u_nom|| (LQR correction)
        
        self.start_time = time.time()

        # Build the PSC NLP in CasADi
        self._build_psc_nlp(initial_state)

        # Solve NLP (offline)
        self._solve_nlp()

        self.solve_time = time.time() - self.start_time

        print(f"[PSC] NLP Solve Time: {self.solve_time}")

        # Internal index for playback of open-loop trajectory
        self.current_step = 0

        # build TVLQR around the trajectory
        self.K_seq = None
        if self.use_tvlqr:
            self._build_tvlqr()


    def get_name(self) -> str:
        return "PSC+TVLQR" if self.use_tvlqr else "PSC"

    def _build_psc_nlp(self, initial_state: np.ndarray) -> None:
        """
        Construct the nonlinear program (NLP) for PSC.

        Args:
            initial_state: the initial state vector of the booster

        """
        nx = self.nx
        nu = self.nu
        Np1 = self.Np1
        Tf = self.Tf

        # Decision variables
        X = MX.sym('X', nx, Np1)
        U = MX.sym('U', nu, Np1)

        # Flatten into a single vector
        z = vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        # Constraint list
        g_list = []

        # Dynamics collocation constraints
        for i in range(Np1):
            X_i = X[:, i]
            U_i = U[:, i]

            dX_dtau_i = 0
            for j in range(Np1):
                dX_dtau_i = dX_dtau_i + self.D[i, j] * X[:, j]

            f_i = self.f_fun(X_i, U_i)
            constraint_i = dX_dtau_i - (Tf / 2.0) * f_i

            # Add nx constraints for node i
            g_list.append(constraint_i)

        # Initial condition constraints
        x0 = np.asarray(initial_state).flatten()
        assert x0.shape[0] == nx, "Initial state dimension mismatch"
        g_init = X[:, 0] - x0
        g_list.append(g_init)

        # Hard final condition constraints
        # g_final = X[:, self.N] - self.x_ref
        # g_list.append(g_final)

        # Stack constraints into vector
        g = vertcat(*g_list)

        # Constraints are equalities so lbg = ubg = 0
        n_g = int(g.size()[0])
        self.lbg = np.zeros(n_g)
        self.ubg = np.zeros(n_g)

        # Cost function
        J = 0
        for i in range(Np1):
            X_i = X[:, i]
            U_i = U[:, i]

            # Error
            dx = X_i - self.x_ref
            du = U_i - self.u_ref

            # Quadratic running cost at node i
            Li = dx.T @ self.Q @ dx + du.T @ self.R @ du 

            # Add quadrature weight at node
            J = J + self.w[i] * Li

        # Scale due to time mapping
        J = (self.Tf / 2.0) * J

        # Variable bounds
        nz = int(z.size()[0])
        lbz = -np.inf * np.ones(nz)
        ubz = +np.inf * np.ones(nz)

        X_size = nx * self.Np1
        U_size = nu * self.Np1

        # U is offset in z vector
        u_offset = X_size

        # Regular bounds per control component
        # u_min = np.array([0.20, -1.0, -1.0, -1.0, -1.0])
        # u_max = np.array([1.00,  1.0,  1.0,  1.0,  1.0])

        # Disable attitude thrusters u[3] and u[4] for trajectory
        u_min = np.array([0.20, -1.0, -1.0, 0.0, 0.0])
        u_max = np.array([1.00,  1.0,  1.0, 0.0, 0.0])

        # Set bounds on control inputs
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

        # Create NLP
        nlp = {"x": self.z, "f": self.J, "g": self.g}

        # Use IPOPT through Casadi
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

    def _solve_nlp(self) -> None:
        """
        Solve the PSC NLP once at initialization and store the optimal
        policy (states and control inputs).
        """
        nz = int(self.z.size()[0])

        # Initial guess
        z0 = np.zeros(nz)
        x0 = self.x0

        nx = self.nx
        nu = self.nu
        Np1 = self.Np1

        X_size = nx * Np1
        u_offset = X_size

        # Initial guess for X -> linear interpolation
        for i in range(Np1):
            alpha = i / self.N 
            x_guess = (1 - alpha) * x0 + alpha * self.x_ref 
            z0[i * nx:(i + 1) * nx] = x_guess

        # Initial guess for U -> zeros except thrust magintude
        for i in range(Np1):
            for j in range(nu):
                idx = u_offset + i * nu + j
                if j == 0:
                    # thrust at midpoint
                    z0[idx] = 0.6
                else:
                    z0[idx] = 0.0

        # Solve NLP
        sol = self.solver(x0=z0, lbg=self.lbg, ubg=self.ubg,
                          lbx=self.lbx, ubx=self.ubx)
        
        # Get the status from the solver
        stats = self.solver.stats()
        status = stats.get("return_status", "")
        print("[PSC] IPOPT status:", status)

        # Only accept successful outcomes
        ok_statuses = ["Solve_Succeeded", "Optimal Solution Found", "Solved_To_Acceptable_Level"]
        if not any(s in status for s in ok_statuses):
            print("[PSC] WARNING: NLP did NOT converge to a feasible optimum.")

            # Return a simple trajectory if it fails to find solution
            self.X_opt = np.tile(self.x0.reshape(-1, 1), (1, self.Np1))
            self.U_opt = np.zeros((self.nu, self.Np1))
            self.U_opt[0, :] = 0.6
            return

        # Extract X_opt and U_opt from z_opt
        z_opt = np.array(sol["x"]).flatten()
        X_flat = z_opt[:X_size]
        U_flat = z_opt[u_offset:u_offset + nu * Np1]

        self.X_opt = X_flat.reshape((self.nx, Np1), order='F')
        self.U_opt = U_flat.reshape((self.nu, Np1), order='F')

        # Store for plotting
        self.last_X_traj = self.X_opt
        self.last_U_traj = self.U_opt            

        if self.debug:
            # Print useful info
            print("[PSC] NLP solved. Final cost J =", float(sol["f"]))

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

            print("[PSC] Controls (first, mid, last):")
            print("  u0 =", self.U_opt[:, 0])
            print("  u_mid =", self.U_opt[:, self.N // 2])
            print("  uN =", self.U_opt[:, -1])

    
    def _build_tvlqr(self) -> None:
        """
        Build a discrete-time time-varying LQR (TVLQR) along the PSC trajectory.
        """

        nx = self.nx
        nu = self.nu
        Np1 = self.Np1
        N = self.N

        # Time step used for discrete approximation in LQR
        h = self.Tf / float(self.N)

        # State cost weights
        Q_tvlqr = np.eye(nx)

        Q_tvlqr[0, 0] = 100.0           # quaternion
        Q_tvlqr[1, 1] = 100.0
        Q_tvlqr[2, 2] = 100.0
        Q_tvlqr[3, 3] = 100.0

        Q_tvlqr[4, 4] = 10.0            # angular X
        Q_tvlqr[5, 5] = 10.0            # angular Y
        Q_tvlqr[6, 6] = 10.0            # angular Z

        Q_tvlqr[7, 7] = 30.0            # pos E
        Q_tvlqr[8, 8] = 30.0            # pos N
        Q_tvlqr[9, 9] = 30.0            # pos U

        Q_tvlqr[10, 10] = 50.0          # vel E
        Q_tvlqr[11, 11] = 50.0          # vel N
        Q_tvlqr[12, 12] = 50.0          # vel U

        Q_tvlqr[13, 13] = 0.0           # thrust
        Q_tvlqr[14, 14] = 0.0           # thrust alpha
        Q_tvlqr[15, 15] = 0.0           # thrust beta

        # Control input cost weights
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

        # Terminal cost
        P = Q_tvlqr.copy()

        # Backward Riccati recursion
        K_seq = [np.zeros((nu, nx)) for _ in range(Np1)]
        for i in reversed(range(N)):
            A = A_seq[i]
            B = B_seq[i]

            # Discretize
            A_d = np.eye(nx) + A * h
            B_d = B * h

            # Riccati step
            S = R_tvlqr + B_d.T @ P @ B_d

            K_i = np.linalg.solve(S, B_d.T @ P @ A_d)
            K_seq[i] = K_i

            # Update
            P = Q_tvlqr + A_d.T @ (P - P @ B_d @ np.linalg.solve(S, B_d.T @ P)) @ A_d

        # Reuse previous gains for the last node
        K_seq[N] = K_seq[N - 1].copy()

        self.K_seq = K_seq
        print(f"[TVLQR] Built time-varying LQR gains for {Np1} nodes.")


    def next(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return control from the PSC trajectory, with optional TVLQR tracking.

        Args:
            observation: Current state (nx,) from the environment

        Returns:
            u: Control (nu,) at current time step
            predictedX: Preview (5, nx) of the next 5 nominal states on the trajectory
        """

        # Advance the time in-line with the simulation
        now = time.time()
        if not hasattr(self, "last_call"):
            self.last_call = now
            dt = 0.0
        else:
            dt = now - self.last_call
            self.last_call = now

        # Clamp dt so we don't jump far along the trajectory after a stall
        if dt > 0.1:
            dt = 0.0

        # Advance PSC time using actual dt
        self.t_elapsed += dt

        # Don't go past the end of the PSC horizon
        if self.t_elapsed > self.Tf:
            self.t_elapsed = self.Tf

        # Map current time to a node index
        idx = np.searchsorted(self.t_grid, self.t_elapsed, side="left")
        idx = int(np.clip(idx, 0, self.N))

        # Nominal control and state from PSC
        u_nom = np.array(self.U_opt[:, idx]).flatten()
        x_nom = np.array(self.X_opt[:, idx]).flatten()

        # Actual state
        x = np.asarray(observation).flatten()
        
        # State error
        x_err = x - x_nom

        # Additional control effort if using TVLQR
        delta_u = np.zeros_like(u_nom)
        u = u_nom.copy()

        if self.use_tvlqr and (self.K_seq is not None):
            x = np.asarray(observation).flatten()
            K = self.K_seq[idx] 
            x_err = x - x_nom           

            delta_u = -K @ x_err      
            u = u_nom + delta_u

            if self.t_elapsed < 0.1 and self.debug:
                print("[TVLQR] ACTIVE at idx", idx)
                print("  ||x_err|| =", np.linalg.norm(x_err))
                print("  u_nom     =", u_nom)
                print("  delta_u   =", delta_u)
                print("  u_total   =", u)

        # Log info for plotting
        self.log_t.append(self.t_elapsed)
        self.log_idx.append(idx)
        self.log_x_actual.append(x)
        self.log_x_nom.append(x_nom)
        self.log_u_total.append(u)
        self.log_u_nom.append(u_nom)
        self.log_x_err_norm.append(float(np.linalg.norm(x_err)))
        self.log_delta_u_norm.append(float(np.linalg.norm(delta_u)))

        # Enforce bounds on control inputs
        u[0] = np.clip(u[0], 0.20, 1.0)
        u[1:] = np.clip(u[1:], -1.0, 1.0)

        # Build preview of nominal states
        NUM_PRED_EPOCHS = 5
        predictedX = np.zeros((NUM_PRED_EPOCHS, self.nx))
        for k in range(NUM_PRED_EPOCHS):
            j = int(np.clip(idx + k, 0, self.N))
            predictedX[k, :] = np.array(self.X_opt[:, j]).flatten()

        return u, predictedX

    
    def debug_plot_collocation_nodes(self) -> None:
        """
        Visualize the Chebyshev-Lobatto collocation nodes and optional
        positions along the PSC trajectory.
        """

        # Plot nodes on [-1, 1]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        ax = axs[0]
        ax.plot(self.tau, 'o-')
        ax.set_xlabel("Node index i")
        ax.set_ylabel(r"$\tau_i$ (dimensionless)")
        ax.set_title("Chebyshev-Lobatto nodes in $[-1, 1]$")
        ax.grid(True)

        # Plot corresponding time grid t in [0, Tf]
        ax = axs[1]
        ax.plot(self.t_grid, 'o-')
        ax.set_xlabel("Node index i")
        ax.set_ylabel("t_i [s]")
        ax.set_title(f"Collocation nodes in time [0, {self.Tf:.2f}]")
        ax.grid(True)

        plt.tight_layout()
        plt.show()
        # plt.savefig("output/Collocation_node_placement.png")

        # Plot altitude vs time
        pos_E = self.X_opt[7, :]  
        pos_N = self.X_opt[8, :]   
        pos_U = self.X_opt[9, :] 

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
        # plt.savefig("output/altitue_v_time.png")

    def debug_plot_tvlqr_tracking(self) -> None:
        """
        Plot:
        - Planned vs actual altitude over time
        - Norm of state tracking error ||x - x_nom||
        - Norm of LQR correction ||u - u_nom||
        - 3D planned vs actual trajectory
        """

        # Logged data from the sim
        t = np.array(self.log_t)
        x_act = np.array(self.log_x_actual)      # (T, nx)
        x_nom = np.array(self.log_x_nom)         # (T, nx)
        u_tot = np.array(self.log_u_total)       # (T, nu)
        u_nom = np.array(self.log_u_nom)         # (T, nu)
        err_norm = np.array(self.log_x_err_norm)
        du_norm = np.array(self.log_delta_u_norm)

        # Altitude index
        IDX_Z = 9

        # Altitude: planned vs actual
        z_act = x_act[:, IDX_Z]
        z_nom = x_nom[:, IDX_Z]

        plt.figure(figsize=(8, 4))
        plt.plot(t, z_nom, label="z_nom (PSC)")
        plt.plot(t, z_act, label="z_actual (sim)", linestyle="--")
        plt.xlabel("time [s]")
        plt.ylabel("altitude [m]")
        plt.title("Altitude: planned vs actual")
        plt.legend()
        plt.grid(True)

        # Tracking error norm
        plt.figure(figsize=(8, 4))
        plt.plot(t, err_norm)
        plt.xlabel("time [s]")
        plt.ylabel(r"$\|x - x_{\mathrm{nom}}\|$")
        plt.title("State tracking error norm")
        plt.grid(True)

        # LQR correction norm
        plt.figure(figsize=(8, 4))
        plt.plot(t, du_norm)
        plt.xlabel("time [s]")
        plt.ylabel(r"$\|u - u_{\mathrm{nom}}\|$")
        plt.title("Control correction norm (LQR effort)")
        plt.grid(True)

        # Planned vs Actual Trajectory in 3D
        if hasattr(self, "last_X_traj") and self.last_X_traj is not None:
            X_plan = self.last_X_traj 
        else:
            print("[TVLQR DEBUG] No last_X_traj found, can't make 3D plot.")
            return

        posE_plan = X_plan[7, :]   
        posN_plan = X_plan[8, :]  
        posZ_plan = X_plan[9, :] 

        posE_act = x_act[:, 7]
        posN_act = x_act[:, 8]
        posZ_act = x_act[:, 9]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(posE_plan, posN_plan, posZ_plan,
                label="Planned (PSC)", linewidth=2)

        ax.plot(posE_act, posN_act, posZ_act,
                linestyle="--", label="Actual (Sim)", linewidth=2)

        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_zlabel("Up [m]")
        ax.set_title("3D Trajectory: Planned vs Actual")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()
        # plt.savefig("output/tvlqr_tracking.png")
