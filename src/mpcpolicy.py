from basecontrol import BaseControl

import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc.rocket_model import export_rocket_ode_model

class MPCPolicy(BaseControl):
    def __init__(self, initial_state, time_horizon=4.0, epochs_per_sec=20):
        super().__init__()

        self.ocp        = AcadosOcp()
        self.model      = export_rocket_ode_model()
        self.ocp.model  = self.model
        self.Tf         = time_horizon
        self.nx         = self.model.x.size()[0]
        self.nu         = self.model.u.size()[0]
        self.ny         = self.nx + self.nu
        self.ny_e       = self.nx
        self.N_horizon  = int(epochs_per_sec*self.Tf)
        self.ocp.dims.N = self.N_horizon

        # set cost module
        self.ocp.cost.cost_type   = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # ### NEW: Store Q_mat as a class property so we can access it in next()
        self.Q_mat                = np.diag(self.model.weight_diag) 
        
        # Initialize R with a default value (will be overwritten in next())
        self.R_mat                = np.diag(np.ones(self.nu, )*100.0)

        self.ocp.cost.W           = scipy.linalg.block_diag(self.Q_mat, self.R_mat)
        self.ocp.cost.W_e         = self.Q_mat
        self.Vu                   = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vx_e        = np.eye(self.nx)

        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = self.Vu
        self.Vu[self.nx : self.nx + self.nu, 0:self.nu] = np.eye(self.nu)

        # Setpoint state
        setpoint_yref        = np.zeros((self.ny, ))
        setpoint_yref[0]     = 1.0  
        # Note: Ensure index 9 corresponds to your Z (altitude) in the state vector
        self.alt_index       = 9 # ### NEW: Store index for altitude
        setpoint_yref[self.alt_index] = 1.0
        
        self.ocp.cost.yref   = setpoint_yref
        self.ocp.cost.yref_e = setpoint_yref[0:self.nx]

        # Constraints
        self.ocp.constraints.constr_type = 'BGH'
        self.ocp.constraints.lbu = np.array([ 0.0, -1.0, -1.0 ])
        self.ocp.constraints.ubu = np.array([ 1.00,  1.0,  1.0 ])
        self.ocp.constraints.x0 = initial_state
        self.ocp.constraints.idxbu = np.array(range(self.nu))

        # Solver options
        self.ocp.solver_options.qp_solver        = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx   = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type  = 'ERK'
        self.ocp.solver_options.nlp_solver_type  = 'SQP_RTI'
        self.ocp.solver_options.qp_solver_cond_N = self.N_horizon
        self.ocp.solver_options.tf               = self.Tf

        solver_json = 'acados_ocp_' + self.model.name + '.json'
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=solver_json)

    def get_name(self):
        return "MPC_Adaptive_R"

    def next(self, observation):
        current_alt = observation[self.alt_index] 
        
        # --- CONFIGURATION ---
        R_low       = 1.0     # Cost when near ground (Cheap)
        R_high      = 10000.0 # Cost when high up (Forbidden)
        
        # The "Wall" location (e.g., 40% of your max altitude)
        # If your max alt is ~10m, set this to 4.0
        alt_threshold = 20.0   
        
        # How sharp is the transition? Higher = sharper wall
        # 1.0 is gentle, 10.0 is a cliff.
        steepness   = 5.0    
        # ---------------------

        # 1. Sigmoid Function (S-Curve)
        # This smoothly transitions from R_low to R_high as you cross the threshold
        sigmoid_factor = 1 / (1 + np.exp(-steepness * (current_alt - alt_threshold)))
        
        thrust_weight = R_low + (R_high - R_low) * sigmoid_factor

        # 2. Update R Matrix
        new_R_diag = np.array([thrust_weight, 100.0, 100.0]) 
        new_R_mat = np.diag(new_R_diag)

        # 3. Combine and Update Solver
        # (Be sure to keep your Q_mat logic from previous steps if you want to keep that)
        new_W = scipy.linalg.block_diag(self.Q_mat, new_R_mat)

        for i in range(self.N_horizon):
            self.acados_ocp_solver.cost_set(i, "W", new_W)

        action = self.acados_ocp_solver.solve_for_x0(x0_bar=observation)

        NUM_PRED_EPOCHS = 5
        step_size = self.N_horizon // NUM_PRED_EPOCHS

        predictedX = np.ndarray((NUM_PRED_EPOCHS, self.nx))
        for i in range(NUM_PRED_EPOCHS):
            predictedX[i,:] = self.acados_ocp_solver.get(i * step_size, "x")

        return action, predictedX