# psc_offline_test.py

import numpy as np
from simrocketenv import SimRocketEnv
from pscpolicy import PSCTVLQRPolicy
from psc_plot_trajectory import (plot_trajectory_3d, 
                                 plot_trajectory_3d_with_orientation_and_thrust,
                                 plot_u_vs_time
)

def run_offline_psc_test():
    env = SimRocketEnv(interactive=False)
    state = env.state.copy()
    initial_state_1 = np.array([
        # q: upright (no rotation)
        1.0, 0.0, 0.0, 0.0,
        # omega: almost no angular rates
        0.0, 0.0, 0.0,
        # pos: small lateral offset, mid altitude
        5.0,  -5.0,  45.0,      # E, N, U
        # vel: slight lateral and downward motion
        -2.0, 1.0,  -3.0,       # V_E, V_N, V_U
        # thrust: mid in the [0.65,0.75]*1800 range
        0.7 * 1800.0,           # 1260 N
        # thrust vector angles
        0.0, 0.0                 # alpha, beta
    ])

    initial_state_2 = np.array([
        0.99904822, 0.04361939, 0.0, 0.0,    # q (5° roll)
        0.0, 0.0, 0.0,                       # omega
        0.0, 0.0, 40.0,                      # pos: above origin
        0.0, 0.0, -5.0,                      # vel: falling slowly
        0.7 * 1800.0,                        # thrust
        0.0, 0.0
    ])

    initial_state_3 = np.array([
        0.99904822, 0.0, 0.04361939, 0.0,    # q (5° pitch)
        0.0, 0.0, 0.0,                       # omega
        -10.0, 10.0, 50.0,                   # pos
        3.0, -1.0, -4.0,                     # vel
        0.72 * 1800.0,                       # 1296 N
        0.0, 0.0
    ])

    # hard case
    initial_state_4 = np.array([
        1.0, 0.0, 0.0, 0.0,                  # upright
        np.deg2rad(3.0),                     # omega_x (3°/s)
        np.deg2rad(-2.0),                    # omega_y (-2°/s)
        0.0,                                 # omega_z
        10.0, -15.0, 35.0,                   # pos (closer to ground)
        -5.0, 2.0, -8.0,                     # vel (faster lateral and downward)
        0.68 * 1800.0,                       # ~1224 N
        0.0, 0.0
    ])

    # near target
    initial_state_5 = np.array([
        1.0, 0.0, 0.0, 0.0,                  # upright
        0.0, 0.0, 0.0,                       # omega
        0.5, -0.5, 3.0,                      # pos: close to (0,0,2.42)
        0.5, -0.3, -0.5,                     # small velocities
        0.5 * 1800.0,                        # ~hover-ish thrust (rough)
        0.0, 0.0
    ])


    # state = initial_state_1
    policy = PSCTVLQRPolicy(state, time_horizon=16.0, N_nodes=50, hover=False, use_tvlqr=True)

    # Use the same dt as PSC node spacing or something small like env.dt_sec
    T = policy.Tf
    t_grid = policy.t_grid  # (N+1,)
    U_opt = policy.U_opt    # (nu, N+1)

    # Simple: at each env step, pick u based on current time
    t = 0.0
    dt = 1.0 / 120.0
    traj_states = []
    traj_controls = []

    while t <= T:
        # choose node index based on t
        alpha = t / T
        i_float = alpha * policy.N
        idx = int(np.clip(np.round(i_float), 0, policy.N))

        u = np.array(U_opt[:, idx]).flatten()
        env.dt_sec = dt
        state, reward, done, _, _ = env.step(u)

        traj_states.append(state.copy())
        traj_controls.append(u.copy())

        t += dt
        if done:
            break

    print("[PSC] Offline rollout finished. Steps:", len(traj_states))
    # here you could print final altitude, attitude, etc.

    # plot the trajectory
    # X_traj = policy.last_X_traj
    # plot_trajectory_3d(X_traj)

    # plot trajectory with orientation
    X_traj = policy.last_X_traj
    plot_trajectory_3d_with_orientation_and_thrust(X_traj)

    # plot collocation points
    policy.debug_plot_collocation_nodes()

    # plot control inputs vs time
    U_traj = policy.last_U_traj
    plot_u_vs_time(U_traj)

if __name__ == "__main__":
    run_offline_psc_test()
