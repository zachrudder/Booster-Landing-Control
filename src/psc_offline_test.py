# psc_offline_test.py

import numpy as np
from simrocketenv import SimRocketEnv
from pscpolicy import PSCTVLQRPolicy

def run_offline_psc_test():
    env = SimRocketEnv(interactive=False)
    state = env.state.copy()

    policy = PSCTVLQRPolicy(state, time_horizon=3.0, N_nodes=40, hover=False)

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

if __name__ == "__main__":
    run_offline_psc_test()
