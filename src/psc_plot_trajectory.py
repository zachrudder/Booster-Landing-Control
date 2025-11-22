import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

def plot_trajectory_3d(X_traj):
    """
    Plot 3D rocket trajectory using position states.
    X_traj: ndarray of shape (N+1, 16)
    """

    # Extract ENU positions
    pos_E = X_traj[7, :]   # East
    pos_N = X_traj[8, :]   # North
    pos_U = X_traj[9, :]   # Up

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory line
    ax.plot(pos_E, pos_N, pos_U, '-o', markersize=3, linewidth=2, label='PSC Traj')

    # Mark start and end
    ax.scatter(pos_E[0], pos_N[0], pos_U[0], color='green', s=70, label='Start')
    ax.scatter(pos_E[-1], pos_N[-1], pos_U[-1], color='red', s=70, label='End')

    # Set labels
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("Rocket PSC Trajectory (3D Position)")

    # Equal scaling for axes
    max_range = np.array([pos_E.max()-pos_E.min(),
                          pos_N.max()-pos_N.min(),
                          pos_U.max()-pos_U.min()]).max() / 2.0

    mid_x = (pos_E.max()+pos_E.min()) * 0.5
    mid_y = (pos_N.max()+pos_N.min()) * 0.5
    mid_z = (pos_U.max()+pos_U.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


def quat_to_R_body_to_nav(qw, qx, qy, qz):
    """
    Convert quaternion (qw, qx, qy, qz) to rotation matrix R_b_to_n.
    This matches the convention used in rocket_model.py:
    q maps from body frame -> navigation (ENU) frame.
    """
    # Precompute products
    q0, q1, q2, q3 = qw, qx, qy, qz

    R = np.zeros((3, 3))
    R[0, 0] = 1.0 - 2.0*q2*q2 - 2.0*q3*q3
    R[0, 1] = 2.0*q1*q2 - 2.0*q3*q0
    R[0, 2] = 2.0*q1*q3 + 2.0*q2*q0

    R[1, 0] = 2.0*q1*q2 + 2.0*q3*q0
    R[1, 1] = 1.0 - 2.0*q1*q1 - 2.0*q3*q3
    R[1, 2] = 2.0*q2*q3 - 2.0*q1*q0

    R[2, 0] = 2.0*q1*q3 - 2.0*q2*q0
    R[2, 1] = 2.0*q2*q3 + 2.0*q1*q0
    R[2, 2] = 1.0 - 2.0*q1*q1 - 2.0*q2*q2

    return R


def plot_trajectory_3d_with_orientation_and_thrust(X_traj):
    """
    Plot 3D rocket trajectory with:
      - position
      - orientation (body z-axis)
      - thrust vector

    Parameters
    ----------
    X_traj : np.ndarray of shape (16, N+1)
        PSC state trajectory:
          0..3  : quaternion [qw, qx, qy, qz]
          4..6  : angular rates
          7..9  : position [E, N, U]
          10..12: velocity [vE, vN, vU]
          13    : thrust magnitude (N)
          14    : t_alpha
          15    : t_beta
    """

    # ---- Extract positions ----
    pos_E = X_traj[7, :]   # East
    pos_N = X_traj[8, :]   # North
    pos_U = X_traj[9, :]   # Up

    # ---- Extract quaternions ----
    qw = X_traj[0, :]
    qx = X_traj[1, :]
    qy = X_traj[2, :]
    qz = X_traj[3, :]

    # ---- Extract thrust-related state ----
    thrust_N = X_traj[13, :]   # thrust magnitude in Newtons
    t_alpha  = X_traj[14, :]   # thrust deflection in x-direction
    t_beta   = X_traj[15, :]   # thrust deflection in y-direction

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # ---- Plot trajectory line ----
    ax.plot(pos_E, pos_N, pos_U, '-o', markersize=3, linewidth=2, label='PSC Traj')

    # Mark start and end
    ax.scatter(pos_E[0], pos_N[0], pos_U[0], color='green', s=70, label='Start')
    ax.scatter(pos_E[-1], pos_N[-1], pos_U[-1], color='red', s=70, label='End')

    # ---- Choose subset of nodes for arrows (to avoid clutter) ----
    Np1 = X_traj.shape[1]
    indices = np.linspace(0, Np1 - 1, num=min(15, Np1), dtype=int)

    # Arrow lengths (visual only, units in meters-ish)
    body_axis_len   = 2.0
    thrust_arrow_len = 3.0

    for i in indices:
        # Rotation body -> nav
        R_b_to_n = quat_to_R_body_to_nav(qw[i], qx[i], qy[i], qz[i])

        # ---- Orientation arrow: body z-axis (0,0,1)^T in body frame ----
        body_z_nav = R_b_to_n @ np.array([0.0, 0.0, 1.0])

        # Arrow start = current position
        x0 = pos_E[i]
        y0 = pos_N[i]
        z0 = pos_U[i]

        # Body axis arrow
        u1 = body_axis_len * body_z_nav[0]
        v1 = body_axis_len * body_z_nav[1]
        w1 = body_axis_len * body_z_nav[2]

        ax.quiver(x0, y0, z0, u1, v1, w1,
                  length=2.0, normalize=False, color='purple')

        # ---- Thrust vector arrow (scaled by magnitude) ----
        T  = thrust_N[i]
        ta = t_alpha[i]
        tb = t_beta[i]

        # Thrust direction in body frame
        thrust_body = np.array([
            T * ta,
            T * tb,
            T
        ])

        # Rotate to navigation frame
        thrust_nav = R_b_to_n @ thrust_body

        # Scale arrow by thrust magnitude
        T_max = 1800.0  # based on your rocket model
        k = 5.0         # visualization scale factor

        arrow_len = k * (T / T_max)

        # Arrow vector in nav frame (not normalized!)
        u2 = arrow_len * thrust_nav[0] / (np.linalg.norm(thrust_nav) + 1e-6)
        v2 = arrow_len * thrust_nav[1] / (np.linalg.norm(thrust_nav) + 1e-6)
        w2 = arrow_len * thrust_nav[2] / (np.linalg.norm(thrust_nav) + 1e-6)

        ax.quiver(x0, y0, z0, u2, v2, w2,
                length=1.0, normalize=False, color='red')


    # ---- Labels, scaling, etc. ----
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("PSC Trajectory: Position + Orientation + Thrust Vector")

    # Equal scaling for axes
    max_range = np.array([pos_E.max()-pos_E.min(),
                          pos_N.max()-pos_N.min(),
                          pos_U.max()-pos_U.min()]).max() / 2.0

    mid_x = (pos_E.max()+pos_E.min()) * 0.5
    mid_y = (pos_N.max()+pos_N.min()) * 0.5
    mid_z = (pos_U.max()+pos_U.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()



def plot_u_vs_time(U, t=None, Tf=None, title="Control inputs vs time"):
    """
    Plot control inputs u_i(t).

    Parameters
    ----------
    U : array-like
        Control trajectory. Shape can be either:
          - (nu, Np1)  (controls x time)
          - (Np1, nu)  (time x controls)
    t : array-like, optional
        Time stamps for each column of U. Shape (Np1,).
        If None, a uniform grid in [0, Tf] (or [0,1] if Tf is None) is used.
    Tf : float, optional
        Final time, only used if t is None. Default is 1.0.
    title : str
        Title for the plot.
    """
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must be 2D, got shape {}".format(U.shape))

    # Make sure U has shape (nu, Np1)
    if U.shape[0] < U.shape[1]:
        nu, Np1 = U.shape
    else:
        # assume (Np1, nu), transpose
        Np1, nu = U.shape
        U = U.T

    # Time vector
    if t is None:
        if Tf is None:
            Tf = 1.0
        t = np.linspace(0.0, Tf, Np1)
    else:
        t = np.asarray(t)
        if t.shape[0] != Np1:
            raise ValueError(f"t has length {t.shape[0]}, but U has {Np1} time steps")

    plt.figure(figsize=(8, 5))
    for i in range(nu):
        plt.plot(t, U[i, :], label=f"u[{i}]")

    plt.xlabel("time [s]")
    plt.ylabel("control input")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
