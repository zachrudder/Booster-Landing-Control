"""
  Physics simulation module using pybullet

  pybullet world frame is ENU EAST (X) NORTH (Y) UP (Z)
  pybullet body frame is FORWARD (X) LEFT (Y) UP (Z)

  (c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from geodetic_toolbox import quat_from_rpy, quat_to_matrix, quat_to_rpy, quat_invert
from scipy.interpolate import CubicSpline

class SimRocketEnv(gym.Env):
    """
    Rocket simulation environment (physics simulation) with an
    OpenAI gym interface / gymnasium interface. pybullet is
    used for the heavy lifting under the hood.
    """
    def __init__(self, interactive=False, scale_obs_space=1.0):
        print("PyRocketCraft")
        self.pybullet_initialized = False
            
        self.attitude_control_on = True # set to True to allow attitude control thrusters

        self.interactive = interactive
        self.reset_count = 0 # keep track of calls to reset() function
        self.time_sec = 0.0 # keep track of simulation time
        self.dt_sec = 1.0 / 120.0 # update rate of the simulation
        self.WAIT_ON_GROUND_SEC = 3.0 # wait for X sec. after engine shutdown

        # <pybullet>
        self.debug_line_thrust = -1
        self.pybullet_body = -1
        self.pybullet_booster_index = -1
        self.pybullet_time_sec = self.time_sec
        self._pybullet_setup_environment() # one time setup of pybullet
        # </pybullet>

        # <vehicle specific>
        self.urdf_file = "./src/modelrocket.urdf"
        self.UMIN = -1.0 # min. control input
        self.UMAX =  1.0 # max. control input for thrust (= 100%)
        if self.attitude_control_on:
            self.ACTUATORCOUNT = 5 # main thrust, 2x thrust vector, 2x attitude
        else: 
            self.ACTUATORCOUNT = 3 # main thrust, 2x thrust vector
        
        self.INITIAL_FUEL = 12.0 # initial fuel mass, kg
        self.TOTAL_FUEL = self.INITIAL_FUEL # fuel mass, kg
        self.ISP = 200.0 # specific impulse, seconds
        self.fuel_consumed_kg = 0.0 # fuel consumed in last step

        self.THRUST_UMIN = 0.0 # min. control input for main thrust
        self.THRUST_MAX_N = 1800.0 # max. thrust in Newton from main engine
        self.THRUST_TAU = 0.3 # PT1 first order delay in thrust response, originally 2.5, made 0.3 for better response
        self.THRUST_VECTOR_TAU = 0.3
        self.THRUST_MAX_ANGLE = np.deg2rad(10.0)
        self.ATT_MAX_THRUST = 50.0 # attitude thruster: max. thrust in Newton
        self.GRAVITY = 9.81 # assume we want to land on Earth
        self.AIR_DENSITY = 1.225 # kg/m^3 at sea level
        self.Cd = 0.47 # Approx Cd for a cylinder
        self.mass_kg = -99999999.9 # will be loaded and updated from URDF
        self.MIN_GROUND_DIST_M = 2.45 # shut off engine below this altitude
        # OFFSET between CoG and nozzle. Is there a way to get this from URDF?
        self.NOZZLE_OFFSET = -2.0
        self.ATT_THRUSTER_OFFSET = 2.0
        self.scale_obs_space = scale_obs_space
        # </vehicle specific>

        # <state vector config>
        self.state_cfg = {}
        self.state = np.zeros((16,))
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # attitude quaternion
        self.omega = np.array([0.0, 0.0, 0.0]) # angular rate (body)
        self.pos_n = np.array([0.0,0.0,0.0]) # East North Up Position (m)
        self.vel_n = np.array([0.0,0.0,0.0]) # East North Up Velocity (m/s)
        self.thrust_current_N = 0.0 # Thrust in Newton
        self.thrust_alpha = 0.0 # Thrust deflection angle alpha in [rad]
        self.thrust_beta = 0    # Thrust deflection angle beta in [rad]
        self.rand_drag_vel_mag = np.random.uniform(-50.0, 50.0) # choose initial mag of wind
        self.rand_drag_dir_roll = np.random.uniform(0.0,np.pi)
        self.rand_drag_dir_pitch = np.random.uniform(0.0,np.pi)
        self.rand_drag_dir_yaw = np.random.uniform(0.0,np.pi) # get angles of rand wind
        self.init_height = 0.0 # will update with self.pos_n initialization
        # </state vector config>
        self.roll_deg = 0.0    # helper: mirror attitude in euler angles
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        # initialize state of the vehicle with the actual values
        state, _ = self.reset() # reset state and fill self.state vector
        # </state>
        self.engine_on = True # not part of state vector

        # Setup Gym environment interface settings
        self.action_space = spaces.Box(low=np.float32(self.UMIN),
                                       high=np.float32(self.UMAX),
                                       shape=(self.ACTUATORCOUNT,),
                                       dtype=np.float32)
        self.action_space.low[0] = np.float32(self.THRUST_UMIN)
        obs_hi = np.ones(state.shape[0]) * 2000.0
        self.observation_space = spaces.Box(low=-np.float32(obs_hi),
                                            high=np.float32(obs_hi),
                                            dtype=np.float32)

    def print_urdf(self, body):
        """
        Helper function to output URDF model information.
        """
        # Run this with your env instance available (uses self.CLIENT and self.pybullet_body)
        body = self.pybullet_body
        print("pybullet body id:", body)

        n_joints = p.getNumJoints(body, physicsClientId=self.CLIENT)
        print("num joints:", n_joints)

        for i in range(n_joints):
            info = p.getJointInfo(body, i, physicsClientId=self.CLIENT)
            # info fields: (jointIndex, name, type, qIndex, uIndex, flags, jointDamp, jointFric, jointLowerLimit, jointUpperLimit, ...)
            name = info[1].decode() if isinstance(info[1], bytes) else info[1]
            jtype = info[2]
            parent = info[16]
            print(f"joint {i}: name={name} type={jtype} parent={parent}")
            dyn = p.getDynamicsInfo(body, i, physicsClientId=self.CLIENT)
            print(f"  mass={dyn[0]}, localInertiaPos={dyn[3]}")

        # base link dynamics
        base_dyn = p.getDynamicsInfo(body, -1, physicsClientId=self.CLIENT)
        print("base mass:", base_dyn[0])

        # visual/collision info
        print("visuals:", p.getVisualShapeData(body, physicsClientId=self.CLIENT))

        # world pose and base velocity
        print("base pose:", p.getBasePositionAndOrientation(body, physicsClientId=self.CLIENT))
        print("base velocity:", p.getBaseVelocity(body, physicsClientId=self.CLIENT))

    def line_projections_z_aligned(self, L, roll_deg, pitch_deg, yaw_deg):
        """
        Helper function to convert vehicle length and roll/pitch/yaw to projected
        lengths for drag calcs.
        """
        phi = np.radians(roll_deg)
        theta = np.radians(pitch_deg)
        psi = np.radians(yaw_deg)

        cphi, sphi = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        Lx = L * (-cpsi*st*cphi + spsi*sphi)
        Ly = L * (-spsi*st*cphi - cpsi*sphi)
        Lz = L * (ct*cphi)
        return Lx, Ly, Lz

    import numpy as np
    

    def random_beta_sample(self, a, b, alpha_range=(1.0, 9.0), beta_range=(1.0, 9.0)):
        """
        Generates a random number in [a,b] from a Beta distribution
        with randomly chosen alpha and beta.

        """
        alpha = np.random.uniform(*alpha_range)
        beta_param = np.random.uniform(*beta_range)
        
        x = np.random.beta(alpha, beta_param)
        return a + (b - a) * x
    
    def _pybullet_setup_environment(self):
        """
        Connect to pybullet environment.
        """
        assert self.pybullet_initialized is False
        self.pybullet_initialized = True

        self.PYBULLET_DT_SEC = 1.0/240.0

        # connect to pybullet and get the client id
        if self.interactive:
            print("\033[33mpybullet physics active.\033[0m")
            self.CLIENT = p.connect(p.GUI)
        else:
            self.CLIENT = p.connect(p.DIRECT)     

    def reset(self, seed=14, options={}) -> float:
        """
        Gym interface. Reset the simulation.
        :return state (state vector), info dict.
        """
        np.random.seed(seed)

        self.engine_on = True
        # <state>
        # SETTING RANDOM INITIAL POSITION
        self.pos_n = np.array([np.random.uniform(-50.0, 50.0),
                               np.random.uniform(-50.0, 50.0),
                               np.random.uniform( 180.0, 220.0)]) # ENU
        self.init_height = self.pos_n[2]
        

        self.vel_n = np.array([np.random.uniform( -8.0,  8.0),
                               np.random.uniform( -8.0,  8.0),
                               np.random.uniform(-15.0,  5.0)]) * self.scale_obs_space # ENU

        # Maintain the attitude as quaternion and Euler angles. The source of truth is
        # the quaternion (self.q) and roll_deg, pitch_deg and yaw_deg will be updated
        # based on the quaternion. But here for initialization the Euler angles are
        # used to initialize the orientation (Euler angles are a bit more readable)
        self.roll_deg  = np.random.uniform(60.0, 180.0) * self.scale_obs_space
        self.pitch_deg = np.random.uniform(-30.0, 30.0) * self.scale_obs_space
        self.yaw_deg   = 0.0
        # Attitude quaternion (transforming from body to navigation system
        # Careful: quaternion order: qw, qx,qy,qz (qw is the real part)
        self.q         = quat_from_rpy(np.deg2rad(self.roll_deg),
                                       np.deg2rad(self.pitch_deg),
                                       np.deg2rad(self.yaw_deg))

        roll_rate_rps  = np.deg2rad(np.random.uniform(-5.0, 5.0)) * self.scale_obs_space
        pitch_rate_rps = np.deg2rad(np.random.uniform(-5.0, 5.0)) * self.scale_obs_space
        yaw_rate_rps   = 0.
        self.omega     = np.array([roll_rate_rps,
                                   pitch_rate_rps,
                                   yaw_rate_rps])

        self.thrust_current_N = 0.0 # set initial thrust to 0.0 #np.random.uniform(0.65, 0.75) * self.THRUST_MAX_N
        self.thrust_alpha = 0.0 # 0 means no deflection of thrust vectoring
        self.thrust_beta = 0.0

        self.last_fuel = self.TOTAL_FUEL
        self.TOTAL_FUEL = self.INITIAL_FUEL

        # </state>
        self._update_state() # create/update state vector

        # <simulation>
        self.time_sec = 0.0
        self.reset_count += 1
        self.epochs = 0
        self.time_on_ground_sec = 0.0 # wait a few seconds after engine shutdown
        # </simulation>

        self._pybullet_reset_environment()
        return self.state, {}



    def _pybullet_reset_environment(self):
        """
        Cleanup all pybullet objects, reset and restart simulation environment.
        """
        self.pybullet_time_sec = self.time_sec
        p.resetSimulation(physicsClientId=self.CLIENT) # remove all objects and reset

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        p.setGravity( 0.0, 0.0, -self.GRAVITY, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYBULLET_DT_SEC, physicsClientId=self.CLIENT)

        plane = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        p.changeDynamics(plane, -1, lateralFriction=1, restitution=0.5, physicsClientId=self.CLIENT)

        # Experimental code to include terrain map:
        # terrainShape = p.createCollisionShape(
        #           shapeType = p.GEOM_HEIGHTFIELD,
        #           meshScale=[0.5,0.5,40.0],
        #           fileName = "heightmaps/wm_height_out.png",
        #           physicsClientId=self.CLIENT)
        # textureId = p.loadTexture("heightmaps/gimp_overlay_out.png",
        #                           physicsClientId=self.CLIENT)
        # terrain  = p.createMultiBody(0, terrainShape,
        #                              physicsClientId=self.CLIENT)
        # p.changeVisualShape(terrain, -1, textureUniqueId = textureId,
        #                     physicsClientId=self.CLIENT)

        initial_position_enu = [self.pos_n[0], self.pos_n[1], self.pos_n[2]]
        self.pybullet_body = p.loadURDF(self.urdf_file,
                                        initial_position_enu,
                                        physicsClientId=self.CLIENT)
        
        # print_urdf(self.pybullet_body) # uncomment to output URDF model info

        # compute and store model length (z-extent) from URDF
        try:
            self.model_length, self.model_zmin, self.model_zmax, self.model_avg_cyl_radius = self._compute_urdf_length(self.urdf_file)
            print(f"model length (z-extent): {self.model_length:.3f} m  (zmin={self.model_zmin:.3f}, zmax={self.model_zmax:.3f}) avg_cyl_radius={self.model_avg_cyl_radius:.4f} m")
        except Exception as e:
            print("failed to compute model length from URDF:", e)
            self.model_length = None
            self.model_avg_cyl_radius = None
        self.pybullet_booster_index = -1

        self.cg_loc = self.model_length * 0.35 # approx. CoG location from base


        q_rosbody_to_enu = self.q
        # pybullet needs the scalar part at the end of the quaternion:
        qxyzw_rosbody_to_enu = [ q_rosbody_to_enu[1],   # img. part x
                                 q_rosbody_to_enu[2],   # img. part y
                                 q_rosbody_to_enu[3],   # img. part z
                                 q_rosbody_to_enu[0] ]  # real part
        p.resetBasePositionAndOrientation(self.pybullet_body,
                                          initial_position_enu,
                                          qxyzw_rosbody_to_enu,
                                          physicsClientId=self.CLIENT)
        self.mass_kg = self._get_total_mass(self.pybullet_body)

        self.debug_line_thrust = -1

    def _set_camera_follow_object(self, object_id, dist=4.5, pitch=-55, yaw=50):
        """
        Helper function to set camera to object.
        """
        pos, _ = p.getBasePositionAndOrientation(object_id,
                                                 physicsClientId=self.CLIENT)
        p.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=pos, physicsClientId=self.CLIENT
        )

    def _pybullet_physics(self, u):
        """
        Advance the physics simulation until self.pybullet_time_sec catches up
        with self.time_sec.
        :param u Control input vector
        """
        self._set_camera_follow_object(self.pybullet_body)

        pybullet_dt_sec = 0.0
        # advance pybullet simulation until current time
        while self.pybullet_time_sec < self.time_sec:
            # thrust dynamics (i.e. the thrust takes some time to react)
            # Time constant is "tau"
            # The thrust vectoring also takes some time to react
            self.thrust_current_N += (self.THRUST_MAX_N     * u[0] - self.thrust_current_N) * self.PYBULLET_DT_SEC / self.THRUST_TAU
            self.thrust_alpha     += (self.THRUST_MAX_ANGLE * u[1] - self.thrust_alpha) * self.PYBULLET_DT_SEC / self.THRUST_VECTOR_TAU
            self.thrust_beta      += (self.THRUST_MAX_ANGLE * u[2] - self.thrust_beta)  * self.PYBULLET_DT_SEC / self.THRUST_VECTOR_TAU

            # thrust vector in body coordinates
            # (assuming small thrust vectoring angles)
            thrust = np.array([self.thrust_alpha,  # x forward
                               self.thrust_beta,   # y left
                               1.0]) * self.thrust_current_N # z up

            # Add force of rocket boost to pybullet simulation
            if self.engine_on and self.TOTAL_FUEL > 0.0:
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=self.pybullet_booster_index,
                                     forceObj=[thrust[0], thrust[1], thrust[2]],
                                     posObj=[0, 0, self.NOZZLE_OFFSET],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)
                thrust_mag = np.linalg.norm(thrust)
                m_dot = thrust_mag / (self.ISP * self.GRAVITY) # mass flow rate
                self.fuel_consumed_kg = m_dot * self.PYBULLET_DT_SEC
                self.TOTAL_FUEL -= self.fuel_consumed_kg
                # print("FUEL LEFT:", self.TOTAL_FUEL)

            # attitude correction thruster
            if self.attitude_control_on:
                att_x_thrust = u[3] * self.ATT_MAX_THRUST # x forward
                att_y_thrust = u[4] * self.ATT_MAX_THRUST # y left
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                    linkIndex=self.pybullet_booster_index,
                                    forceObj=[att_x_thrust, att_y_thrust, 0.0],
                                    posObj=[0, 0, self.ATT_THRUSTER_OFFSET],
                                    flags=p.LINK_FRAME,
                                    physicsClientId=self.CLIENT)
            
            # STANDARD DRAG FROM X/Y/Z DIRS
            curr_vel = self.vel_n
            vel_x, vel_y, vel_z = curr_vel

            # print(self.model_length)
            # print(self.roll_deg, self.pitch_deg, self.yaw_deg)
            # print("VELOCITIES:", vel_x, vel_y, vel_z)
            Lx, Ly, Lz = self.line_projections_z_aligned(self.model_length, self.roll_deg, self.pitch_deg, self.yaw_deg)
            # print("PROJECTIONS:", Lx, Ly, Lz)
            Ax = Lx * 2.0 * self.model_avg_cyl_radius
            Ay = Ly * 2.0 * self.model_avg_cyl_radius
            Az = Lz * 2.0 * self.model_avg_cyl_radius
            # print("AREAS:", Ax, Ay, Az)

            drag_x = 1/2 * self.AIR_DENSITY * self.Cd * Ax * (vel_x**2)
            drag_y = 1/2 * self.AIR_DENSITY * self.Cd * Ay * (vel_y**2)
            drag_z = 1/2 * self.AIR_DENSITY * self.Cd * Az * (vel_z**2)

            # print("DRAGS:", drag_x, drag_y, drag_z)

            p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                 linkIndex=self.pybullet_booster_index,
                                 forceObj=[drag_x, drag_y, drag_z],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT)
            
            # RANDOM DRAG BURSTS
            rand_drag_vel_x, rand_drag_vel_y, rand_drag_vel_z = self.line_projections_z_aligned(self.rand_drag_vel_mag, 
                                                                                             self.rand_drag_dir_roll,
                                                                                             self.rand_drag_dir_pitch, 
                                                                                             self.rand_drag_dir_yaw) 
            
            Lx_rand, Ly_rand, Lz_rand = self.line_projections_z_aligned(self.model_length, 
                                                        self.rand_drag_dir_roll,
                                                        self.rand_drag_dir_pitch, 
                                                        self.rand_drag_dir_yaw) 
            Ax_rand = Lx_rand * 2.0 * self.model_avg_cyl_radius
            Ay_rand = Ly_rand * 2.0 * self.model_avg_cyl_radius
            # Az_rand = Lz_rand * 2.0 * self.model_avg_cyl_radius

            height_scalar = self.pos_n[2] / self.init_height
            
            rand_drag_x = 1/2 * self.AIR_DENSITY * self.Cd * Ax_rand * (rand_drag_vel_x**2) * height_scalar**2
            rand_drag_y = 1/2 * self.AIR_DENSITY * self.Cd * Ay_rand * (rand_drag_vel_y**2) * height_scalar**2
            # rand_drag_z = 1/2 * self.AIR_DENSITY * self.Cd * Az_rand * (rand_drag_vel_z**2) * height_scalar**2

            p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                 linkIndex=self.pybullet_booster_index,
                                 forceObj=[rand_drag_x, rand_drag_y, 0.0],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT)
            
            vel_mag_step = self.random_beta_sample(-0.1,0.1)
            rand_roll_step = self.random_beta_sample(-0.1,0.1)
            rand_pitch_step = self.random_beta_sample(-0.1,0.1)
            rand_yaw_step = self.random_beta_sample(-0.1,0.1)

            self.rand_drag_vel_mag += vel_mag_step
            self.rand_drag_dir_roll += rand_roll_step
            self.rand_drag_dir_pitch += rand_pitch_step
            self.rand_drag_dir_yaw += rand_yaw_step

            # print("RANDOM DRAG VEL MAG:", self.rand_drag_vel_mag)
            # print("RANDOM DRAG DIRS (roll,pitch,yaw):", self.rand_drag_dir_roll, self.rand_drag_dir_pitch, self.rand_drag_dir_yaw)

            p.stepSimulation(physicsClientId=self.CLIENT)
            self.pybullet_time_sec += self.PYBULLET_DT_SEC
            pybullet_dt_sec += self.PYBULLET_DT_SEC

        # debug
        # print(f"t={self.time_sec:.2f}, z={self.pos_n[2]:.2f}, engine_on={self.engine_on}, thrust_N={self.thrust_current_N:.1f}")

        # Draw a red line to illustrate the thrust and the thrust vectoring
        vec_line_scale = 6.0 * self.thrust_current_N / self.THRUST_MAX_N
        thrust_vec_line = -np.array([self.thrust_alpha,
                                     self.thrust_beta,
                                     1.0]) * vec_line_scale
        thrust_start_point = [0,0, self.NOZZLE_OFFSET]
        thrust_end_point = [thrust_vec_line[0],
                            thrust_vec_line[1],
                            thrust_vec_line[2]-2.0]
        thrust_color = [1.0, 0.0, 0.0]
        thrust_line_width = 6.0
        if self.interactive:
            if self.debug_line_thrust == -1:
                self.debug_line_thrust = p.addUserDebugLine(thrust_start_point,
                                       thrust_end_point,
                                       lineColorRGB=thrust_color,
                                       parentObjectUniqueId=self.pybullet_body,
                                       parentLinkIndex=self.pybullet_booster_index,
                                       lineWidth=thrust_line_width)
            else:
                self.debug_line_thrust = p.addUserDebugLine(thrust_start_point,
                                       thrust_end_point,
                                       lineColorRGB=thrust_color,
                                       parentObjectUniqueId=self.pybullet_body,
                                       parentLinkIndex=self.pybullet_booster_index,
                                       replaceItemUniqueId=self.debug_line_thrust,
                                       lineWidth=thrust_line_width)


        # <EXTRACT CURRENT STATE FROM PYBULLET>
        position, orientation = p.getBasePositionAndOrientation(
                self.pybullet_body, physicsClientId=self.CLIENT)
        linear_velocity, omega_enu = p.getBaseVelocity(
                self.pybullet_body, physicsClientId=self.CLIENT)
        self.pos_n = np.array([position[0], position[1], position[2]])
        self.vel_n = np.array([linear_velocity[0],
                               linear_velocity[1],
                               linear_velocity[2]])
        # reorder pybullet quaternion to our internal order:
        q_rosbody_to_enu = np.array([orientation[3],
                                     orientation[0],
                                     orientation[1],
                                     orientation[2]])
        self.q = q_rosbody_to_enu
        # transform the body rotation rates that are given in the ENU world
        # frame to the PyRocketCraft body frame.
        R_enu_to_rosbody = quat_to_matrix(quat_invert(q_rosbody_to_enu))
        omega_rosbody = R_enu_to_rosbody @ np.array([omega_enu[0],
                                                     omega_enu[1],
                                                     omega_enu[2]])
        self.omega = omega_rosbody
        # </EXTRACT CURRENT STATE FROM PYBULLET>

    def _update_state(self):
        """
        Internal helper function to update self.state vector based on
        attributes such as self.q, self.pos, etc.
        """
        euler          = quat_to_rpy(self.q)
        self.roll_deg  = np.rad2deg(euler[0])
        self.pitch_deg = np.rad2deg(euler[1])
        self.yaw_deg   = np.rad2deg(euler[2])

        # Produce state vector:
        state_index = 0

        self.state[state_index:(state_index+4)] = self.q
        self.state_cfg["q_index"] = state_index
        state_index += 4

        self.state[state_index:(state_index+3)] = self.omega
        self.state_cfg["omega_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+3)] = self.pos_n
        self.state_cfg["pos_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+3)] = self.vel_n
        self.state_cfg["vel_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+1)] = self.thrust_current_N
        self.state_cfg["thrust_index"] = state_index
        state_index += 1

        self.state[state_index:(state_index+1)] = self.thrust_alpha
        self.state_cfg["thrust_alpha_index"] = state_index
        state_index += 1

        self.state[state_index:(state_index+1)] = self.thrust_beta
        self.state_cfg["thrust_beta_index"] = state_index
        state_index += 1

    def step(self, action: np.array):
        """
        Gym interface step function to simulate the system.
        :param action Control input to the simulation, i.e. motor/rotor
            setpoints between 0 and 1 (actually umin and umax to be precise)
        :return state (state vector), reward (score), done (simulation done?)
        """

        action[1] = np.clip(action[1], self.UMIN, self.UMAX)
        action[2] = np.clip(action[2], self.UMIN, self.UMAX)
        action[0] = np.clip(action[0], self.THRUST_UMIN, self.UMAX) # thrust has a different limit

        done = False
        self.time_sec = self.time_sec + self.dt_sec
        try:
             self._pybullet_physics(action)
        except Exception as e:
            print("pybullet exception:", e)
            done = True

        self.epochs += 1
        self._update_state()

        reward = self._calculate_reward()
        if self.engine_on is False:
            self.time_on_ground_sec += self.dt_sec
            if self.time_on_ground_sec > self.WAIT_ON_GROUND_SEC:
                done = True

        # Stop the non-interactive simulation if the attitude is way off
        if self.interactive is False:
            if np.abs(self.pitch_deg) > 90.0 or np.abs(self.roll_deg) > 90.0:
                reward -= 100.0
                done = True

        info = {}
        return self.state, reward, done, False, info

    def print_state(self):
        """
        Helper function to print the current state vector to stdout
        """

        print("ENU=(%6.2f,%6.2f,%6.2f m) V=(%6.1f,%6.1f,%6.1f m/s) RPY=(%6.1f,%6.1f,%6.1f °) o=(%6.1f,%6.1f,%6.1f °/s) Thrust=%6.1f N alpha=%.1f beta=%.1f" %
              ( self.pos_n[0], self.pos_n[1], self.pos_n[2],
                self.vel_n[0], self.vel_n[1], self.vel_n[2],
                self.roll_deg, self.pitch_deg, self.yaw_deg,
                np.rad2deg(self.omega[0]),
                np.rad2deg(self.omega[1]),
                np.rad2deg(self.omega[2]),
                self.thrust_current_N,
                np.rad2deg(self.thrust_alpha), np.rad2deg(self.thrust_beta)),
                end=" ")
        print("")

    def render(self):
        """
        Gym interface. Render current simulation status.
        """
        if self.interactive:
            self.print_state()

    def _calculate_reward(self):
        """
        Calculate the current reward score.
        """
        # Constants for reward calculation - these may need tuning
        POSITION_WEIGHT = 1.0
        VELOCITY_WEIGHT = 1.5
        ORIENTATION_WEIGHT = 1.0
        FUEL_WEIGHT = 1.0
        MAX_POS_REWARD = 50.0   # Maximum reward for position
        MAX_VEL_REWARD = 50.0   # Maximum reward for velocity
        MAX_ORI_REWARD = 2.0    # Maximum reward for orientation (cos(0) + cos(0))
        MAX_FUEL_REWARD = 50.0  # Maximum reward for fuel use

        max_fuel_used_per_step = (self.THRUST_MAX_N / (self.ISP * self.GRAVITY)) * self.dt_sec
        fuel_used_ratio = self.fuel_consumed_kg / max_fuel_used_per_step
        fuel_used_ratio = np.clip(fuel_used_ratio, 0.0, 1.0)

        # fuel_remaining_ratio = self.TOTAL_FUEL / self.INITIAL_FUEL

        # Thrust reward
        current_thrust = self.thrust_current_N

        cutoff_alt_ratio = 0.1 # above -> fuel reward is high for high fuel and high for low vel, below -> opposite
        # Altitude ratio (1.0 at start, 0.0 at ground) for velocity and fuel rewards
        altitude_ratio = max(0.0, min(1.0, self.pos_n[2] / self.init_height))

        # Calculate the negative distance from the target position (0,0,0)
        target_pos = np.array([0, 0, 0])
        distance = np.linalg.norm(self.pos_n - target_pos)
        distance_reward = MAX_POS_REWARD - POSITION_WEIGHT * distance

        # magnitude of velocity
        velocity_magnitude = np.linalg.norm(self.vel_n)
        
        if altitude_ratio > cutoff_alt_ratio:
            # === HIGH ALTITUDE ===
            # Reward Falling, Reward SAVING fuel
            
            # Downward velocity is good (positive reward)
            velocity_reward = VELOCITY_WEIGHT * (-self.vel_n[2])
            # print("VELOCITY Z:", self.vel_n[2])
            # print("VEL REWARD HIGH ALT:", velocity_reward)
            
            # Low thrust is good
            thrust_reward = MAX_FUEL_REWARD * FUEL_WEIGHT * (1.0 - current_thrust / self.THRUST_MAX_N)

            # Low fuel used is good
            fuel_reward = MAX_FUEL_REWARD * FUEL_WEIGHT * (1.0 - fuel_used_ratio)
                    
        else:
            # === LOW ALTITUDE ===
            # Reward Stopping, Reward BURNING fuel
            
            # Low velocity magnitude is good
            velocity_reward = MAX_VEL_REWARD - (VELOCITY_WEIGHT * velocity_magnitude)
            
            # High thrust is good. (1.0 usage = Max Reward)
            thrust_reward = MAX_FUEL_REWARD * FUEL_WEIGHT * (current_thrust / self.THRUST_MAX_N)

            fuel_reward = MAX_FUEL_REWARD * FUEL_WEIGHT * (fuel_used_ratio)

        # print("FUEL REWARD: ", fuel_reward)
        # Calculate orientation reward
        # Converting degrees to radians for cosine calculation
        roll_rad = np.radians(self.roll_deg)
        pitch_rad = np.radians(self.pitch_deg)
        orientation_reward = ORIENTATION_WEIGHT * (np.cos(roll_rad) + np.cos(pitch_rad))

        # Normalize rewards
        normalized_distance_reward = distance_reward / MAX_POS_REWARD
        normalized_velocity_reward = velocity_reward / MAX_VEL_REWARD
        normalized_orientation_reward = orientation_reward / MAX_ORI_REWARD
        normalized_thrust_reward = thrust_reward / MAX_FUEL_REWARD
        normalized_fuel_reward = fuel_reward / MAX_FUEL_REWARD

        # Total reward
        total_reward = 0.0
        total_reward += ( normalized_distance_reward
                        + normalized_velocity_reward
                        + normalized_orientation_reward 
                        + normalized_thrust_reward
                        + normalized_fuel_reward )
        total_reward *= self.dt_sec # normalize

        # Shut off engine near the ground and give a huge reward bonus for
        # landing upright and with low velocity
        if self.pos_n[2] < self.MIN_GROUND_DIST_M:
            if self.engine_on is True:
                self.engine_on = False
                if self.interactive:
                    print("Engine off at altitude: %.3f (AGL)" % (self.pos_n[2]), " with velocity %.3f m/s" % velocity_magnitude, " and fuel left %.3f kg" % self.TOTAL_FUEL)
                if np.abs(self.roll_deg) < 5.0:
                    total_reward += 128.0
                if np.abs(self.pitch_deg) < 5.0:
                    total_reward += 256.0
                if np.linalg.norm(self.omega) < np.deg2rad(7.0):
                    total_reward += 512.0
                if velocity_magnitude < 0.25:
                    total_reward += 3000.0
                # Bonus for using most of the fuel
                fuel_remaining_ratio = self.TOTAL_FUEL / self.INITIAL_FUEL
                print("REMAINING FUEL RATIO: ", fuel_remaining_ratio)
                if fuel_remaining_ratio < 0.05:  # less than 5% fuel remaining
                    total_reward += 1000.0

        return total_reward

    def _get_total_mass(self, body_id):
        """
        Get total body mass from object tree.
        """
        # Start with the mass of the base link (index -1)
        total_mass = p.getDynamicsInfo(body_id, -1,
                                       physicsClientId=self.CLIENT)[0]

        # Add up the masses of all other links
        num_links = p.getNumJoints(body_id, physicsClientId=self.CLIENT)
        for i in range(num_links):
            total_mass += p.getDynamicsInfo(body_id, i,
                                            physicsClientId=self.CLIENT)[0]

        return total_mass
    
    def _compute_urdf_length(self, urdf_path):
        """
        Parse URDF and compute z-extent (max_z - min_z) and average cylinder radius.
        Returns (length_m, zmin, zmax, avg_cylinder_radius).
        This version prefers the collision tag (where your URDF stores radius).
        """
        import xml.etree.ElementTree as ET
        from collections import deque

        def parse_xyz(attr):
            if attr is None:
                return np.zeros(3)
            return np.array([float(x) for x in attr.split()])

        def rpy_to_rot(roll, pitch, yaw):
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
            Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
            Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
            return Rz @ Ry @ Rx

        def get_geom_info(link_elem):
            # prefer collision (your URDF stores cylinder radius there), then visual, then inertial
            for tag in ("collision", "visual", "inertial"):
                g = link_elem.find(tag)
                if g is None:
                    continue
                origin = g.find("origin")
                geom = g.find("geometry")
                if geom is None:
                    continue
                ox = parse_xyz(origin.get("xyz") if origin is not None else None)
                orpy = parse_xyz(origin.get("rpy") if origin is not None else None)

                cyl = geom.find("cylinder")
                if cyl is not None:
                    length = float(cyl.get("length")) if cyl.get("length") is not None else 0.0
                    # URDF standard attribute is "radius"
                    radius_attr = cyl.get("radius")
                    radius = float(radius_attr) if radius_attr is not None else None
                    return ("cylinder", ox, orpy, {"length": length, "radius": radius})

                box = geom.find("box")
                if box is not None:
                    size = np.array([float(x) for x in box.get("size").split()])
                    return ("box", ox, orpy, {"size": size})

                sph = geom.find("sphere")
                if sph is not None:
                    r = float(sph.get("radius"))
                    return ("sphere", ox, orpy, {"r": r})

            return (None, np.zeros(3), np.zeros(3), {})

        tree = ET.parse(urdf_path)
        root = tree.getroot()
        links = {ln.get("name"): ln for ln in root.findall("link")}

        # build parent->children map and find root link
        children_map = {}
        child_set = set()
        for j in root.findall("joint"):
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            origin = j.find("origin")
            xyz = parse_xyz(origin.get("xyz") if origin is not None else None)
            rpy = parse_xyz(origin.get("rpy") if origin is not None else None)
            children_map.setdefault(parent, []).append((child, xyz, rpy))
            child_set.add(child)

        root_link = None
        for name in links:
            if name not in child_set:
                root_link = name
                break
        if root_link is None:
            raise RuntimeError("could not determine root link in URDF")

        # BFS compute global pose for each link
        link_pose = {root_link: (np.zeros(3), np.eye(3))}
        q = deque([root_link])
        while q:
            parent = q.popleft()
            p_pos, p_rot = link_pose[parent]
            for (child, j_xyz, j_rpy) in children_map.get(parent, []):
                Rj = rpy_to_rot(j_rpy[0], j_rpy[1], j_rpy[2])
                child_pos = p_pos + p_rot @ j_xyz
                child_rot = p_rot @ Rj
                link_pose[child] = (child_pos, child_rot)
                q.append(child)

        z_min = float("inf")
        z_max = float("-inf")
        cylinder_radii = []

        for lname, link in links.items():
            gtype, g_origin, g_rpy, gmeta = get_geom_info(link)
            if gtype is None:
                continue

            # collect explicit radius from geometry (collision preferred above)
            if gtype == "cylinder" and gmeta.get("radius") is not None:
                try:
                    cylinder_radii.append(float(gmeta["radius"]))
                except Exception:
                    pass

            link_pos, link_rot = link_pose.get(lname, (np.zeros(3), np.eye(3)))
            Rg = link_rot @ rpy_to_rot(g_rpy[0], g_rpy[1], g_rpy[2])
            g_center = link_pos + link_rot @ g_origin

            if gtype == "cylinder":
                L = gmeta.get("length", 0.0)
                p_plus = g_center + Rg @ np.array([0,0, L/2.0])
                p_minus = g_center + Rg @ np.array([0,0,-L/2.0])
                z_min = min(z_min, p_plus[2], p_minus[2])
                z_max = max(z_max, p_plus[2], p_minus[2])
            elif gtype == "box":
                sx, sy, sz = gmeta["size"]
                for sx_sign in (-0.5, 0.5):
                    for sy_sign in (-0.5, 0.5):
                        for sz_sign in (-0.5, 0.5):
                            local = np.array([sx_sign*sx, sy_sign*sy, sz_sign*sz])
                            pnt = g_center + Rg @ local
                            z_min = min(z_min, pnt[2])
                            z_max = max(z_max, pnt[2])
            elif gtype == "sphere":
                r = gmeta["r"]
                z_min = min(z_min, g_center[2] - r)
                z_max = max(z_max, g_center[2] + r)

        if z_min == float("inf"):
            raise RuntimeError("no geometries found in URDF")

        avg_radius = float(np.mean(cylinder_radii)) if len(cylinder_radii) > 0 else 0.0
        return float(z_max - z_min), float(z_min), float(z_max), avg_radius
