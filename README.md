Booster Landing Control
-----------------------

Developed for UPenn MEAM 5170 Final Project by Cody Hopkins, Zach Rudder, and Finn Maniscalco. 
This repo was adapted from the [PyRocketCraft](https://github.com/jnz/PyRocketCraft) repo by Jan Zwiener, 
which uses model predictive control (MPC) and neural networks (NN) to land the rocket booster.


Overview
--------

This project implements a full nonlinear trajectory optimization and feedback control pipeline for a simulated rocket booster performing a soft landing.
We use Pseudo-Spectral Collocation (PSC) to compute a globally optimal open-loop trajectory, and a Time-Varying LQR (TVLQR) controller to stabilize and 
track that trajectory in a PyBullet simulation. The system models a rocket with 16 states (quaternion attitude, angular velocity, ENU position, linear velocity, 
thrust magnitude, and thrust vectoring angles), and simulates the dynamics, including drag and fuel usage.


Program structure
-----------------

project-root/
│
├── README.md
├── pyproject.toml             # Dependencies
│
├── src/
│   ├── rocketcraft.py         # Main entry point (run this file)
│   ├── pscpolicy.py           # PSC trajectory optimizer + TVLQR controller
│   ├── basecontrol.py         # Base controller interface  
│   ├── simrocketenv.py        # Physics simulation with gym interface, using pybullet
│   ├── geodetic_toolbox.py    # Helper functions
│   ├── psc_offline_test.py    # Script to solve PSC trajectory offline without sim
│   ├── psc_plot_trajectory.py # Functions to plot PSC trajectory
│   ├── modelrocket.urdf       # URDF of the rocket
│   ├── psc/
│        └── rocket_model.py   # CasADi rocket dynamics model  
└── acados/                    # Required library (not tracked by git)


Block diagram:
--------------

The main function in rocketcraft.py runs the PSC code decoupled from the
physics simulation in a thread. The simulation part is in the simrocketenv file
that is using the OpenAI gym / Gymnasium interface and using pybullet in the
background for the heavy lifting of the physics simulation incl. collision
detection.
The `ctrl_thread_func` will call the PSCPolicy.py.

    ┌───────────────────┐
    │  rocketcraft.py   │
    │  --------------   │   'state' ┌─────────────────────┐    ┌─────────────────┐
    │                   │◄──────────│  simrocketenv.py    │    │ pybullet        │
    │  main()           │           │  ---------------    │──► │ --------        │
    │                   │   'u'     │                     │    │                 │
    │                   │──────────►│  OpenAI gym env.    │    │ Physics engine  │
    └───────┬───────────┘           │  Physics simulation │    │ and GUI         │
            │      ▲                └─────────────────────┘    └─────────────────┘
            │      │
    'state' │      │ 'u'
            │      │
            ▼      │
    ┌───────────────────┐
    │                   │
    │ Controller Thread │ 'state' >
    │ ctrl_thread_func()│ < 'u'  ┌─────────────────┐       ┌─────────────────┐
    │                   │◄───-──►│ PSCPolicy.py    │◄────► │ rocket_model.py │
    │                   │        │ --------------  │       │ --------------  │
    └───────────────────┘        │                 │       │                 │
                                 │ PSC controller  │       │ Model and       │
                                 │ u = next(state) │       │ dynamics        │
                                 └─────────────────┘       └───┬─────────────┘
                                                               │   ┌────────────────┐
                                                               └─► │ acados         │
                                                                   │ ------         │
                                                                   │                │
                                                                   │ Auto generated │
                                                                   │ C-code         │
                                                                   └────────────────┘


Installation on Linux and macOS
-------------------------------

## 1. Install Prerequisites
- Git  
- C/C++ toolchain:
  - macOS: `xcode-select --install`
  - Linux: `sudo apt install build-essential`
- Conda (Miniforge recommended)

Verify installation:

```
conda --version
```

## 2. Clone Repository + Initialize Submodules

```bash
git clone <REPO_URL>
cd Booster-Landing-Control
git submodule update --init --recursive
```

## 3. Create Conda Environment

```bash
conda create -n rocket python=3.9
conda activate rocket
```

## 4. Install Core Python Dependencies

```bash
conda install -c conda-forge pybullet gymnasium numpy scipy pytorch stable-baselines3
```

## 5. Build acados

```bash
cd acados
cmake -DACADOS_WITH_QPOASES=OFF .
make install -j4
cd ..
```

## 6. Install Project (Editable Mode)

```bash
pip install -e . --no-deps
pip install -e acados/interfaces/acados_template
```

## 7. Set Environment Variables for acados

### macOS
```bash
export ACADOS_SOURCE_DIR=/path/to/Booster-Landing-Control/acados
export DYLD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$DYLD_LIBRARY_PATH
```

### Linux
```bash
export ACADOS_SOURCE_DIR=/path/to/Booster-Landing-Control/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
```

Add to your shell config (`~/.zshrc` or `~/.bashrc`) and reload:

```bash
source ~/.zshrc
```


Running the Simulation
----------------------

Run it from the project root:

```bash
python src/rocketcraft.py
```

This will:
1. Load the PyBullet rocket environment  
2. Build the PSC nonlinear program  
3. Solve the trajectory using IPOPT  
4. Build TVLQR gains  
5. Execute the closed-loop simulation  


Coordinate Frames
-----------------

pybullet is using:

 - World Frame (enu) East/North/Up(ENU): X = East, Y = North, Z = Up
 - Body Frame (rosbody), X = Forward, Y = Left, Z = Up
