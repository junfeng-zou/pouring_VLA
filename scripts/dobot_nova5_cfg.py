"""
DOBOT Nova 5 - Isaac Lab Articulation Configuration
====================================================

This module provides the ArticulationCfg for the DOBOT Nova 5 robot arm,
ready to be used with Isaac Lab's Articulation class for simulation and
reinforcement learning tasks.

Usage:
    from dobot_nova5_cfg import DOBOT_NOVA5_CFG

    # In your Isaac Lab environment setup:
    robot = Articulation(cfg=DOBOT_NOVA5_CFG)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ── Resolve the USD asset path ─────────────────────────────────────────────
# Relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_USD_PATH = os.path.join(
    _PROJECT_DIR,
    "assets",
    "DOBOT Nova 5-20221011.SLDASM",
    "usd",
    "dobot_nova5.usd",
)


# ── Joint names ────────────────────────────────────────────────────────────
# DOBOT Nova 5 has 6 revolute joints
JOINT_NAMES = [
    "joint_1",  # Base rotation      (±360°, 150 Nm)
    "joint_2",  # Shoulder            (±180°, 150 Nm)
    "joint_3",  # Elbow               (±160°, 150 Nm)
    "joint_4",  # Wrist 1 rotation    (±360°, 28 Nm)
    "joint_5",  # Wrist 2 bend        (±360°, 28 Nm)
    "joint_6",  # Wrist 3 rotation    (±360°, 28 Nm)
]


# ── Articulation Configuration ─────────────────────────────────────────────
DOBOT_NOVA5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Default joint positions (all zeros = home position)
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
        },
        # Default joint velocities
        joint_vel={
            "joint_.*": 0.0,
        },
    ),
    actuators={
        # Large joints (shoulder/elbow): higher gains
        "arm_large": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-3]"],
            effort_limit=150.0,
            velocity_limit=1.7453,  # 100 deg/s
            stiffness=400.0,
            damping=40.0,
        ),
        # Wrist joints: lower gains
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_[4-6]"],
            effort_limit=28.0,
            velocity_limit=1.7453,
            stiffness=200.0,
            damping=20.0,
        ),
    },
)
"""DOBOT Nova 5 robot arm configuration for Isaac Lab.

Attributes:
    spawn: USD file configuration with rigid body and articulation root properties.
    init_state: Initial joint positions and velocities (home position = all zeros).
    actuators: Two actuator groups:
        - ``arm_large``: joints 1-3 (shoulder/elbow), effort_limit=150 Nm, Kp=400, Kd=40
        - ``arm_wrist``: joints 4-6 (wrist), effort_limit=28 Nm, Kp=200, Kd=20
"""
