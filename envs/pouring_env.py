"""
DOBOT Nova 5 — Pouring Simulation Environment
===============================================

A reusable Isaac Lab DirectRLEnv for VLA water-pouring tasks.

Scene:
    - DOBOT Nova 5 robot arm (from converted USD)
    - Table (rigid cuboid)
    - Cup (rigid open-top cylinder on the table)
    - Bottle (rigid cylinder, fixed-jointed to robot end-effector link_6)
    - Water (small rigid spheres, initially inside the bottle)

Usage:
    from envs.pouring_env import PouringEnv, PouringEnvCfg
    cfg = PouringEnvCfg()
    env = PouringEnv(cfg)
"""

from __future__ import annotations

import os
import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ENVS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_ENVS_DIR)
_USD_PATH = os.path.join(
    _PROJECT_DIR,
    "assets",
    "DOBOT Nova 5-20221011.SLDASM",
    "usd",
    "dobot_nova5.usd",
)

# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------

# -- Water (rigid sphere) parameters ----------------------------------------
NUM_WATER_SPHERES = 80
WATER_SPHERE_RADIUS = 0.008  # 8 mm
WALL_THICKNESS = 0.004       # container wall thickness
GRIPPER_MAX_OPEN = 0.055     # max finger travel (must match URDF upper limit)

# -- Geometry ---------------------------------------------------------------
TABLE_SIZE = (2.0, 1.6, 0.75)  # length, width, height (X length doubled)
# Keep +X edge unchanged while extending only toward -X:
# old X span [-0.3, 0.9] -> new X span [-1.5, 0.9]
TABLE_POS = (-0.3, 0.0, TABLE_SIZE[2] / 2.0)  # robot at x=0, table extends from x=-1.5 to x=0.9

CUP_RADIUS = 0.04
CUP_HEIGHT = 0.10
CUP_POS = (-0.224, -0.410, TABLE_SIZE[2] + CUP_HEIGHT / 2.0)

BOTTLE_RADIUS = 0.03
BOTTLE_HEIGHT = 0.20
BOTTLE_POS = (-0.224, -0.290, TABLE_SIZE[2] + BOTTLE_HEIGHT / 2.0)  # near cup, ~12cm offset

# Second bottle (Sprite-like green), decorative / distractor; no liquid inside.
SPRITE_BOTTLE_POS = (-0.08, -0.52, TABLE_SIZE[2] + BOTTLE_HEIGHT / 2.0)
SPRITE_BOTTLE_COLOR = (0.22, 0.78, 0.32)  # lime / Sprite-like green

# Tabletop XY bounds (local frame) for random placement — inside table with margin.
_TABLE_X_HALF = TABLE_SIZE[0] * 0.5
_TABLE_Y_HALF = TABLE_SIZE[1] * 0.5
_TABLE_MARGIN = 0.10
TABLE_XY_XMIN = TABLE_POS[0] - _TABLE_X_HALF + _TABLE_MARGIN + CUP_RADIUS
TABLE_XY_XMAX = TABLE_POS[0] + _TABLE_X_HALF - _TABLE_MARGIN - CUP_RADIUS
TABLE_XY_YMIN = TABLE_POS[1] - _TABLE_Y_HALF + _TABLE_MARGIN + CUP_RADIUS
TABLE_XY_YMAX = TABLE_POS[1] + _TABLE_Y_HALF - _TABLE_MARGIN - CUP_RADIUS

# IK joint sign convention correction (joint_1..joint_6).
# User-observed convention: all reversed except joint_3.
# We apply this in IK space so Cartesian control follows the expected positive directions.
IK_JOINT_SIGN = (-1.0, -1.0, 1.0, -1.0, -1.0, -1.0)


def _quat_wxyz_to_rotmat_torch(q: torch.Tensor) -> torch.Tensor:
    """Convert batched quaternions (w, x, y, z) to rotation matrices."""
    w, x, y, z = q.unbind(dim=-1)
    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _quat_wxyz_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """Single quaternion (w, x, y, z) → 3×3 rotation matrix (numpy)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _debug_make_arrow_curve(stage, path: str, color_rgb: tuple[float, float, float]):
    """One line strip for EE axis debug (UsdGeom.BasisCurves)."""
    from pxr import Gf, UsdGeom

    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)])
    curve.CreateWidthsAttr([0.006])
    curve.CreateDisplayColorAttr(
        [Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))]
    )
    return curve


def _debug_create_ee_frame_curves(stage, parent_path: str) -> dict:
    """RGB 三轴箭头：X 红、Y 绿、Z 蓝（末端 link_6 体坐标系在世界里）。"""
    from pxr import UsdGeom

    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Xform.Define(stage, parent_path)
    curves: dict[str, dict[str, object]] = {}
    for stem, rgb in (
        ("axis_x", (1.0, 0.0, 0.0)),
        ("axis_y", (0.0, 1.0, 0.0)),
        ("axis_z", (0.0, 0.0, 1.0)),
    ):
        base = f"{parent_path}/{stem}"
        curves[stem] = {
            "shaft": _debug_make_arrow_curve(stage, f"{base}_shaft", rgb),
            "head_l": _debug_make_arrow_curve(stage, f"{base}_head_l", rgb),
            "head_r": _debug_make_arrow_curve(stage, f"{base}_head_r", rgb),
        }
    return curves


def _debug_update_ee_frame_curves(
    curves: dict,
    origin: np.ndarray,
    r_world_from_ee: np.ndarray,
    axis_length: float,
) -> None:
    from pxr import Gf

    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    side_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_dirs = {
        "axis_x": np.array([1.0, 0.0, 0.0]),
        "axis_y": np.array([0.0, 1.0, 0.0]),
        "axis_z": np.array([0.0, 0.0, 1.0]),
    }
    for stem, local in local_dirs.items():
        if stem not in curves:
            continue
        axis_dir = r_world_from_ee @ local
        n = float(np.linalg.norm(axis_dir))
        if n < 1e-9:
            continue
        axis_dir = axis_dir / n
        o = origin.astype(np.float64, copy=False)
        end = o + axis_length * axis_dir
        tangent = np.cross(axis_dir, up_hint)
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.cross(axis_dir, side_hint)
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-8)
        head_len = axis_length * 0.22
        head_w = axis_length * 0.10
        left = end - head_len * axis_dir + head_w * tangent
        right = end - head_len * axis_dir - head_w * tangent
        c = curves[stem]
        c["shaft"].GetPointsAttr().Set(
            [
                Gf.Vec3f(float(o[0]), float(o[1]), float(o[2])),
                Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
            ]
        )
        c["head_l"].GetPointsAttr().Set(
            [
                Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
                Gf.Vec3f(float(left[0]), float(left[1]), float(left[2])),
            ]
        )
        c["head_r"].GetPointsAttr().Set(
            [
                Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
                Gf.Vec3f(float(right[0]), float(right[1]), float(right[2])),
            ]
        )


@configclass
class PouringEnvCfg(DirectRLEnvCfg):
    """Configuration for the DOBOT Nova 5 pouring environment."""

    # -- env --
    episode_length_s = 240.0
    decimation = 2
    action_space = 7       # 3 EE position deltas + 3 EE orientation deltas + 1 gripper
    observation_space = 20  # 6 jpos + 6 jvel + 2 gripper_pos + 3 ee_pos + 3 cup_pos
    state_space = 0

    # -- simulation --
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.1,
        ),
    )

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # -- data collection camera (mounted relative to robot base frame) --
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/MainCamera",
        spawn=sim_utils.PinholeCameraCfg(
            # Approximate Orbbec Femto Bolt RGB intrinsics:
            # HFOV ~79-80 deg with 16:9 aspect.
            focal_length=2.80,
            horizontal_aperture=4.80,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.8, -0.56, 0.34),
            # Yaw +45deg in parent (robot-base) frame:
            # camera heading aligns with the bisector of +X and +Y.
            # rot=(1, 0, 0, 0.),
            rot=(0.9811, -0.0000, 0.1934, 0.0000),
            convention="world",
        ),
        # Femto Bolt RGB commonly used at 1280x720.
        width=1280,
        height=720,
        data_types=["rgb", "distance_to_image_plane"],
        # 0 = 每仿真步刷新；固定 30Hz 时可能与策略读帧相位耦合导致长时间读到同一 buffer
        update_period=0.0,
    )

    # Operator guidance camera (overhead angle)
    cam_side = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/CamSide",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.80,
            horizontal_aperture=4.80,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0, -0.8, 0.5),
            # rot=(0.5897, -0.1237, 0.0920, 0.7928),
            rot=(0.5742, -0.1824, 0.1713, 0.7795),
            convention="world",
        ),
        width=1280,
        height=720,
        data_types=["rgb"],
        update_period=1.0 / 30.0,
    )

    # -- robot --
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_SIZE[2]),  # on the table surface
            joint_pos={
                "joint_1": -0.6020,
                "joint_2": -0.2310,
                "joint_3": 2.1043,   # elbow up
                "joint_4": 0.7646,
                "joint_5": 1.5708,  # wrist down
                "joint_6": 4.2412e-04,
                # "joint_1": 0,
                # "joint_2": 0,
                # "joint_3": 0,   # elbow up
                # "joint_4": 0,
                # "joint_5": 0,
                # "joint_6": 0,
                "finger_left_joint": GRIPPER_MAX_OPEN,   # open
                "finger_right_joint": GRIPPER_MAX_OPEN,  # open
            },
        ),
        actuators={
            "arm_large": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-3]"],
                effort_limit=150.0,
                velocity_limit=1.7453,
                stiffness=400.0,
                damping=40.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint_[4-6]"],
                effort_limit=28.0,
                velocity_limit=1.7453,
                stiffness=200.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint"],
                effort_limit=60.0,
                velocity_limit=0.2,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    # -- table (rigid cuboid) --
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2),  # wood-like
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TABLE_POS,
        ),
    )

    # Cup and bottle are created as hollow containers in _setup_scene via USD API
    # (CylinderCfg only creates solid cylinders)

    # -- water spheres --
    water_sphere = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Water_.*",
        spawn=sim_utils.SphereCfg(
            radius=WATER_SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.15, 0.1),  # brown-red water
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BOTTLE_POS,  # start inside the bottle
        ),
    )

    # -- reward scales --
    water_in_cup_reward_scale = 10.0
    action_penalty_scale = 0.01
    bottle_orientation_reward_scale = 1.0

    # -- action scale --
    pos_action_scale = 0.005   # meters per step (for EE position)
    rot_action_scale = 0.01    # radians per step (for EE orientation)
    teleop_delta_frame = "ee"  # "ee": deltas interpreted in end-effector frame

    # -- debug visualization (USD BasisCurves; link_6 body frame: X红 Y绿 Z蓝) --
    debug_visualize_ee_frame: bool = False
    debug_ee_frame_axis_length: float = 0.15  # meters
    debug_ee_frame_prim_path: str = "/World/DebugEEFrame"

    # -- position randomization (for data diversity) --
    randomize_positions = True
    # Moderate patch around default layout (~-0.22, -0.35) so MainCamera still sees cup + both bottles.
    # Box diagonal ~0.54 m — enough for max_cup_bottle_dist=0.50. Clamped to TABLE_XY_* in _sample_*.
    cup_pos_x_range = (-0.43, -0.02)
    cup_pos_y_range = (-0.58, -0.22)
    bottle_pos_x_range = (-0.43, -0.02)
    bottle_pos_y_range = (-0.58, -0.22)
    sprite_bottle_pos_x_range = (-0.43, -0.02)
    sprite_bottle_pos_y_range = (-0.58, -0.22)
    min_cup_bottle_dist = 0.10
    max_cup_bottle_dist = 0.50
    # Sprite bottle must not overlap cup / cola bottle centers too closely.
    min_sprite_sep_dist = 0.12


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PouringEnv(DirectRLEnv):
    """DOBOT Nova 5 pouring environment.

    The robot must pour water (rigid spheres) from a bottle into a cup on a table.
    """

    cfg: PouringEnvCfg

    def __init__(self, cfg: PouringEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        # Joint targets (position control)
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        # End-effector link index
        self.ee_link_idx = self._robot.find_bodies("link_6")[0][0]

        # -- Differential IK Controller --
        self._ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self._ik_controller = DifferentialIKController(
            self._ik_cfg, num_envs=cfg.scene.num_envs, device=self.device
        )

        # Resolve robot entity for IK (arm joints only, 6-DoF)
        self._robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["joint_[1-6]"],
            body_names=["link_6"],
        )
        self._robot_entity_cfg.resolve(self.scene)
        # Jacobian frame index: for fixed-base, body_id - 1
        if self._robot.is_fixed_base:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0] - 1
        else:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0]

        # Cup position (local coords, updated on reset if randomization is enabled)
        self.cup_pos_local = torch.tensor(
            [CUP_POS] * self.num_envs, device=self.device, dtype=torch.float32
        )
        # Bottle position (local coords, updated on reset)
        self.bottle_pos_local = torch.tensor(
            [BOTTLE_POS] * self.num_envs, device=self.device, dtype=torch.float32
        )
        self.sprite_bottle_pos_local = torch.tensor(
            [SPRITE_BOTTLE_POS] * self.num_envs, device=self.device, dtype=torch.float32
        )

        # Water tracking
        self.num_water = NUM_WATER_SPHERES
        self.water_in_cup_count = torch.zeros(self.num_envs, device=self.device)

        # PhysX view for bottle — created lazily on first reset
        self._bottle_physx_view = None

        # Optional: RGB axis arrows at end-effector (link_6)
        self._ee_frame_curves: dict | None = None
        if cfg.debug_visualize_ee_frame:
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                self._ee_frame_curves = _debug_create_ee_frame_curves(
                    stage, cfg.debug_ee_frame_prim_path
                )
            except Exception as exc:
                self._ee_frame_curves = None
                print(f"[PouringEnv] debug_visualize_ee_frame 初始化失败: {exc}")

    def _setup_scene(self):
        """Create the full scene: robot, table, cup, bottle, water, lights."""
        # -- Robot --
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # -- Table --
        self._table = RigidObject(self.cfg.table)
        self.scene.rigid_objects["table"] = self._table

        # -- Camera (third-person, Femto Bolt) --
        from isaaclab.sensors import Camera
        self._camera = Camera(self.cfg.camera)
        self.scene.sensors["camera"] = self._camera

        # -- Operator guidance cameras (side view only) --
        from isaaclab.sensors import Camera
        self._cam_side = Camera(self.cfg.cam_side)
        self.scene.sensors["cam_side"] = self._cam_side

        # -- Cup and Bottle (hollow containers via USD API) --
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, Gf
        stage = omni.usd.get_context().get_stage()

        self._create_hollow_container(
            stage, "/World/envs/env_0/Cup",
            radius=CUP_RADIUS, height=CUP_HEIGHT,
            thickness=WALL_THICKNESS,
            color=(0.9, 0.9, 0.9),
            position=CUP_POS,
        )
        self._create_hollow_container(
            stage, "/World/envs/env_0/Bottle",
            radius=BOTTLE_RADIUS, height=BOTTLE_HEIGHT,
            thickness=WALL_THICKNESS,
            color=(0.12, 0.05, 0.02),  # cola-like very dark brown
            position=BOTTLE_POS,
            kinematic=False,  # dynamic — can be grasped and moved
            mass=0.3,         # 300g bottle
        )
        self._bottle_prim_path = "/World/envs/env_0/Bottle"

        self._create_hollow_container(
            stage, "/World/envs/env_0/BottleSprite",
            radius=BOTTLE_RADIUS, height=BOTTLE_HEIGHT,
            thickness=WALL_THICKNESS,
            color=SPRITE_BOTTLE_COLOR,
            position=SPRITE_BOTTLE_POS,
            kinematic=False,
            mass=0.3,
        )

        # Wrap cup and bottle as RigidObjects for physics-level repositioning.
        # spawn=None because prims already exist from _create_hollow_container.
        cup_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cup",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=CUP_POS),
        )
        self._cup_obj = RigidObject(cup_cfg)
        self.scene.rigid_objects["cup"] = self._cup_obj

        bottle_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bottle",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=BOTTLE_POS),
        )
        self._bottle_obj = RigidObject(bottle_cfg)
        self.scene.rigid_objects["bottle"] = self._bottle_obj

        sprite_bottle_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/BottleSprite",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=SPRITE_BOTTLE_POS),
        )
        self._sprite_bottle_obj = RigidObject(sprite_bottle_cfg)
        self.scene.rigid_objects["bottle_sprite"] = self._sprite_bottle_obj

        # -- Water spheres --
        # Concentric-ring packing: 1 center + 6 ring = 7 balls per layer.
        # This keeps all 80 spheres inside the bottle.
        _inner_r = BOTTLE_RADIUS - WALL_THICKNESS - WATER_SPHERE_RADIUS  # ~0.018
        _n_ring = 6
        _bpl = 1 + _n_ring  # balls per layer = 7
        _layer_sp = WATER_SPHERE_RADIUS * 2.0  # vertical layer spacing = diameter
        _bx, _by, _bz = BOTTLE_POS
        _bot_z = _bz - BOTTLE_HEIGHT / 2.0 + WALL_THICKNESS + WATER_SPHERE_RADIUS

        self._water_spheres: list[RigidObject] = []
        for i in range(NUM_WATER_SPHERES):
            layer = i // _bpl
            pos_in_layer = i % _bpl
            sz = _bot_z + layer * _layer_sp
            if pos_in_layer == 0:
                dx, dy = 0.0, 0.0
            else:
                angle = 2 * math.pi * (pos_in_layer - 1) / _n_ring
                if layer % 2 == 1:  # offset alternate layers for hex packing
                    angle += math.pi / _n_ring
                dx = _inner_r * math.cos(angle)
                dy = _inner_r * math.sin(angle)

            water_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Water_{i:03d}",
                spawn=self.cfg.water_sphere.spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_bx + dx, _by + dy, sz),
                ),
            )
            water_obj = RigidObject(water_cfg)
            self._water_spheres.append(water_obj)
            self.scene.rigid_objects[f"water_{i:03d}"] = water_obj

        # -- Ground plane (manual, no Nucleus dependency) --
        ground_prim = stage.DefinePrim("/World/ground", "Xform")
        plane_prim = UsdGeom.Mesh.Define(stage, "/World/ground/plane")
        plane_prim.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
        plane_prim.CreateFaceVertexCountsAttr([4])
        plane_prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        plane_prim.CreateNormalsAttr([(0, 0, 1)] * 4)
        UsdPhysics.CollisionAPI.Apply(plane_prim.GetPrim())
        plane_prim.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])

        # -- Clone envs --
        self.scene.clone_environments(copy_from_source=False)

        # -- Lights --
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    # ----- Hollow container helper -------------------------------------------

    @staticmethod
    def _create_hollow_container(
        stage, prim_path, radius, height, thickness, color, position,
        num_segments=24, kinematic=True, mass=None,
    ):
        """Create a hollow open-top cylinder from compound collision shapes.

        Structure: 1 bottom disk + N thin wall segments arranged in a ring.

        Args:
            kinematic: If True, container is fixed in space.
                       If False, container is a dynamic rigid body (can be moved/grasped).
            mass: Mass in kg (only used when kinematic=False).
        """
        from pxr import UsdGeom, UsdPhysics, Gf

        # Root Xform with position
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))

        # Rigid body
        rb = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rb.CreateKinematicEnabledAttr(kinematic)

        # Mass (for dynamic bodies)
        if not kinematic and mass is not None:
            mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
            mass_api.CreateMassAttr(mass)

        # --- Friction material for collision surfaces ---
        mat_path = f"{prim_path}/friction_material"
        UsdShade = __import__("pxr", fromlist=["UsdShade"]).UsdShade
        mat = UsdShade.Material.Define(stage, mat_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(1.0)
        phys_mat.CreateDynamicFrictionAttr(1.0)
        phys_mat.CreateRestitutionAttr(0.0)

        def _apply_friction(prim):
            """Bind friction material to a collision prim."""
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

        # --- Bottom plate (thin cylinder) ---
        bottom = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
        bottom.CreateRadiusAttr(radius)
        bottom.CreateHeightAttr(thickness)
        bottom.CreateAxisAttr("Z")
        bottom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
        _apply_friction(bottom.GetPrim())
        UsdGeom.Xformable(bottom.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0, 0, -height / 2.0 + thickness / 2.0)
        )

        # --- Wall segments (thin boxes tangent to circle) ---
        seg_angle = 2 * math.pi / num_segments
        # Use tan so adjacent segments slightly overlap → no gaps
        seg_width = 2 * radius * math.tan(seg_angle / 2.0)

        for i in range(num_segments):
            angle = seg_angle * i
            cx = radius * math.cos(angle)
            cy = radius * math.sin(angle)

            seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{i:02d}")
            seg.CreateSizeAttr(1.0)  # unit cube, scaled below
            seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
            _apply_friction(seg.GetPrim())

            sx = UsdGeom.Xformable(seg.GetPrim())
            sx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0))
            sx.AddRotateZOp().Set(math.degrees(angle))
            # X = thickness (radial), Y = seg_width (tangential), Z = height
            sx.AddScaleOp().Set(Gf.Vec3f(thickness, seg_width, height))

    # ----- Pre-physics -------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Convert Cartesian EE actions to joint position targets via differential IK.

        Actions: 7-dim = [dx, dy, dz, droll, dpitch, dyaw, gripper].
        - [0:3] — EE position delta in root frame (meters)
        - [3:6] — EE orientation delta as axis-angle in root frame (radians)
        - [6]   — gripper command: +1 = open, -1 = closed
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Scale Cartesian deltas
        cart_delta = torch.zeros(self.num_envs, 6, device=self.device)
        cart_delta[:, 0:3] = self.actions[:, 0:3] * self.cfg.pos_action_scale
        cart_delta[:, 3:6] = self.actions[:, 3:6] * self.cfg.rot_action_scale

        # Only update arm targets when there is actual Cartesian input.
        # Otherwise keep previous targets so PD controller holds position against gravity.
        has_input = cart_delta.abs().sum(dim=1) > 1e-8  # (num_envs,)

        if has_input.any():
            # Get current EE pose in root frame
            ee_pose_w = self._robot.data.body_pose_w[:, self._robot_entity_cfg.body_ids[0]]
            root_pose_w = self._robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )

            # Set IK command (relative delta).
            # For teleoperation, interpret deltas in EE frame and convert to base frame.
            if self.cfg.teleop_delta_frame == "ee":
                R_ee_to_base = _quat_wxyz_to_rotmat_torch(ee_quat_b)
                dpos_base = torch.bmm(R_ee_to_base, cart_delta[:, 0:3].unsqueeze(-1)).squeeze(-1)
                drot_base = torch.bmm(R_ee_to_base, cart_delta[:, 3:6].unsqueeze(-1)).squeeze(-1)
                cart_delta_cmd = torch.cat([dpos_base, drot_base], dim=-1)
            else:
                cart_delta_cmd = cart_delta
            self._ik_controller.set_command(cart_delta_cmd, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

            # Get Jacobian and current arm joint positions
            jacobian = self._robot.root_physx_view.get_jacobians()[
                :, self._ee_jacobi_idx, :, self._robot_entity_cfg.joint_ids
            ]
            arm_joint_pos = self._robot.data.joint_pos[:, self._robot_entity_cfg.joint_ids]

            # Convert to corrected joint-sign convention:
            # q_conv = S q_raw,  J_conv = J_raw S, where S=diag(IK_JOINT_SIGN).
            sign = torch.tensor(IK_JOINT_SIGN, device=self.device, dtype=arm_joint_pos.dtype).view(1, -1)
            jacobian_conv = jacobian * sign.view(1, 1, -1)
            arm_joint_pos_conv = arm_joint_pos * sign

            # Compute IK: returns actual_pos + delta
            arm_joint_targets_conv = self._ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian_conv, arm_joint_pos_conv
            )

            # Extract pure delta and apply to PREVIOUS TARGETS (not actual pos).
            # This prevents "accepting" gravity drift on each input step.
            joint_ids = self._robot_entity_cfg.joint_ids
            ik_delta_conv = arm_joint_targets_conv - arm_joint_pos_conv  # pure IK delta (conv-space)
            ik_delta = ik_delta_conv * sign  # map back to raw joint convention
            for i in range(self.num_envs):
                if has_input[i]:
                    self.robot_dof_targets[i, joint_ids] += ik_delta[i]

        # Gripper (index 6): single action controls both fingers
        # Map action [-1, 1] to [0, GRIPPER_MAX_OPEN] (closed to open)
        gripper_cmd = (self.actions[:, 6:7] + 1.0) / 2.0 * GRIPPER_MAX_OPEN
        self.robot_dof_targets[:, 6] = gripper_cmd[:, 0]   # finger_left
        self.robot_dof_targets[:, 7] = gripper_cmd[:, 0]   # finger_right

        # Clamp all to joint limits
        self.robot_dof_targets[:] = torch.clamp(
            self.robot_dof_targets,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

    def _apply_action(self):
        """Send joint position targets to the robot."""
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ----- Post-physics ------------------------------------------------------

    def _get_observations(self) -> dict:
        """Observation: arm_jpos(6) + arm_jvel(6) + gripper_pos(2) + EE_pos(3) + cup_pos(3) = 20."""
        if self._ee_frame_curves is not None:
            try:
                pos_w = self._robot.data.body_pos_w[0, self.ee_link_idx].detach().cpu().numpy()
                quat_w = self._robot.data.body_quat_w[0, self.ee_link_idx].detach().cpu().numpy()
                R = _quat_wxyz_to_rotmat_np(quat_w)
                _debug_update_ee_frame_curves(
                    self._ee_frame_curves,
                    pos_w,
                    R,
                    float(self.cfg.debug_ee_frame_axis_length),
                )
            except Exception:
                pass

        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_link_idx]

        obs = torch.cat([
            joint_pos[:, :6],                    # (N, 6)
            joint_vel[:, :6] * 0.1,              # (N, 6)
            joint_pos[:, 6:8],                   # (N, 2)
            ee_pos_w - self.scene.env_origins,   # (N, 3)
            self.cup_pos_local,                  # (N, 3) — randomized per reset
        ], dim=-1)

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        """Reward based on water spheres reaching the cup region."""
        cup_center_xy = self.cup_pos_local[:, :2]  # (N, 2) — randomized
        cup_top_z = self.cup_pos_local[:, 2] + CUP_HEIGHT / 2.0  # (N,)

        water_in_cup = torch.zeros(self.num_envs, device=self.device)
        for water_obj in self._water_spheres:
            w_pos = water_obj.data.root_pos_w
            w_pos_local = w_pos - self.scene.env_origins

            dist_xy = torch.norm(w_pos_local[:, :2] - cup_center_xy, dim=-1)
            in_cup_xy = dist_xy < CUP_RADIUS
            in_cup_z = (w_pos_local[:, 2] > TABLE_SIZE[2]) & (w_pos_local[:, 2] < cup_top_z + 0.05)
            in_cup = in_cup_xy & in_cup_z
            water_in_cup += in_cup.float()

        self.water_in_cup_count = water_in_cup

        # Reward
        water_reward = water_in_cup / NUM_WATER_SPHERES  # normalized [0, 1]
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        rewards = (
            self.cfg.water_in_cup_reward_scale * water_reward
            - self.cfg.action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "water_in_cup": water_in_cup.mean(),
            "water_reward": (self.cfg.water_in_cup_reward_scale * water_reward).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
        }

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode terminates if timeout."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments with optional position randomization."""
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Reset robot joints
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.05, 0.05,
            (n, self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # --- Randomize or use default cup / bottle positions ---
        table_z = TABLE_SIZE[2]
        if self.cfg.randomize_positions:
            cup_xy, bottle_xy, sprite_xy = self._sample_cup_bottle_sprite_positions(n)
        else:
            cup_xy = torch.tensor([[CUP_POS[0], CUP_POS[1]]], device=self.device).repeat(n, 1)
            bottle_xy = torch.tensor([[BOTTLE_POS[0], BOTTLE_POS[1]]], device=self.device).repeat(n, 1)
            sprite_xy = torch.tensor([[SPRITE_BOTTLE_POS[0], SPRITE_BOTTLE_POS[1]]], device=self.device).repeat(n, 1)

        # Update stored local positions
        self.cup_pos_local[env_ids, 0] = cup_xy[:, 0]
        self.cup_pos_local[env_ids, 1] = cup_xy[:, 1]
        self.cup_pos_local[env_ids, 2] = table_z + CUP_HEIGHT / 2.0

        self.bottle_pos_local[env_ids, 0] = bottle_xy[:, 0]
        self.bottle_pos_local[env_ids, 1] = bottle_xy[:, 1]
        self.bottle_pos_local[env_ids, 2] = table_z + BOTTLE_HEIGHT / 2.0

        self.sprite_bottle_pos_local[env_ids, 0] = sprite_xy[:, 0]
        self.sprite_bottle_pos_local[env_ids, 1] = sprite_xy[:, 1]
        self.sprite_bottle_pos_local[env_ids, 2] = table_z + BOTTLE_HEIGHT / 2.0

        # --- Reset cup (kinematic) ---
        cup_pos_w = self.cup_pos_local[env_ids].clone()
        cup_pos_w += self.scene.env_origins[env_ids]
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(n, 1)
        self._cup_obj.write_root_pose_to_sim(
            torch.cat([cup_pos_w, identity_quat], dim=-1), env_ids=env_ids
        )

        # --- Reset bottle (dynamic) ---
        bottle_pos_w = self.bottle_pos_local[env_ids].clone()
        bottle_pos_w += self.scene.env_origins[env_ids]
        bottle_vel = torch.zeros((n, 6), device=self.device)
        self._bottle_obj.write_root_pose_to_sim(
            torch.cat([bottle_pos_w, identity_quat], dim=-1), env_ids=env_ids
        )
        self._bottle_obj.write_root_velocity_to_sim(bottle_vel, env_ids=env_ids)

        # --- Reset Sprite bottle (dynamic, empty) ---
        sprite_pos_w = self.sprite_bottle_pos_local[env_ids].clone()
        sprite_pos_w += self.scene.env_origins[env_ids]
        self._sprite_bottle_obj.write_root_pose_to_sim(
            torch.cat([sprite_pos_w, identity_quat], dim=-1), env_ids=env_ids
        )
        self._sprite_bottle_obj.write_root_velocity_to_sim(bottle_vel, env_ids=env_ids)

        # --- Reset water spheres inside the new bottle position ---
        inner_r = BOTTLE_RADIUS - WALL_THICKNESS - WATER_SPHERE_RADIUS
        n_ring = 6
        balls_per_layer = 1 + n_ring
        layer_spacing = WATER_SPHERE_RADIUS * 2.0
        zero_vel = torch.zeros((n, 6), device=self.device)

        for env_local_idx in range(n):
            eid = env_ids[env_local_idx]
            bx = self.bottle_pos_local[eid, 0].item()
            by = self.bottle_pos_local[eid, 1].item()
            bz = self.bottle_pos_local[eid, 2].item()
            bot_z = bz - BOTTLE_HEIGHT / 2.0 + WALL_THICKNESS + WATER_SPHERE_RADIUS

            for i, water_obj in enumerate(self._water_spheres):
                layer = i // balls_per_layer
                pos_in_layer = i % balls_per_layer
                sz = bot_z + layer * layer_spacing
                if pos_in_layer == 0:
                    dx, dy = 0.0, 0.0
                else:
                    angle = 2 * math.pi * (pos_in_layer - 1) / n_ring
                    if layer % 2 == 1:
                        angle += math.pi / n_ring
                    dx = inner_r * math.cos(angle)
                    dy = inner_r * math.sin(angle)

                w_pos = torch.tensor(
                    [[bx + dx, by + dy, sz]], device=self.device
                )
                w_pos += self.scene.env_origins[eid:eid+1]
                w_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
                single_eid = env_ids[env_local_idx:env_local_idx+1]
                water_obj.write_root_pose_to_sim(
                    torch.cat([w_pos, w_rot], dim=-1), env_ids=single_eid
                )
                water_obj.write_root_velocity_to_sim(
                    zero_vel[env_local_idx:env_local_idx+1], env_ids=single_eid
                )

        # Reset counters
        self.water_in_cup_count[env_ids] = 0

    def _clamp_xy_to_table(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, TABLE_XY_XMIN), TABLE_XY_XMAX),
            min(max(y, TABLE_XY_YMIN), TABLE_XY_YMAX),
        )

    def _sample_cup_bottle_sprite_positions(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample cup, cola bottle, and Sprite bottle XY on the table.

        Cup–cola distance in [min_cup_bottle_dist, max_cup_bottle_dist].
        Sprite kept at least min_sprite_sep_dist from both.
        """
        cfg = self.cfg
        max_attempts = 120

        cup_xy = torch.zeros(n, 2, device=self.device)
        bottle_xy = torch.zeros(n, 2, device=self.device)
        sprite_xy = torch.zeros(n, 2, device=self.device)

        for i in range(n):
            ok = False
            for _ in range(max_attempts):
                cx = sample_uniform(cfg.cup_pos_x_range[0], cfg.cup_pos_x_range[1], (1, 1), self.device).item()
                cy = sample_uniform(cfg.cup_pos_y_range[0], cfg.cup_pos_y_range[1], (1, 1), self.device).item()
                cx, cy = self._clamp_xy_to_table(cx, cy)
                bx = sample_uniform(cfg.bottle_pos_x_range[0], cfg.bottle_pos_x_range[1], (1, 1), self.device).item()
                by = sample_uniform(cfg.bottle_pos_y_range[0], cfg.bottle_pos_y_range[1], (1, 1), self.device).item()
                bx, by = self._clamp_xy_to_table(bx, by)
                dist_cb = math.hypot(cx - bx, cy - by)
                if not (cfg.min_cup_bottle_dist <= dist_cb <= cfg.max_cup_bottle_dist):
                    continue
                sx = sample_uniform(
                    cfg.sprite_bottle_pos_x_range[0], cfg.sprite_bottle_pos_x_range[1], (1, 1), self.device
                ).item()
                sy = sample_uniform(
                    cfg.sprite_bottle_pos_y_range[0], cfg.sprite_bottle_pos_y_range[1], (1, 1), self.device
                ).item()
                sx, sy = self._clamp_xy_to_table(sx, sy)
                if math.hypot(sx - cx, sy - cy) < cfg.min_sprite_sep_dist:
                    continue
                if math.hypot(sx - bx, sy - by) < cfg.min_sprite_sep_dist:
                    continue
                cup_xy[i] = torch.tensor([cx, cy], device=self.device)
                bottle_xy[i] = torch.tensor([bx, by], device=self.device)
                sprite_xy[i] = torch.tensor([sx, sy], device=self.device)
                ok = True
                break
            if not ok:
                # Fallback: fixed layout
                cup_xy[i, 0], cup_xy[i, 1] = CUP_POS[0], CUP_POS[1]
                bottle_xy[i, 0], bottle_xy[i, 1] = BOTTLE_POS[0], BOTTLE_POS[1]
                sprite_xy[i, 0], sprite_xy[i, 1] = SPRITE_BOTTLE_POS[0], SPRITE_BOTTLE_POS[1]

        return cup_xy, bottle_xy, sprite_xy
