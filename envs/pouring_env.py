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
NUM_WATER_SPHERES = 50
WATER_SPHERE_RADIUS = 0.008  # 8 mm
WALL_THICKNESS = 0.004       # container wall thickness
GRIPPER_MAX_OPEN = 0.055     # max finger travel (must match URDF upper limit)

# -- Geometry ---------------------------------------------------------------
TABLE_SIZE = (1.2, 0.8, 0.75)  # length, width, height
TABLE_POS = (0.3, 0.0, TABLE_SIZE[2] / 2.0)  # robot at x=0, table extends from x=-0.3 to x=0.9

CUP_RADIUS = 0.04
CUP_HEIGHT = 0.10
CUP_POS = (0.5, 0.0, TABLE_SIZE[2] + CUP_HEIGHT / 2.0)

BOTTLE_RADIUS = 0.035
BOTTLE_HEIGHT = 0.20
BOTTLE_POS = (0.5, 0.3, TABLE_SIZE[2] + BOTTLE_HEIGHT / 2.0)  # on the table, offset from cup in Y

@configclass
class PouringEnvCfg(DirectRLEnvCfg):
    """Configuration for the DOBOT Nova 5 pouring environment."""

    # -- env --
    episode_length_s = 120.0
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

    # -- third-person camera (Orbbec Femto Bolt RGB mode) --
    # Femto Bolt RGB: 1920x1080, HFOV ≈ 80°
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.49,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # pos=(0.56, 0.01, 1.67),
            pos=(0.81, 0.15, 0.96),             # in front of workspace
            rot=(0.0000, 0.0000, 0.0000, 1.0000),   # 90° around Y → look along -X
            # rot=(0, 0, 0.34645, 0.93807),  # (w, x, y, z)
            convention="world",
        ),
        width=640,
        height=360,
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.1,
    )

    # cam_top removed for performance

    # Side-view camera: from the front, looking back at workspace
    # 90° rotation around Y: camera -Z → world -X (looking toward table)
    cam_side = CameraCfg(
        prim_path="/World/envs/env_.*/CamSide",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.56, 0.636, 1.54),
            rot=(-0.6423, -0.2958, -0.2958, 0.6423),  # (w, x, y, z)
            # rot=(0.5, 0.5, 0.5, 0.5),   # 90° around Y → look along -X
            convention="world",
        ),
        width=640,
        height=480,
        data_types=["rgb"],
        update_period=0.1,
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
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": -1.5708,   # elbow up
                "joint_4": 0.0,
                "joint_5": -1.5708,  # wrist down
                "joint_6": 0.0,
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

        # Cup position (fixed, per env)
        self.cup_pos = torch.tensor(
            [CUP_POS] * self.num_envs, device=self.device, dtype=torch.float32
        )
        # Adjust for env origins
        self.cup_pos += self.scene.env_origins

        # Water tracking
        self.num_water = NUM_WATER_SPHERES
        self.water_in_cup_count = torch.zeros(self.num_envs, device=self.device)

        # PhysX view for bottle — created lazily on first reset
        self._bottle_physx_view = None

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
            color=(0.1, 0.3, 0.8),
            position=BOTTLE_POS,
            kinematic=False,  # dynamic — can be grasped and moved
            mass=0.3,         # 300g bottle
        )
        self._bottle_prim_path = "/World/envs/env_0/Bottle"

        # Wrap the bottle as an Isaac Lab RigidObject so we can use
        # write_root_pose_to_sim() for proper physics-level reset.
        # spawn=None because the prim already exists from _create_hollow_container.
        bottle_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bottle",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=BOTTLE_POS,
            ),
        )
        self._bottle_obj = RigidObject(bottle_cfg)
        self.scene.rigid_objects["bottle"] = self._bottle_obj

        # -- Water spheres --
        # We spawn them individually with unique prim paths
        self._water_spheres: list[RigidObject] = []
        for i in range(NUM_WATER_SPHERES):
            water_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Water_{i:03d}",
                spawn=self.cfg.water_sphere.spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.65 + i * WATER_SPHERE_RADIUS * 2.5),
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

            # Set IK command (relative delta)
            self._ik_controller.set_command(cart_delta, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

            # Get Jacobian and current arm joint positions
            jacobian = self._robot.root_physx_view.get_jacobians()[
                :, self._ee_jacobi_idx, :, self._robot_entity_cfg.joint_ids
            ]
            arm_joint_pos = self._robot.data.joint_pos[:, self._robot_entity_cfg.joint_ids]

            # Compute IK: returns actual_pos + delta
            arm_joint_targets = self._ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, arm_joint_pos
            )

            # Extract pure delta and apply to PREVIOUS TARGETS (not actual pos).
            # This prevents "accepting" gravity drift on each input step.
            joint_ids = self._robot_entity_cfg.joint_ids
            ik_delta = arm_joint_targets - arm_joint_pos  # pure IK delta
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
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        # End-effector position in world frame
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_link_idx]

        # Cup position (subtract env origin to get local coords)
        cup_pos_local = torch.tensor(
            [CUP_POS] * self.num_envs, device=self.device, dtype=torch.float32
        )

        obs = torch.cat([
            joint_pos[:, :6],                   # (N, 6), arm joint positions
            joint_vel[:, :6] * 0.1,             # (N, 6), arm joint velocities, scaled
            joint_pos[:, 6:8],                  # (N, 2), gripper finger positions
            ee_pos_w - self.scene.env_origins,   # (N, 3), local EE pos
            cup_pos_local,                       # (N, 3), local cup pos
        ], dim=-1)

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        """Reward based on water spheres reaching the cup region."""
        # Count water spheres inside the cup region
        cup_center_xy = torch.tensor(
            [[CUP_POS[0], CUP_POS[1]]] * self.num_envs, device=self.device
        )
        cup_top_z = CUP_POS[2] + CUP_HEIGHT / 2.0

        water_in_cup = torch.zeros(self.num_envs, device=self.device)
        for water_obj in self._water_spheres:
            w_pos = water_obj.data.root_pos_w  # (num_envs, 3)
            w_pos_local = w_pos - self.scene.env_origins

            # Check if sphere is within cup XY radius and below cup top Z
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
        """Reset specified environments."""
        super()._reset_idx(env_ids)

        # Reset robot joints
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.05, 0.05,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # Reset bottle (dynamic rigid body) using RigidObject API
        bottle_pos = torch.tensor(
            [BOTTLE_POS], device=self.device, dtype=torch.float32
        ).repeat(len(env_ids), 1)
        bottle_pos += self.scene.env_origins[env_ids]
        bottle_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device
        ).repeat(len(env_ids), 1)
        bottle_vel = torch.zeros((len(env_ids), 6), device=self.device)
        self._bottle_obj.write_root_pose_to_sim(
            torch.cat([bottle_pos, bottle_quat], dim=-1), env_ids=env_ids
        )
        self._bottle_obj.write_root_velocity_to_sim(bottle_vel, env_ids=env_ids)

        # Reset water spheres — arrange inside the bottle
        bottle_x, bottle_y, bottle_z = BOTTLE_POS
        bottle_bottom_z = bottle_z - BOTTLE_HEIGHT / 2.0 + WALL_THICKNESS + WATER_SPHERE_RADIUS
        for i, water_obj in enumerate(self._water_spheres):
            # Spiral arrangement inside bottle
            angle = (i / NUM_WATER_SPHERES) * 2 * math.pi * 5
            r = BOTTLE_RADIUS * 0.5 * ((i % 5) / 5.0)
            dx = r * math.cos(angle)
            dy = r * math.sin(angle)
            dz = bottle_bottom_z + (i // 5) * WATER_SPHERE_RADIUS * 2.5

            w_pos = torch.tensor(
                [[bottle_x + dx, bottle_y + dy, dz]],
                device=self.device
            ).repeat(len(env_ids), 1)
            w_pos += self.scene.env_origins[env_ids]
            w_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(len(env_ids), 1)
            w_vel = torch.zeros((len(env_ids), 6), device=self.device)
            water_obj.write_root_pose_to_sim(
                torch.cat([w_pos, w_rot], dim=-1), env_ids=env_ids
            )
            water_obj.write_root_velocity_to_sim(w_vel, env_ids=env_ids)

        # Reset counters
        self.water_in_cup_count[env_ids] = 0
