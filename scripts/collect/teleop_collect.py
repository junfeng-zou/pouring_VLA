#!/usr/bin/env python3
"""
DOBOT CR5 — Gamepad Teleoperation & OpenVLA Data Collection
=============================================================

Collects demonstration episodes for fine-tuning OpenVLA-7B.
Each episode is saved as HDF5 with RGB (224×224), depth, proprioceptive
state, 7-DoF Cartesian actions, and a language task instruction.

Launch with:
    /home/zjf/IsaacLab/isaaclab.sh -p scripts/collect/teleop_collect.py

    # headless:
    /home/zjf/IsaacLab/isaaclab.sh -p scripts/collect/teleop_collect.py --headless

Then on the local PC (with gamepad):
    python tools/teleop/gamepad_sender.py --ip <3090_TAILSCALE_IP>

Gamepad mapping (Xbox / generic dual-stick) — Cartesian EE control:
    Left  stick X           → EE Y  (left/right)
    Left  stick Y           → EE X  (forward/backward)
    Right stick X           → EE yaw rotation
    Right stick Y           → EE Z  (up/down)
    LT            (axis 4)  → pitch–
    RT            (axis 5)  → pitch+
    LB  (btn 4)             → roll–
    RB  (btn 5)             → roll+
    A   (btn 0)             → toggle recording
    B   (btn 1)             → discard current episode
    X   (btn 2)             → toggle gripper open/close
    Y   (btn 3)             → reset environment
    Start (btn 7)           → quit
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

# -- Make project root importable --
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from isaacsim import SimulationApp


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gamepad teleoperation for VLA data collection")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--port", type=int, default=9876, help="UDP port for gamepad receiver")
    p.add_argument("--hz", type=int, default=10, help="Control / recording frequency (Hz)")
    p.add_argument("--deadzone", type=float, default=0.15, help="Joystick deadzone")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Dataset save directory (default: data/vla_dataset/)")
    p.add_argument("--image_size", type=int, default=224, help="Saved image size (square)")
    p.add_argument("--task", type=str, default="pour cola from bottle into cup",
                   help="Language task instruction for OpenVLA fine-tuning")
    p.add_argument("--min_steps", type=int, default=10,
                   help="Minimum steps for a valid episode (shorter ones are discarded)")
    p.add_argument("--debug", action="store_true", default=False, help="Enable debug prints")
    p.add_argument("--show_joint_axes", action="store_true", default=False,
                   help="Draw robot joint axes in viewport (debug visualization)")
    p.add_argument("--show_ee_frame", action="store_true", default=False,
                   help="Draw EE link_6 RGB=XYZ frame arrows (same as PouringEnv debug)")
    return p.parse_args()


# ── Gamepad → Action mapping ────────────────────────────────────────────────

class GamepadMapper:
    """Convert raw gamepad data to a 7-DoF Cartesian EE action + button events.

    Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    All outputs are in [-1, 1] (env applies the actual scaling).
    """

    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.gripper_open = True  # start open

        # Button edge detection (previous state)
        self._prev_buttons: list[int] = []

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _button_pressed(self, buttons: list[int], idx: int) -> bool:
        """True on the rising edge (was 0, now 1)."""
        if idx >= len(buttons):
            return False
        curr = buttons[idx]
        prev = self._prev_buttons[idx] if idx < len(self._prev_buttons) else 0
        return curr == 1 and prev == 0

    def map(self, data: dict | None) -> tuple[np.ndarray, dict]:
        """
        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    All in [-1, 1]. gripper: +1=open, -1=closed
            events: dict of boolean flags for button presses
        """
        action = np.zeros(7, dtype=np.float32)
        events = {
            "toggle_record": False,
            "discard": False,
            "toggle_gripper": False,
            "toggle_pause_record": False,
            "reset_env": False,
            "quit": False,
        }

        if data is None:
            self._prev_buttons = []
            action[6] = 1.0 if self.gripper_open else -1.0
            return action, events

        axes = data.get("axes", [])
        buttons = data.get("buttons", [])

        def axis(idx: int) -> float:
            return self._apply_deadzone(axes[idx]) if idx < len(axes) else 0.0

        # ---- Cartesian EE control ----
        # Left stick → XY translation (robot base frame)
        #   stick X → EE Y (left/right)
        #   stick Y → EE X (forward/backward, inverted)
        action[0] = -axis(1)    # dx: push stick forward → EE moves +X
        action[1] = axis(0)     # dy: push stick right  → EE moves +Y

        # Right stick → Z height + yaw rotation
        #   stick Y → EE Z (up/down, inverted)
        #   stick X → EE yaw (rotate around Z axis)
        action[2] = -axis(3)    # dz:   push stick up    → EE moves +Z
        action[5] = axis(2)     # dyaw: push stick right → yaw CW

        # Triggers → pitch rotation
        #   LT/RT rest at -1.0, pressed → +1.0
        if len(axes) > 5:
            lt = (axes[4] + 1.0) / 2.0  # 0..1
            rt = (axes[5] + 1.0) / 2.0  # 0..1
            action[4] = rt - lt          # dpitch

        # Shoulder buttons → roll rotation
        lb_held = buttons[4] if len(buttons) > 4 else 0
        rb_held = buttons[5] if len(buttons) > 5 else 0
        action[3] = float(rb_held - lb_held)  # droll

        # Reverse teleop motion mapping direction (XYZ + RPY).
        # Gripper command (index 6) is intentionally unchanged.
        action[:6] *= -1.0

        # ---- Gripper toggle (X button) ----
        if self._button_pressed(buttons, 2):
            self.gripper_open = not self.gripper_open
            events["toggle_gripper"] = True

        action[6] = 1.0 if self.gripper_open else -1.0

        # ---- Button events (edge-triggered) ----
        events["toggle_record"] = self._button_pressed(buttons, 0)  # A
        events["discard"] = self._button_pressed(buttons, 1)        # B
        events["reset_env"] = self._button_pressed(buttons, 3)      # Y
        events["toggle_pause_record"] = self._button_pressed(buttons, 6)  # Back
        events["quit"] = self._button_pressed(buttons, 7)           # Start

        # Save for next edge detection
        self._prev_buttons = list(buttons)

        return action, events


# ── Direction compass helpers ────────────────────────────────────────────────

def _quat_wxyz_to_rotmat(q):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

# Isaac Lab world→OpenGL conversion: Euler(π/2, −π/2, 0, "XYZ")
_WORLD_TO_GL = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)

def _compute_compass(cam_rot_wxyz):
    """Compute screen-space arrow directions for world +X/+Y/+Z axes.

    Uses the camera's CameraCfg convention="world" quaternion.
    Returns: list of (label, screen_dir_xy, bgr_color).
    """
    R_world = _quat_wxyz_to_rotmat(cam_rot_wxyz)
    R_gl = R_world @ _WORLD_TO_GL
    R_w2c = R_gl.T

    result = []
    for label, world_dir, color in [
        ("L\u2191",  [1, 0, 0], (60, 60, 255)),   # left stick up  → +X  (red)
        ("L\u2192",  [0, 1, 0], (60, 220, 60)),    # left stick right → +Y (green)
    ]:
        d = R_w2c @ np.array(world_dir, dtype=float)
        sd = np.array([d[0], -d[1]])  # image coords: x-right, y-down
        n = np.linalg.norm(sd)
        if n > 1e-6:
            sd /= n
        result.append((label, sd, color))
    return result


def _draw_compass(img, compass, cx, cy, radius=30):
    """Draw a direction compass overlay on an image."""
    import cv2 as _cv2
    _cv2.circle(img, (cx, cy), 3, (180, 180, 180), -1)
    for label, sd, color in compass:
        ex = int(cx + sd[0] * radius)
        ey = int(cy + sd[1] * radius)
        _cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.3)
        tx = int(cx + sd[0] * (radius + 14))
        ty = int(cy + sd[1] * (radius + 14))
        _cv2.putText(img, label, (tx - 8, ty + 5),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, _cv2.LINE_AA)


def _axis_token_to_vec(axis_token: str) -> np.ndarray:
    token = str(axis_token).upper()
    if token == "X":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if token == "Y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _get_joint_local_axes(robot_prim_path: str, joint_names: list[str]) -> dict[str, np.ndarray]:
    """Try reading revolute axis from USD joints; fallback to local +Z."""
    axis_map: dict[str, np.ndarray] = {}
    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        for name in joint_names:
            axis_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            prim = stage.GetPrimAtPath(f"{robot_prim_path}/{name}")
            if prim and prim.IsValid():
                rev_joint = UsdPhysics.RevoluteJoint(prim)
                if rev_joint and rev_joint.GetPrim().IsValid():
                    axis_attr = rev_joint.GetAxisAttr().Get()
                    axis_vec = _axis_token_to_vec(axis_attr if axis_attr is not None else "Z")
            axis_map[name] = axis_vec
    except Exception:
        for name in joint_names:
            axis_map[name] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return axis_map


def _draw_joint_axes(draw_iface, env, joint_names: list[str], joint_body_idx: list[int], joint_local_axes: list[np.ndarray]):
    """Draw per-joint rotation axes as yellow lines in world frame."""
    if draw_iface is None:
        return
    body_pos_w = env._robot.data.body_pos_w[0].cpu().numpy()
    body_quat_w = env._robot.data.body_quat_w[0].cpu().numpy()

    starts = []
    ends = []
    colors = []
    sizes = []
    axis_len = 0.16
    for i, joint_name in enumerate(joint_names):
        body_idx = joint_body_idx[i]
        if body_idx < 0:
            continue
        origin = body_pos_w[body_idx]
        R = _quat_wxyz_to_rotmat(body_quat_w[body_idx])
        axis_dir = R @ joint_local_axes[i]
        n = np.linalg.norm(axis_dir)
        if n < 1e-8:
            continue
        axis_dir = axis_dir / n

        starts.append((float(origin[0]), float(origin[1]), float(origin[2])))
        end = origin + axis_len * axis_dir
        ends.append((float(end[0]), float(end[1]), float(end[2])))
        colors.append((1.0, 1.0, 0.0, 1.0))  # yellow: rotation axis
        sizes.append(3.0)

    # Draw all joints in one batch for efficiency.
    if starts:
        draw_iface.draw_lines(starts, ends, colors, sizes)


def _create_joint_axis_curves(parent_prim_path: str, joint_names: list[str]):
    """Create persistent USD arrow curves for joint-axis visualization."""
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, parent_prim_path)
    curves = {}

    def _make_curve(path: str):
        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)])
        curve.CreateWidthsAttr([0.006])
        curve.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 0.0)])  # yellow
        return curve

    for name in joint_names:
        curves[name] = {
            "shaft": _make_curve(f"{parent_prim_path}/{name}_axis_shaft"),
            "head_l": _make_curve(f"{parent_prim_path}/{name}_axis_head_l"),
            "head_r": _make_curve(f"{parent_prim_path}/{name}_axis_head_r"),
        }
    return curves


def _update_joint_axis_curves(curves, env, joint_names: list[str], joint_body_idx: list[int], joint_local_axes: list[np.ndarray], axis_len: float = 0.16):
    """Update USD arrow curves to match current joint axis directions in world frame."""
    from pxr import Gf

    body_pos_w = env._robot.data.body_pos_w[0].cpu().numpy()
    body_quat_w = env._robot.data.body_quat_w[0].cpu().numpy()
    root_quat_w = env._robot.data.root_quat_w[0].cpu().numpy()
    R_root_to_w = _quat_wxyz_to_rotmat(root_quat_w)
    axis_dirs_w = [None] * len(joint_names)

    # Preferred: use Jacobian angular part for true +qdot rotation direction.
    # This gives signed axis directions consistent with robot kinematics.
    try:
        jacobian = env._robot.root_physx_view.get_jacobians()[
            0, env._ee_jacobi_idx, :, env._robot_entity_cfg.joint_ids
        ].cpu().numpy()  # (6, num_arm_joints)
        for i in range(min(len(joint_names), jacobian.shape[1])):
            axis_root = jacobian[3:6, i]
            axis_dirs_w[i] = R_root_to_w @ axis_root
    except Exception:
        pass

    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    side_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for i, joint_name in enumerate(joint_names):
        body_idx = joint_body_idx[i]
        if body_idx < 0 or joint_name not in curves:
            continue
        origin = body_pos_w[body_idx]
        axis_dir = axis_dirs_w[i]
        if axis_dir is None:
            # Fallback: rotate local axis by child link orientation.
            R = _quat_wxyz_to_rotmat(body_quat_w[body_idx])
            axis_dir = R @ joint_local_axes[i]
        n = np.linalg.norm(axis_dir)
        if n < 1e-8:
            continue
        axis_dir = axis_dir / n
        end = origin + axis_len * axis_dir

        # Build an arrow head in a stable plane orthogonal to axis_dir.
        tangent = np.cross(axis_dir, up_hint)
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.cross(axis_dir, side_hint)
        tangent = tangent / max(np.linalg.norm(tangent), 1e-8)
        head_len = axis_len * 0.25
        head_w = axis_len * 0.12
        left = end - head_len * axis_dir + head_w * tangent
        right = end - head_len * axis_dir - head_w * tangent

        curves[joint_name]["shaft"].GetPointsAttr().Set([
            Gf.Vec3f(float(origin[0]), float(origin[1]), float(origin[2])),
            Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
        ])
        curves[joint_name]["head_l"].GetPointsAttr().Set([
            Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
            Gf.Vec3f(float(left[0]), float(left[1]), float(left[2])),
        ])
        curves[joint_name]["head_r"].GetPointsAttr().Set([
            Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
            Gf.Vec3f(float(right[0]), float(right[1]), float(right[2])),
        ])


def _capture_pause_snapshot(env, mapper, t: float, step: int, steps_since_reset: int) -> dict:
    """Capture env + controller states for pause/resume during recording."""
    snap = {
        "t": float(t),
        "step": int(step),
        "steps_since_reset": int(steps_since_reset),
        "gripper_open": bool(mapper.gripper_open),
        "robot_joint_pos": env._robot.data.joint_pos.clone(),
        "robot_joint_vel": env._robot.data.joint_vel.clone(),
        "robot_dof_targets": env.robot_dof_targets.clone(),
        "cup_root_state_w": env._cup_obj.data.root_state_w.clone(),
        "bottle_root_state_w": env._bottle_obj.data.root_state_w.clone(),
        "sprite_bottle_root_state_w": env._sprite_bottle_obj.data.root_state_w.clone(),
        "water_root_state_w": [w.data.root_state_w.clone() for w in env._water_spheres],
    }
    return snap


def _restore_pause_snapshot(env, mapper, snap: dict):
    """Restore env + controller states captured by _capture_pause_snapshot."""
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    mapper.gripper_open = bool(snap["gripper_open"])

    robot_joint_pos = snap["robot_joint_pos"]
    robot_joint_vel = snap["robot_joint_vel"]
    env._robot.set_joint_position_target(snap["robot_dof_targets"])
    env._robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel)
    env.robot_dof_targets.copy_(snap["robot_dof_targets"])

    cup_state = snap["cup_root_state_w"]
    env._cup_obj.write_root_pose_to_sim(cup_state[:, :7], env_ids=env_ids)
    env._cup_obj.write_root_velocity_to_sim(cup_state[:, 7:], env_ids=env_ids)

    bottle_state = snap["bottle_root_state_w"]
    env._bottle_obj.write_root_pose_to_sim(bottle_state[:, :7], env_ids=env_ids)
    env._bottle_obj.write_root_velocity_to_sim(bottle_state[:, 7:], env_ids=env_ids)

    sprite_state = snap["sprite_bottle_root_state_w"]
    env._sprite_bottle_obj.write_root_pose_to_sim(sprite_state[:, :7], env_ids=env_ids)
    env._sprite_bottle_obj.write_root_velocity_to_sim(sprite_state[:, 7:], env_ids=env_ids)

    for i, water_obj in enumerate(env._water_spheres):
        w_state = snap["water_root_state_w"][i]
        water_obj.write_root_pose_to_sim(w_state[:, :7], env_ids=env_ids)
        water_obj.write_root_velocity_to_sim(w_state[:, 7:], env_ids=env_ids)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve save directory
    save_dir = args.save_dir or os.path.join(_REPO_ROOT, "data", "vla_dataset")

    # ── Launch SimulationApp ──
    sim_app = SimulationApp({
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "enable_cameras": True,
    })

    import carb
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

    # ── Isaac Lab imports (after SimulationApp) ──
    from envs.pouring_env import PouringEnv, PouringEnvCfg
    from tools.teleop.gamepad_receiver import GamepadReceiver
    from tools.vla.vla_data_writer import EpisodeRecorder, next_episode_path

    # ── Environment ──
    cfg = PouringEnvCfg()
    cfg.scene.num_envs = 1
    if args.show_ee_frame:
        cfg.debug_visualize_ee_frame = True
    env = PouringEnv(cfg)

    # ── Optional joint-axis debug draw ──
    joint_axis_curves = None
    joint_axis_names: list[str] = []
    joint_axis_body_idx: list[int] = []
    joint_axis_local: list[np.ndarray] = []
    if args.show_joint_axes:
        try:
            joint_axis_names = [f"joint_{i}" for i in range(1, 7)]
            joint_axis_map = _get_joint_local_axes("/World/envs/env_0/Robot", joint_axis_names)
            for jn in joint_axis_names:
                link_name = f"link_{jn.split('_')[-1]}"
                body_ids = env._robot.find_bodies(link_name)[0]
                joint_axis_body_idx.append(int(body_ids[0]) if len(body_ids) > 0 else -1)
                joint_axis_local.append(joint_axis_map[jn])
            joint_axis_curves = _create_joint_axis_curves("/World/DebugJointAxes", joint_axis_names)
            print("[INFO] Joint-axis visualization enabled (USD yellow lines).")
        except Exception as exc:
            joint_axis_curves = None
            print(f"[WARN] Failed to enable joint-axis visualization: {exc}")

    # ── Gamepad receiver ──
    receiver = GamepadReceiver(port=args.port)
    receiver.start()

    # ── Mapper & recorder ──
    mapper = GamepadMapper(deadzone=args.deadzone)
    recorder = EpisodeRecorder(image_size=args.image_size, task_description=args.task)

    # Language prompt pool for OpenVLA diversity.
    # Each recording randomly picks one; all are semantically equivalent.
    TASK_PROMPTS = [
        args.task,  # user-specified default
        "pour cola from bottle into cup",
        "pour the cola into the cup",
        "pick up the bottle and pour cola into the cup",
        "fill the cup with cola from the bottle",
        "grasp the bottle and pour its cola into the cup",
        "transfer cola from the bottle to the cup",
        "pour cola from the bottle into the target cup",
        "grab the bottle, move it over the cup, and pour cola",
        "i want to drink cola",
        "i feel like drinking cola",
        "i would like a cup of cola",
        "prepare a cup of cola for drinking",
        "pour me some cola into the cup",
        "i want some cola in the cup",
        "serve cola by pouring it into the cup",
        "let me drink cola by pouring it into a cup",
        "pour coke from the bottle into the cup",
        "pour the coke into the cup",
        "pick up the coke bottle and pour into the cup",
        "fill this cup with coke",
        "transfer coke into the cup",
        "move the bottle above the cup and pour coke",
        "tilt the bottle to pour cola into the cup",
        "pour a drink of cola into the cup",
        "get me a cup of cola",
        "i would like to drink a cup of cola",
        "i want a cola drink in the cup",
        "please pour cola into the cup for me",
        "serve me cola by pouring it into a cup",
        "prepare my cola by pouring it into the cup",
        "i'm thirsty for cola",
        "i want to have some coke",
        "pour some coke for me",
        "let's pour cola into the cup",
        "complete the task by pouring cola into the cup",
        "carefully pour cola from bottle to cup",
    ]
    # Deduplicate while preserving order
    seen = set()
    TASK_PROMPTS = [p for p in TASK_PROMPTS if not (p in seen or seen.add(p))]

    # ── State ──
    recording = False
    pause_recording = False
    pause_snapshot = None
    episode_count = 0
    sim_dt = cfg.sim.dt * cfg.decimation
    control_interval = max(1, int(1.0 / (args.hz * sim_dt)))  # sim steps per control step

    WARMUP_STEPS = 5  # skip first N steps after reset for camera warm-up

    # Observation vector layout: jpos(6) + jvel(6) + grip(2) + ee_pos(3) + cup_pos(3) = 20
    OBS_EE_POS_SLICE = slice(14, 17)

    print("\n" + "=" * 64)
    print("DOBOT CR5 — OpenVLA Data Collection (Gamepad Teleoperation)")
    print("=" * 64)
    print(f"  UDP port         : {args.port}")
    print(f"  Control freq     : {args.hz} Hz  (record every {control_interval} sim steps)")
    print(f"  Deadzone         : {args.deadzone}")
    print(f"  Image size       : {args.image_size}×{args.image_size}")
    print(f"  Save directory   : {save_dir}")
    print(f"  Task instruction : {args.task} (+{len(TASK_PROMPTS)-1} variants)")
    print(f"  Min episode steps: {args.min_steps}")
    print("=" * 64)
    print("  A = toggle recording | B = discard | X = gripper | Back = pause/resume rec")
    print("  Y = reset | Start = quit")
    print("  Waiting for gamepad data...\n")

    # ── OpenCV visualization ──
    import cv2
    import numpy as np

    TELEOP_WIN = "Teleop Cameras"
    cv2.namedWindow(TELEOP_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TELEOP_WIN, 1280, 720)

    # ── Precompute direction compasses for each camera ──
    main_compass = _compute_compass(cfg.camera.offset.rot)
    side_compass = _compute_compass(cfg.cam_side.offset.rot)

    # ── Reset ──
    obs, info = env.reset()
    step = 0
    t = 0.0
    steps_since_reset = 0

    try:
        while True:
            loop_t0 = time.time()

            # ---------- Read gamepad ----------
            gp_data = receiver.get_latest()

            # ===== DEBUG POINT 1: Raw gamepad data =====
            if args.debug and step % 30 == 0:
                if gp_data is None:
                    print("[DEBUG-1] gp_data = None (no UDP packets received yet)")
                else:
                    axes = gp_data.get('axes', [])
                    buttons = gp_data.get('buttons', [])
                    seq = gp_data.get('seq', '?')
                    print(f"[DEBUG-1] seq={seq}  axes({len(axes)})={[round(a,3) for a in axes]}  buttons({len(buttons)})={buttons}")

            action_np, events = mapper.map(gp_data)

            # ===== DEBUG POINT 2: Mapped action =====
            if args.debug and step % 30 == 0:
                nonzero = any(abs(a) > 1e-6 for a in action_np[:6])
                print(f"[DEBUG-2] action=[{', '.join(f'{a:+.5f}' for a in action_np)}]  nonzero_arm={nonzero}  gripper={'OPEN' if action_np[6]>0 else 'CLOSED'}")

            # ---------- Handle events ----------
            if events["quit"]:
                print("\n[INFO] Quit requested via gamepad.")
                break

            if events["reset_env"]:
                if recording:
                    print("[WARN] Recording stopped & discarded due to reset.")
                    recorder.reset()
                    recording = False
                pause_recording = False
                pause_snapshot = None
                obs, info = env.reset()
                mapper.gripper_open = True
                step = 0
                t = 0.0
                steps_since_reset = 0
                print("[INFO] Environment reset.")
                continue

            if events["toggle_record"]:
                if not recording:
                    recording = True
                    pause_recording = False
                    pause_snapshot = None
                    recorder.reset()
                    chosen_task = random.choice(TASK_PROMPTS)
                    recorder.task_description = chosen_task
                    print(f"[●REC] Recording started — episode {episode_count}")
                    print(f"       Task: \"{chosen_task}\"")
                else:
                    if recorder.num_steps >= args.min_steps:
                        ep_path = next_episode_path(save_dir)
                        recorder.save(ep_path, dt=1.0 / args.hz)
                        print(f"[✓SAVE] Episode saved: {ep_path}  ({recorder.num_steps} steps)")
                        episode_count += 1
                    else:
                        print(f"[✗SKIP] Episode too short ({recorder.num_steps} < {args.min_steps}), discarded")
                    recording = False
                    pause_recording = False
                    pause_snapshot = None
                    recorder.reset()

            if events["discard"]:
                if recording:
                    print(f"[✗DISCARD] Episode discarded ({recorder.num_steps} steps)")
                    recorder.reset()
                    recording = False
                    pause_recording = False
                    pause_snapshot = None

            if events["toggle_pause_record"]:
                if not recording:
                    print("[WARN] Pause ignored: recording is not active.")
                elif not pause_recording:
                    pause_snapshot = _capture_pause_snapshot(env, mapper, t, step, steps_since_reset)
                    pause_recording = True
                    print("[⏸PAUSE] Recording paused. You may do informal operations now.")
                else:
                    _restore_pause_snapshot(env, mapper, pause_snapshot)
                    t = pause_snapshot["t"]
                    step = pause_snapshot["step"]
                    steps_since_reset = pause_snapshot["steps_since_reset"]
                    obs = env._get_observations()
                    pause_recording = False
                    pause_snapshot = None
                    print("[▶RESUME] Snapshot restored. Continue formal recording.")

            if events["toggle_gripper"]:
                state = "OPEN" if mapper.gripper_open else "CLOSED"
                print(f"[GRIPPER] {state}")

            # ---------- Step environment ----------
            action_tensor = torch.tensor(action_np, device=env.device, dtype=torch.float32).unsqueeze(0)

            # ===== DEBUG POINT 3: Joint targets before & after step =====
            if args.debug and step % 30 == 0:
                jpos_before = env._robot.data.joint_pos[0].cpu().numpy()
                jtarget_before = env.robot_dof_targets[0].cpu().numpy()
                print(f"[DEBUG-3] PRE-STEP  joint_pos =[{', '.join(f'{j:+.4f}' for j in jpos_before[:6])}]")
                print(f"[DEBUG-3] PRE-STEP  dof_target=[{', '.join(f'{j:+.4f}' for j in jtarget_before[:6])}]")

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            t += sim_dt
            steps_since_reset += 1

            if joint_axis_curves is not None:
                _update_joint_axis_curves(
                    joint_axis_curves,
                    env,
                    joint_axis_names,
                    joint_axis_body_idx,
                    joint_axis_local,
                )

            # ===== DEBUG POINT 3b: Joint targets after step =====
            if args.debug and step % 30 == 0:
                jtarget_after = env.robot_dof_targets[0].cpu().numpy()
                jpos_after = env._robot.data.joint_pos[0].cpu().numpy()
                print(f"[DEBUG-3] POST-STEP dof_target=[{', '.join(f'{j:+.4f}' for j in jtarget_after[:6])}]")
                print(f"[DEBUG-3] POST-STEP joint_pos =[{', '.join(f'{j:+.4f}' for j in jpos_after[:6])}]")
                print(f"[DEBUG-3] target_delta        =[{', '.join(f'{(jtarget_after[i]-jtarget_before[i]):+.5f}' for i in range(6))}]")
                print()

            # ---------- Record data (at control frequency, after camera warm-up) ----------
            if recording and (not pause_recording) and step % control_interval == 0 and steps_since_reset > WARMUP_STEPS:
                cam_data = env._camera.data
                if "rgb" in cam_data.output and "distance_to_image_plane" in cam_data.output:
                    rgb = cam_data.output["rgb"][0].cpu().numpy()
                    if rgb.shape[-1] == 4:
                        rgb = rgb[:, :, :3]
                    depth = cam_data.output["distance_to_image_plane"][0].cpu().numpy()
                    depth = depth.squeeze(-1)

                    # Side camera (optional, for multi-view VLA)
                    rgb_side = None
                    side_data = env._cam_side.data
                    if "rgb" in side_data.output and side_data.output["rgb"].numel() > 0:
                        rgb_s = side_data.output["rgb"][0].cpu().numpy()
                        if rgb_s.shape[-1] == 4:
                            rgb_s = rgb_s[:, :, :3]
                        rgb_side = rgb_s

                    state_vec = obs["policy"][0].cpu().numpy()
                    # Physical (non-normalized) action deltas used by IK:
                    # position in meters, rotation in radians.
                    action_raw = action_np.copy()
                    action_raw[:3] *= cfg.pos_action_scale
                    action_raw[3:6] *= cfg.rot_action_scale
                    recorder.add_step(
                        rgb=rgb,
                        depth=depth,
                        state=state_vec,
                        action=action_np,
                        action_raw=action_raw,
                        timestamp=t,
                        rgb_side=rgb_side,
                    )

            # ---------- Visualization ----------
            if step % 6 == 0:
                panel_h = 540  # fixed height for all panels (higher display resolution)
                panels = []

                # -- Main camera (data collection camera) --
                cam_data = env._camera.data
                if "rgb" in cam_data.output and cam_data.output["rgb"].numel() > 0:
                    rgb_vis = cam_data.output["rgb"][0].cpu().numpy()
                    if rgb_vis.shape[-1] == 4:
                        rgb_vis = rgb_vis[:, :, :3]
                    main_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
                    h, w = main_bgr.shape[:2]
                    panel_w = int(w * panel_h / h)
                    main_bgr = cv2.resize(main_bgr, (panel_w, panel_h))
                else:
                    main_bgr = np.zeros((panel_h, int(panel_h * 16 / 9), 3), dtype=np.uint8)
                    cv2.putText(main_bgr, "Main: No Data", (10, panel_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                panels.append(main_bgr)

                # -- Auxiliary camera (side view) --
                for cam_obj, label, compass in [
                    (env._cam_side, "Side View", side_compass),
                ]:
                    cam_d = cam_obj.data
                    if "rgb" in cam_d.output and cam_d.output["rgb"].numel() > 0:
                        aux_rgb = cam_d.output["rgb"][0].cpu().numpy()
                        if aux_rgb.shape[-1] == 4:
                            aux_rgb = aux_rgb[:, :, :3]
                        aux_bgr = cv2.cvtColor(aux_rgb, cv2.COLOR_RGB2BGR)
                        h, w = aux_bgr.shape[:2]
                        panel_w = int(w * panel_h / h)
                        aux_bgr = cv2.resize(aux_bgr, (panel_w, panel_h))
                    else:
                        aux_bgr = np.zeros((panel_h, panel_h, 3), dtype=np.uint8)
                    cv2.putText(aux_bgr, label, (5, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    panels.append(aux_bgr)

                # -- Stitch side-by-side --
                display = np.hstack(panels)

                # -- HUD overlay --
                rec_text = f"[REC {recorder.num_steps}]" if recording else "[IDLE]"
                if recording and pause_recording:
                    rec_text = f"[PAUSED {recorder.num_steps}]"
                gp_text = "GP:OK" if receiver.is_connected else "GP:--"
                ee_pos = obs["policy"][0, OBS_EE_POS_SLICE].cpu().numpy()
                grip_text = "OPEN" if mapper.gripper_open else "CLOSED"
                hud = f"{rec_text} Ep:{episode_count} {gp_text}  EE:[{ee_pos[0]:.2f},{ee_pos[1]:.2f},{ee_pos[2]:.2f}]  Grip:{grip_text}"
                cv2.putText(display, hud, (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if recording else (200, 200, 200), 1, cv2.LINE_AA)
                if recording and pause_recording:
                    cv2.putText(
                        display,
                        "PAUSED  (Back to Resume)",
                        (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(TELEOP_WIN, display)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break

            step += 1

            # ---------- Real-time sync ----------
            loop_elapsed = time.time() - loop_t0
            sleep_needed = sim_dt - loop_elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

            # Handle resets from env
            if terminated.any() or truncated.any():
                if recording and (not pause_recording):
                    if recorder.num_steps >= args.min_steps:
                        ep_path = next_episode_path(save_dir)
                        recorder.save(ep_path, dt=1.0 / args.hz)
                        print(f"[✓SAVE] Auto-saved on episode end: {ep_path}  ({recorder.num_steps} steps)")
                        episode_count += 1
                    else:
                        print(f"[✗SKIP] Auto-episode too short ({recorder.num_steps} < {args.min_steps}), discarded")
                    recording = False
                    pause_recording = False
                    pause_snapshot = None
                    recorder.reset()
                obs, info = env.reset()
                step = 0
                t = 0.0
                steps_since_reset = 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        if recording and (not pause_recording) and recorder.num_steps >= args.min_steps:
            ep_path = next_episode_path(save_dir)
            recorder.save(ep_path, dt=1.0 / args.hz)
            print(f"[✓SAVE] Saved on interrupt: {ep_path}  ({recorder.num_steps} steps)")
        elif recording:
            print(f"[✗SKIP] Episode too short on interrupt ({recorder.num_steps} < {args.min_steps}), discarded")

    # ── Cleanup ──
    if joint_axis_curves is not None:
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            stage.RemovePrim("/World/DebugJointAxes")
        except Exception:
            pass
    cv2.destroyAllWindows()
    receiver.stop()
    env.close()
    sim_app.close()
    print(f"\n[INFO] Done. Total episodes saved: {episode_count}")


if __name__ == "__main__":
    main()
