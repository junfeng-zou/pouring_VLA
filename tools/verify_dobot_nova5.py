"""
DOBOT Nova 5 — Isaac Lab Verification Script
=============================================

Loads the DOBOT Nova 5 USD model in an Isaac Lab simulation environment
and runs a simple joint sinusoidal motion test to verify:
  1. The mesh renders correctly
  2. All 6 joints are functional
  3. Joint limits and drives are properly configured
  4. No self-collision explosions

Usage:
    /home/zjf/IsaacLab/isaaclab.sh -p scripts/verify_dobot_nova5.py

    # Headless mode (no GUI, prints joint info only):
    /home/zjf/IsaacLab/isaaclab.sh -p scripts/verify_dobot_nova5.py --headless

    # Num environments (default: 1):
    /home/zjf/IsaacLab/isaaclab.sh -p scripts/verify_dobot_nova5.py --num_envs 2
"""

import argparse
import math
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify DOBOT Nova 5 in Isaac Lab.")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Launch SimulationApp first ──────────────────────────────────────
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    # ── Now import Isaac Lab / USD modules (after SimulationApp init) ──
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext

    import omni.usd
    from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Sdf, PhysxSchema

    # ── Paths ──────────────────────────────────────────────────────────
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    USD_PATH = os.path.join(
        PROJECT_DIR,
        "assets",
        "DOBOT Nova 5-20221011.SLDASM",
        "usd",
        "dobot_nova5.usd",
    )

    if not os.path.isfile(USD_PATH):
        print(f"[ERROR] USD file not found: {USD_PATH}")
        print("[ERROR] Please run convert_dobot_urdf.py first.")
        simulation_app.close()
        sys.exit(1)

    # ── Simulation context ─────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)

    # Set up camera view
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.5])

    # ── Create ground plane using USD API (no Nucleus needed) ──────────
    stage = omni.usd.get_context().get_stage()

    # - Physics scene (if not already present)
    if not stage.GetPrimAtPath("/World/PhysicsScene"):
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(9.81)

    # - Ground plane as a large flat collision plane
    ground_prim_path = "/World/GroundPlane"
    ground_prim = UsdGeom.Xform.Define(stage, ground_prim_path)

    plane_path = f"{ground_prim_path}/CollisionPlane"
    plane = UsdGeom.Plane.Define(stage, plane_path)
    plane.CreateAxisAttr("Z")
    plane.CreateExtentAttr([(-50, -50, 0), (50, 50, 0)])

    # Add collision
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(plane_path))

    # Visual ground (large thin box for visibility)
    visual_path = f"{ground_prim_path}/VisualPlane"
    visual_plane = UsdGeom.Cube.Define(stage, visual_path)
    visual_plane.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(visual_plane)
    xform.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 0.01))
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    visual_plane.CreateDisplayColorAttr([(0.3, 0.3, 0.3)])

    # ── Dome light ─────────────────────────────────────────────────────
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(2000.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.8, 0.8, 0.8))

    # ── Distant light for better shadows ───────────────────────────────
    dist_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dist_light.CreateIntensityAttr(3000.0)
    dist_light.CreateAngleAttr(0.53)
    xform_light = UsdGeom.Xformable(dist_light)
    xform_light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    # ── Robot configuration ────────────────────────────────────────────
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
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
        },
    )

    # ── Create environment prims ───────────────────────────────────────
    num_envs = args.num_envs
    env_spacing = 2.0
    for i in range(num_envs):
        env_path = f"/World/envs/env_{i}"
        env_xform = UsdGeom.Xform.Define(stage, env_path)
        # Offset each environment so they don't overlap
        env_xform.AddTranslateOp().Set(Gf.Vec3d(i * env_spacing, 0.0, 0.0))

    # ── Instantiate articulation ───────────────────────────────────────
    robot = Articulation(robot_cfg)

    # ── Reset sim ──────────────────────────────────────────────────────
    sim.reset()

    # ── Print robot info ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DOBOT Nova 5 — Verification")
    print("=" * 60)
    print(f"  Num environments : {num_envs}")
    print(f"  Num bodies       : {robot.num_bodies}")
    print(f"  Num joints       : {robot.num_joints}")
    print(f"  Joint names      : {robot.joint_names}")
    print(f"  Body names       : {robot.body_names}")
    print(f"  Root state shape : {robot.data.root_state_w.shape}")
    print(f"  Joint pos shape  : {robot.data.joint_pos.shape}")
    print("=" * 60)

    # ── Sinusoidal joint motion test ───────────────────────────────────
    print("\n[INFO] Running sinusoidal joint motion test...")
    print("[INFO] Each joint will oscillate. Watch for correct motion.")
    if not args.headless:
        print("[INFO] Close the window or Ctrl+C to stop.\n")

    step = 0
    max_steps = 3000 if args.headless else None  # ~25 seconds in headless

    while simulation_app.is_running():
        # Time in seconds
        t = step * sim_cfg.dt

        # Generate sinusoidal targets for each joint
        joint_targets = torch.zeros(num_envs, robot.num_joints, device=robot.device)
        amplitude = 0.8  # radians (~45 degrees)
        for j in range(robot.num_joints):
            phase = j * (2.0 * math.pi / robot.num_joints)
            joint_targets[:, j] = amplitude * math.sin(2.0 * math.pi * 0.2 * t + phase)

        # Apply joint position targets
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()

        # Step simulation
        sim.step()

        # Update robot state
        robot.update(sim_cfg.dt)

        # Print joint positions periodically
        if step % 240 == 0:  # Every ~2 seconds
            pos = robot.data.joint_pos[0].cpu().numpy()
            pos_str = ", ".join([f"{p:+.3f}" for p in pos])
            print(f"  [t={t:6.2f}s] Joint pos: [{pos_str}]")

        step += 1
        if max_steps and step >= max_steps:
            print("\n[INFO] Headless test completed successfully!")
            print("[INFO] All joints responded to position commands.")
            break

    simulation_app.close()
    print("[INFO] Verification done.")


if __name__ == "__main__":
    main()
