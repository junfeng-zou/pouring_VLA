"""
DOBOT Nova 5 URDF to USD Converter for Isaac Sim
=================================================

This script converts the DOBOT Nova 5 URDF file to USD format using
the low-level isaacsim.asset.importer.urdf API directly (bypassing
Isaac Lab's UrdfConverter wrapper for maximum compatibility).

It configures the joint drive API with position control mode and
appropriate PD gains.

Usage (Isaac Lab / Isaac Sim environment):
    # Option 1: Using Isaac Lab's launcher
    <IsaacLab>/isaaclab.sh -p scripts/convert_dobot_urdf.py [--headless]

    # Option 2: Using Isaac Sim's Python directly
    <isaac-sim>/python.sh scripts/convert_dobot_urdf.py [--headless]
"""

import argparse
import math
import os
import sys

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# URDF input
URDF_PATH = os.path.join(
    PROJECT_DIR,
    "assets",
    "DOBOT Nova 5-20221011.SLDASM",
    "urdf",
    "dobot_nova5.urdf",
)

# USD output
USD_OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "assets",
    "DOBOT Nova 5-20221011.SLDASM",
    "usd",
)
USD_OUTPUT_PATH = os.path.join(USD_OUTPUT_DIR, "dobot_nova5.usd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DOBOT Nova 5 URDF to USD for Isaac Lab."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode (no GUI).",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=URDF_PATH,
        help=f"Path to the input URDF file. Default: {URDF_PATH}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=USD_OUTPUT_PATH,
        help=f"Path for the output USD file. Default: {USD_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        default=True,
        help="Fix the robot base (default: True).",
    )
    parser.add_argument(
        "--merge-joints",
        action="store_true",
        default=False,
        help="Merge fixed joints (default: False).",
    )
    parser.add_argument(
        "--joint-stiffness",
        type=float,
        default=400.0,
        help="Joint drive stiffness (Kp). Default: 400.0",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=40.0,
        help="Joint drive damping (Kd). Default: 40.0",
    )
    parser.add_argument(
        "--drive-type",
        type=str,
        choices=["force", "acceleration"],
        default="force",
        help='Joint drive type. Default: "force"',
    )
    parser.add_argument(
        "--joint-target-type",
        type=str,
        choices=["none", "position", "velocity"],
        default="position",
        help='Joint target type. Default: "position"',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate input ──────────────────────────────────────────────────
    if not os.path.isfile(args.urdf):
        print(f"[ERROR] URDF file not found: {args.urdf}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("DOBOT Nova 5 URDF → USD Converter")
    print("=" * 60)
    print(f"  URDF input  : {args.urdf}")
    print(f"  USD output  : {args.output}")
    print(f"  fix_base    : {args.fix_base}")
    print(f"  merge_joints: {args.merge_joints}")
    print(f"  drive_type  : {args.drive_type}")
    print(f"  target_type : {args.joint_target_type}")
    print(f"  stiffness   : {args.joint_stiffness}")
    print(f"  damping     : {args.joint_damping}")
    print(f"  headless    : {args.headless}")
    print("=" * 60)

    # ── Launch Isaac Sim App ────────────────────────────────────────────
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    # ── Enable URDF importer extension ──────────────────────────────────
    import omni.kit.app
    import omni.kit.commands

    from isaacsim.core.utils.extensions import enable_extension

    # Try to enable the extension - handle both old and new naming
    manager = omni.kit.app.get_app().get_extension_manager()
    urdf_ext_enabled = False
    for ext_name in [
        "isaacsim.asset.importer.urdf",
        "omni.importer.urdf",
    ]:
        try:
            enable_extension(ext_name)
            urdf_ext_enabled = True
            print(f"[INFO] Enabled extension: {ext_name}")
            break
        except Exception:
            continue

    if not urdf_ext_enabled:
        print("[ERROR] Could not enable URDF importer extension.")
        simulation_app.close()
        sys.exit(1)

    # ── Import URDF types ──────────────────────────────────────────────
    try:
        from isaacsim.asset.importer.urdf._urdf import (
            UrdfJointDriveType,
            UrdfJointTargetType,
        )
    except ImportError:
        from omni.importer.urdf._urdf import (
            UrdfJointDriveType,
            UrdfJointTargetType,
        )

    # ── Create import config ────────────────────────────────────────────
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

    # Basic settings
    import_config.set_distance_scale(1.0)  # meters
    import_config.set_make_default_prim(True)
    import_config.set_create_physics_scene(False)

    # Density (0 = auto-compute from mesh)
    import_config.set_density(0.0)

    # Collision settings
    import_config.set_convex_decomp(False)
    import_config.set_collision_from_visuals(False)

    # Merge fixed joints
    import_config.set_merge_fixed_joints(args.merge_joints)
    # Note: set_merge_fixed_ignore_inertia may not exist in all versions,
    # so we call it only if available
    if hasattr(import_config, "set_merge_fixed_ignore_inertia"):
        import_config.set_merge_fixed_ignore_inertia(args.merge_joints)

    # Fix base
    import_config.set_fix_base(args.fix_base)

    # Self collision
    import_config.set_self_collision(False)

    # ── Parse URDF ──────────────────────────────────────────────────────
    print("\n[INFO] Parsing URDF file...")
    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=args.urdf,
        import_config=import_config,
    )

    if not result:
        print(f"[ERROR] Failed to parse URDF file: {args.urdf}")
        simulation_app.close()
        sys.exit(1)

    print(f"[INFO] Successfully parsed URDF. Found {len(robot_model.joints)} joints.")

    # ── Configure joint drives ──────────────────────────────────────────
    drive_type_map = {
        "force": UrdfJointDriveType.JOINT_DRIVE_FORCE,
        "acceleration": UrdfJointDriveType.JOINT_DRIVE_ACCELERATION,
    }
    target_type_map = {
        "none": UrdfJointTargetType.JOINT_DRIVE_NONE,
        "position": UrdfJointTargetType.JOINT_DRIVE_POSITION,
        "velocity": UrdfJointTargetType.JOINT_DRIVE_VELOCITY,
    }

    drive_type_enum = drive_type_map[args.drive_type]
    target_type_enum = target_type_map[args.joint_target_type]

    # Convert stiffness/damping from radians to degrees (required by PhysX)
    stiffness_deg = math.pi / 180.0 * args.joint_stiffness
    damping_deg = math.pi / 180.0 * args.joint_damping

    for joint_name, joint in robot_model.joints.items():
        joint.drive.set_drive_type(drive_type_enum)
        joint.drive.set_target_type(target_type_enum)
        joint.drive.set_strength(stiffness_deg)
        joint.drive.set_damping(damping_deg)
        print(
            f"  [Joint] {joint_name}: "
            f"drive={args.drive_type}, target={args.joint_target_type}, "
            f"Kp={args.joint_stiffness}, Kd={args.joint_damping}"
        )

    # ── Import to USD ──────────────────────────────────────────────────
    print(f"\n[INFO] Converting to USD: {args.output}")
    omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_path=args.urdf,
        urdf_robot=robot_model,
        import_config=import_config,
        dest_path=args.output,
    )
    print(f"[INFO] USD file generated at: {args.output}")

    # ── If not headless, keep the window open for inspection ──────────
    if not args.headless:
        print("\n[INFO] Press Ctrl+C or close the window to exit.")
        print("[INFO] Press PLAY in the GUI to test physics simulation.")
        while simulation_app.is_running():
            simulation_app.update()

    # ── Cleanup ────────────────────────────────────────────────────────
    simulation_app.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
