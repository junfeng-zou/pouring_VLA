#!/usr/bin/env python3
"""Camera quaternion converter for Isaac Lab CameraCfg.

Isaac Lab UI displays the camera quaternion in the OpenGL/USD convention.
When using CameraCfg with convention="world", Isaac Lab internally converts:

    world → opengl: R_opengl = R_world @ Euler(π/2, -π/2, 0, "XYZ")
    opengl → world: R_world  = R_opengl @ Euler(π/2, -π/2, 0, "XYZ")^T

Conventions:
    opengl: forward=-Z, up=+Y  (what the UI shows)
    world:  forward=+X, up=+Z  (what CameraCfg convention="world" expects)

Usage:
    # Convert UI quaternion (w,x,y,z) to CameraCfg convention="world"
    python tools/cam_quat_helper.py --from-ui  -0.5 -0.5 -0.5 0.5

    # Convert CameraCfg world quaternion to what UI should show (verify)
    python tools/cam_quat_helper.py --to-ui  1.0 0.0 0.0 0.0
"""

import argparse
import math
import torch


def _euler_to_matrix(angles, convention="XYZ"):
    """Convert Euler angles to rotation matrix."""
    cos = [math.cos(a) for a in angles]
    sin = [math.sin(a) for a in angles]

    Rx = torch.tensor([
        [1, 0, 0],
        [0, cos[0], -sin[0]],
        [0, sin[0], cos[0]],
    ], dtype=torch.float64)

    Ry = torch.tensor([
        [cos[1], 0, sin[1]],
        [0, 1, 0],
        [-sin[1], 0, cos[1]],
    ], dtype=torch.float64)

    Rz = torch.tensor([
        [cos[2], -sin[2], 0],
        [sin[2], cos[2], 0],
        [0, 0, 1],
    ], dtype=torch.float64)

    return Rx @ Ry @ Rz


def _quat_to_matrix(q_wxyz):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=torch.float64)


def _matrix_to_quat(R):
    """3x3 rotation matrix to quaternion (w,x,y,z)."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R.numpy())
    q_xyzw = r.as_quat()
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])


# The fixed rotation used by Isaac Lab: Euler(π/2, -π/2, 0, "XYZ")
_CONV_MAT = _euler_to_matrix([math.pi / 2, -math.pi / 2, 0])


def opengl_to_world(q_opengl):
    """Convert OpenGL/UI quaternion → CameraCfg convention='world' quaternion.

    R_world = R_opengl @ conv_mat^T
    """
    R_gl = _quat_to_matrix(q_opengl)
    R_world = R_gl @ _CONV_MAT.T
    return _matrix_to_quat(R_world)


def world_to_opengl(q_world):
    """Convert CameraCfg convention='world' quaternion → OpenGL/UI quaternion.

    R_opengl = R_world @ conv_mat
    """
    R_w = _quat_to_matrix(q_world)
    R_gl = R_w @ _CONV_MAT
    return _matrix_to_quat(R_gl)


def main():
    parser = argparse.ArgumentParser(
        description="Convert camera quaternions between Isaac Lab UI (opengl) and CameraCfg (world) conventions"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-ui", nargs=4, type=float, metavar=("W", "X", "Y", "Z"),
                       help="Convert UI quaternion (opengl) → CameraCfg (world)")
    group.add_argument("--to-ui", nargs=4, type=float, metavar=("W", "X", "Y", "Z"),
                       help="Convert CameraCfg (world) → UI quaternion (opengl)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    if args.from_ui:
        q_in = tuple(args.from_ui)
        q_out = opengl_to_world(q_in)
        print(f"  Input (UI/opengl):       rot=({q_in[0]:.4f}, {q_in[1]:.4f}, {q_in[2]:.4f}, {q_in[3]:.4f})")
        print(f"  Output (CameraCfg/world): rot=({q_out[0]:.4f}, {q_out[1]:.4f}, {q_out[2]:.4f}, {q_out[3]:.4f})")
        print(f"\n  Copy-paste:")
        print(f"    rot=({q_out[0]:.4f}, {q_out[1]:.4f}, {q_out[2]:.4f}, {q_out[3]:.4f}),  # convention=\"world\"")
    else:
        q_in = tuple(args.to_ui)
        q_out = world_to_opengl(q_in)
        print(f"  Input (CameraCfg/world): rot=({q_in[0]:.4f}, {q_in[1]:.4f}, {q_in[2]:.4f}, {q_in[3]:.4f})")
        print(f"  Output (UI/opengl):      rot=({q_out[0]:.4f}, {q_out[1]:.4f}, {q_out[2]:.4f}, {q_out[3]:.4f})")
        print(f"\n  UI should show: ({q_out[0]:.4f}, {q_out[1]:.4f}, {q_out[2]:.4f}, {q_out[3]:.4f})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
