#!/usr/bin/env python3
"""
Export a human-readable trajectory summary from an HDF5 episode.

Usage:
  python tools/export/export_readable_traj.py \
      --input data/vla_dataset/episode_0000.hdf5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export readable trajectory from VLA HDF5 episode")
    p.add_argument("--input", type=str, required=True, help="Input episode .hdf5")
    p.add_argument("--out_csv", type=str, default=None, help="Output CSV path")
    p.add_argument("--out_png", type=str, default=None, help="Output PNG path")
    return p.parse_args()


def default_outputs(input_path: Path, out_csv: str | None, out_png: str | None) -> tuple[Path, Path]:
    stem = input_path.with_suffix("")
    csv_path = Path(out_csv) if out_csv else Path(f"{stem}_traj_readable.csv")
    png_path = Path(out_png) if out_png else Path(f"{stem}_traj_xy.png")
    return csv_path, png_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_csv, out_png = default_outputs(input_path, args.out_csv, args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f:
        states = f["observations/state"][:]  # (T, 20)
        actions = f["actions"][:]  # (T, 7)
        t = f["timestamps"][:]  # (T,)
        dt = float(f.attrs.get("dt", 0.1))

    # State layout in env:
    # [0:6] jpos, [6:12] jvel, [12:14] gripper_pos, [14:17] ee_pos, [17:20] cup_pos
    ee = states[:, 14:17]
    cup = states[:, 17:20]
    dist_ee_cup = np.linalg.norm(ee - cup, axis=1)
    action_norm = np.linalg.norm(actions[:, :6], axis=1)

    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "step",
                "time_s",
                "ee_x",
                "ee_y",
                "ee_z",
                "cup_x",
                "cup_y",
                "cup_z",
                "ee_to_cup_dist",
                "action_dx",
                "action_dy",
                "action_dz",
                "action_droll",
                "action_dpitch",
                "action_dyaw",
                "gripper_cmd",
                "action_norm_6d",
            ]
        )
        for i in range(len(t)):
            writer.writerow(
                [
                    i,
                    float(t[i]),
                    float(ee[i, 0]),
                    float(ee[i, 1]),
                    float(ee[i, 2]),
                    float(cup[i, 0]),
                    float(cup[i, 1]),
                    float(cup[i, 2]),
                    float(dist_ee_cup[i]),
                    float(actions[i, 0]),
                    float(actions[i, 1]),
                    float(actions[i, 2]),
                    float(actions[i, 3]),
                    float(actions[i, 4]),
                    float(actions[i, 5]),
                    float(actions[i, 6]),
                    float(action_norm[i]),
                ]
            )

    # Optional XY trajectory plot for quick visual understanding.
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        plt.plot(ee[:, 0], ee[:, 1], label="EE XY trajectory", linewidth=1.6)
        plt.scatter([ee[0, 0]], [ee[0, 1]], c="green", s=60, label="EE start")
        plt.scatter([ee[-1, 0]], [ee[-1, 1]], c="red", s=60, label="EE end")
        plt.scatter([cup[0, 0]], [cup[0, 1]], c="blue", s=60, label="Cup")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Readable XY Trajectory ({input_path.name})")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        print(f"[OK] Wrote plot: {out_png}")
    except Exception as exc:
        print(f"[WARN] Plot skipped: {exc}")

    duration = len(t) * dt
    print(f"[OK] Wrote CSV: {out_csv}")
    print(f"[INFO] Steps={len(t)}, dt={dt:.4f}s, approx_duration={duration:.2f}s")
    print(
        "[INFO] Dist(EE,Cup): "
        f"min={dist_ee_cup.min():.4f}, mean={dist_ee_cup.mean():.4f}, max={dist_ee_cup.max():.4f}"
    )


if __name__ == "__main__":
    main()

