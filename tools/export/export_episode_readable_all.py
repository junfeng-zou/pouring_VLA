#!/usr/bin/env python3
"""
Export all episode contents into a human-readable folder.

Outputs include:
- metadata json
- per-step json table
- full arrays (.json)
- RGB / side RGB frames (.jpg)
- depth raw (.json) and depth visualization (.jpg)

Usage:
  python tools/export/export_episode_readable_all.py \
      --input data/vla_dataset/episode_0000.hdf5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export all episode data to readable files")
    p.add_argument("--input", type=str, required=True, help="Input .hdf5 episode path")
    p.add_argument("--out_dir", type=str, default=None, help="Output folder")
    p.add_argument("--depth_clip_max", type=float, default=2.0, help="Max depth for visualization (meters)")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def depth_to_vis(depth_hw: np.ndarray, clip_max: float) -> np.ndarray:
    d = np.nan_to_num(depth_hw.astype(np.float32), nan=clip_max, posinf=clip_max, neginf=0.0)
    d = np.clip(d, 0.0, clip_max)
    d_norm = (d / max(clip_max, 1e-6) * 255.0).astype(np.uint8)
    return d_norm


def save_png(path: Path, img_hwc: np.ndarray) -> None:
    from PIL import Image

    arr = np.asarray(img_hwc)
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
        return
    # Input arrays are RGB in the dataset; PIL expects RGB ordering directly.
    Image.fromarray(arr).save(path)


def save_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else input_path.with_suffix("")
    out_dir = out_dir.parent / f"{out_dir.name}_readable_all"

    # Folder layout
    p_rgb = out_dir / "images" / "rgb"
    p_side = out_dir / "images" / "rgb_side"
    p_depth_vis = out_dir / "images" / "depth_vis"
    p_depth_raw = out_dir / "images" / "depth_raw_json"
    p_arrays = out_dir / "arrays_json"
    p_tables = out_dir / "tables"
    ensure_dir(p_rgb)
    ensure_dir(p_side)
    ensure_dir(p_depth_vis)
    ensure_dir(p_depth_raw)
    ensure_dir(p_arrays)
    ensure_dir(p_tables)

    with h5py.File(input_path, "r") as f:
        attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
        actions = f["actions"][:]
        has_actions_raw = "actions_raw" in f
        actions_raw = f["actions_raw"][:] if has_actions_raw else None
        states = f["observations/state"][:]
        timestamps = f["timestamps"][:]
        rgb = f["observations/images/rgb"][:]
        depth = f["observations/images/depth"][:]  # (T,H,W,1)
        has_side = "rgb_side" in f["observations/images"]
        rgb_side = f["observations/images/rgb_side"][:] if has_side else None

    T = actions.shape[0]
    # Save arrays as JSON for human readability (large files expected).
    save_json(p_arrays / "actions.json", actions.tolist())
    if has_actions_raw:
        save_json(p_arrays / "actions_raw.json", actions_raw.tolist())
    save_json(p_arrays / "state.json", states.tolist())
    save_json(p_arrays / "timestamps.json", timestamps.tolist())
    save_json(p_arrays / "rgb.json", rgb.tolist())
    save_json(p_arrays / "depth.json", depth.tolist())
    if has_side:
        save_json(p_arrays / "rgb_side.json", rgb_side.tolist())

    # Save metadata
    meta = {
        "source_file": str(input_path),
        "num_steps": int(T),
        "attrs": attrs,
        "has_rgb_side": bool(has_side),
        "shapes": {
            "actions": list(actions.shape),
            "actions_raw": list(actions_raw.shape) if has_actions_raw else None,
            "state": list(states.shape),
            "timestamps": list(timestamps.shape),
            "rgb": list(rgb.shape),
            "depth": list(depth.shape),
            "rgb_side": list(rgb_side.shape) if has_side else None,
        },
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Save per-step table (CSV + JSON)
    ee = states[:, 14:17]
    cup = states[:, 17:20]
    dist_ee_cup = np.linalg.norm(ee - cup, axis=1)
    action_norm = np.linalg.norm(actions[:, :6], axis=1)
    with (p_tables / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
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
                "dx",
                "dy",
                "dz",
                "droll",
                "dpitch",
                "dyaw",
                "gripper",
                "action_norm_6d",
            ]
        )
        for i in range(T):
            w.writerow(
                [
                    i,
                    float(timestamps[i]),
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
    steps_json: list[dict[str, float | int]] = []
    for i in range(T):
        steps_json.append(
            {
                "step": i,
                "time_s": float(timestamps[i]),
                "ee_x": float(ee[i, 0]),
                "ee_y": float(ee[i, 1]),
                "ee_z": float(ee[i, 2]),
                "cup_x": float(cup[i, 0]),
                "cup_y": float(cup[i, 1]),
                "cup_z": float(cup[i, 2]),
                "ee_to_cup_dist": float(dist_ee_cup[i]),
                "dx": float(actions[i, 0]),
                "dy": float(actions[i, 1]),
                "dz": float(actions[i, 2]),
                "droll": float(actions[i, 3]),
                "dpitch": float(actions[i, 4]),
                "dyaw": float(actions[i, 5]),
                "gripper": float(actions[i, 6]),
                "action_norm_6d": float(action_norm[i]),
            }
        )
    save_json(p_tables / "steps.json", steps_json)

    # Save per-frame images
    for i in range(T):
        name = f"frame_{i:06d}.jpg"
        save_png(p_rgb / name, rgb[i])
        if has_side:
            save_png(p_side / name, rgb_side[i])
        d = depth[i, :, :, 0]
        save_json(p_depth_raw / f"frame_{i:06d}.json", d.tolist())
        save_png(p_depth_vis / name, depth_to_vis(d, args.depth_clip_max))

    print(f"[OK] Exported readable bundle: {out_dir}")
    print(f"[INFO] Steps={T}, has_side={has_side}")
    print(f"[INFO] has_actions_raw={has_actions_raw}")
    print(f"[INFO] RGB frames: {p_rgb}")
    print(f"[INFO] Depth vis/raw: {p_depth_vis} , {p_depth_raw}")
    print(f"[INFO] Table: {p_tables / 'steps.csv'}")


if __name__ == "__main__":
    main()

