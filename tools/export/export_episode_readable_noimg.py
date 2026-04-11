#!/usr/bin/env python3
"""
Export episode to a readable folder: RGB images, depth visualization images, actions JSON.
Does not dump full RGB/depth arrays as JSON (keeps output small).

Usage:
  python tools/export/export_episode_readable_noimg.py \\
      --input data/vla_dataset_real/episode_0000.hdf5 \\
      --out_dir data/vla_dataset_real/readable_noimg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RGB + depth vis + actions JSON (no array JSON dumps)")
    p.add_argument("--input", type=str, required=True, help="Input .hdf5 episode path")
    p.add_argument("--out_dir", type=str, default=None, help="Output folder (default: <parent>/readable_noimg)")
    p.add_argument(
        "--depth_clip_mm",
        type=float,
        default=5000.0,
        help="Depth visualization clip in mm (Orbbec-style uint16/mm float); values above clip map to white",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def depth_hw_to_u8_vis(depth_hw: np.ndarray, clip_mm: float) -> np.ndarray:
    """Single-channel uint8 depth visualization (grayscale)."""
    d = np.asarray(depth_hw, dtype=np.float32).squeeze()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    valid = d > 0
    if not np.any(valid):
        return np.zeros(d.shape, dtype=np.uint8)
    d_clip = np.clip(d, 0.0, clip_mm)
    d_norm = (d_clip / max(clip_mm, 1e-6) * 255.0).astype(np.uint8)
    return d_norm


def save_image(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    a = np.asarray(arr)
    if a.ndim == 2:
        Image.fromarray(a, mode="L").save(path)
    else:
        Image.fromarray(a).save(path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = input_path.parent / "readable_noimg"

    p_rgb = out_dir / "images" / "rgb"
    p_rgb_raw = out_dir / "images" / "rgb_raw"
    p_depth = out_dir / "images" / "depth_vis"
    ensure_dir(p_rgb)
    ensure_dir(p_depth)

    with h5py.File(input_path, "r") as f:
        attrs = {k: (v.tolist() if hasattr(v, "tolist") else (v.decode("utf-8") if isinstance(v, bytes) else v)) for k, v in f.attrs.items()}
        actions = f["actions"][:]
        timestamps = f["timestamps"][:]
        rgb = f["observations/images/rgb"][:]
        depth = f["observations/images/depth"][:]
        has_rgb_raw = "rgb_raw" in f["observations/images"]
        rgb_raw = f["observations/images/rgb_raw"][:] if has_rgb_raw else None

    T = int(actions.shape[0])
    action_names = attrs.get("action_names", ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"])
    if isinstance(action_names, np.ndarray):
        action_names = action_names.tolist()
    action_names = [str(x) for x in action_names]

    actions_out: list[dict[str, float]] = []
    for i in range(T):
        row = {name: float(actions[i, j]) for j, name in enumerate(action_names) if j < actions.shape[1]}
        row["step"] = i
        row["timestamp"] = float(timestamps[i])
        actions_out.append(row)

    meta = {
        "source_file": str(input_path.resolve()),
        "num_steps": T,
        "attrs": attrs,
        "shapes": {
            "actions": list(actions.shape),
            "timestamps": list(timestamps.shape),
            "rgb": list(rgb.shape),
            "depth": list(depth.shape),
            "rgb_raw": list(rgb_raw.shape) if rgb_raw is not None else None,
        },
    }

    ensure_dir(out_dir)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    with (out_dir / "actions.json").open("w", encoding="utf-8") as fp:
        json.dump(actions_out, fp, ensure_ascii=False, indent=2)

    if has_rgb_raw and rgb_raw is not None:
        ensure_dir(p_rgb_raw)

    for i in range(T):
        name = f"frame_{i:06d}.png"
        save_image(p_rgb / name, rgb[i])
        if has_rgb_raw and rgb_raw is not None:
            save_image(p_rgb_raw / name, rgb_raw[i])
        d = depth[i]
        if d.ndim == 3:
            d = d[:, :, 0]
        save_image(p_depth / name, depth_hw_to_u8_vis(d, args.depth_clip_mm))

    print(f"[OK] Exported to {out_dir.resolve()}")
    print(f"[INFO] steps={T}, rgb={p_rgb}, depth_vis={p_depth}, actions={out_dir / 'actions.json'}")
    if has_rgb_raw:
        print(f"[INFO] rgb_raw={p_rgb_raw}")


if __name__ == "__main__":
    main()
