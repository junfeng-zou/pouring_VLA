#!/usr/bin/env python3
"""
VLA Dataset Writer — OpenVLA-7B Compatible
============================================

Collects per-step data during teleoperation and saves each episode
as an HDF5 file for fine-tuning OpenVLA-7B.

HDF5 layout
------------
episode_XXXX.hdf5
├── observations/
│   ├── images/
│   │   ├── rgb          (T, H, W, 3)  uint8    — main camera
│   │   ├── depth        (T, H, W, 1)  float32  — main camera depth
│   │   └── rgb_side     (T, H, W, 3)  uint8    — side camera (optional)
│   └── state            (T, D_state)  float32
├── actions               (T, 7)       float32   — [dx,dy,dz,droll,dpitch,dyaw,gripper]
├── actions_raw           (T, 7)       float32   — physical delta action (optional)
├── timestamps            (T,)         float64
└── attrs
    ├── num_steps         int
    ├── dt                float
    ├── task              str          — language instruction for VLA
    ├── action_names      list[str]
    ├── action_range      str
    └── model_target      str          — "OpenVLA-7B"
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import h5py
import numpy as np


class EpisodeRecorder:
    """Buffer per-step data and flush to HDF5 when an episode ends."""

    def __init__(self, image_size: int = 224, task_description: str = "pour water from bottle to cup"):
        self.image_size = image_size
        self.task_description = task_description
        self.reset()

    # ---- Public API ----------------------------------------------------------

    def reset(self):
        """Clear all buffers to start a new episode."""
        self._rgb: list[np.ndarray] = []
        self._depth: list[np.ndarray] = []
        self._rgb_side: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._action_raw: list[np.ndarray] = []
        self._timestamps: list[float] = []

    @property
    def num_steps(self) -> int:
        return len(self._timestamps)

    def add_step(
        self,
        rgb: np.ndarray,                        # (H, W, 3) uint8
        depth: np.ndarray,                      # (H, W) or (H, W, 1) float32
        state: np.ndarray,                      # (D_state,) float32
        action: np.ndarray,                     # (D_action,) float32
        action_raw: Optional[np.ndarray] = None,  # (D_action,) float32, optional physical action
        timestamp: Optional[float] = None,
        rgb_side: Optional[np.ndarray] = None,  # (H, W, 3) uint8, optional side camera
    ):
        """Add one time-step of data to the episode buffer."""
        rgb_resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        self._rgb.append(rgb_resized)

        if depth.ndim == 2:
            depth = depth[:, :, None]
        depth_resized = cv2.resize(
            depth.squeeze(-1), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        self._depth.append(depth_resized[:, :, None])

        if rgb_side is not None:
            side_resized = cv2.resize(rgb_side, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            self._rgb_side.append(side_resized)

        self._state.append(state.astype(np.float32))
        self._action.append(action.astype(np.float32))
        if action_raw is not None:
            self._action_raw.append(action_raw.astype(np.float32))
        self._timestamps.append(timestamp if timestamp is not None else time.time())

    def save(self, filepath: str, dt: float = 0.1) -> str:
        """
        Write the buffered episode to an HDF5 file.

        Args:
            filepath: Full path to the output .hdf5 file.
            dt: Simulation time-step between recorded frames.

        Returns:
            The filepath that was written.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with h5py.File(filepath, "w") as f:
            T = self.num_steps

            obs_grp = f.create_group("observations")
            img_grp = obs_grp.create_group("images")
            img_grp.create_dataset(
                "rgb",
                data=np.stack(self._rgb, axis=0),
                compression="gzip",
                compression_opts=4,
            )
            img_grp.create_dataset(
                "depth",
                data=np.stack(self._depth, axis=0).astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )

            if self._rgb_side and len(self._rgb_side) == T:
                img_grp.create_dataset(
                    "rgb_side",
                    data=np.stack(self._rgb_side, axis=0),
                    compression="gzip",
                    compression_opts=4,
                )

            obs_grp.create_dataset(
                "state",
                data=np.stack(self._state, axis=0),
                compression="gzip",
            )

            f.create_dataset(
                "actions",
                data=np.stack(self._action, axis=0),
                compression="gzip",
            )
            if self._action_raw and len(self._action_raw) == T:
                f.create_dataset(
                    "actions_raw",
                    data=np.stack(self._action_raw, axis=0),
                    compression="gzip",
                )

            f.create_dataset("timestamps", data=np.array(self._timestamps, dtype=np.float64))

            # -- metadata (OpenVLA-7B compatible) --
            f.attrs["num_steps"] = T
            f.attrs["dt"] = dt
            f.attrs["task"] = self.task_description
            f.attrs["image_size"] = self.image_size
            f.attrs["model_target"] = "OpenVLA-7B"
            f.attrs["action_dim"] = 7
            f.attrs["action_names"] = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
            f.attrs["action_range"] = "[-1, 1]"
            f.attrs["robot"] = "DOBOT CR5"

        return filepath


def next_episode_path(save_dir: str) -> str:
    """Return the path for the next episode file, e.g. episode_0003.hdf5.

    Uses max existing index + 1 to avoid overwriting after deletions.
    """
    os.makedirs(save_dir, exist_ok=True)
    max_idx = -1
    for f in os.listdir(save_dir):
        if f.startswith("episode_") and f.endswith(".hdf5"):
            try:
                idx = int(f[len("episode_"):-len(".hdf5")])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return os.path.join(save_dir, f"episode_{max_idx + 1:04d}.hdf5")
