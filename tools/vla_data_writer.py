#!/usr/bin/env python3
"""
VLA Dataset Writer
==================

Collects per-step data during teleoperation and saves each episode
as an HDF5 file compatible with LeRobot / VLA training pipelines.

HDF5 layout
------------
episode_XXXX.hdf5
├── observations/
│   ├── images/
│   │   ├── rgb        (T, H, W, 3) uint8
│   │   └── depth      (T, H, W, 1) float32
│   └── state          (T, D_state) float32
├── actions             (T, D_action) float32
├── timestamps          (T,) float64
└── attrs
    ├── num_steps       int
    ├── dt              float
    └── task            str
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
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._timestamps: list[float] = []

    @property
    def num_steps(self) -> int:
        return len(self._timestamps)

    def add_step(
        self,
        rgb: np.ndarray,           # (H, W, 3) uint8
        depth: np.ndarray,         # (H, W) or (H, W, 1) float32
        state: np.ndarray,         # (D_state,) float32
        action: np.ndarray,        # (D_action,) float32
        timestamp: Optional[float] = None,
    ):
        """Add one time-step of data to the episode buffer."""
        # Resize RGB to target size
        rgb_resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        self._rgb.append(rgb_resized)

        # Depth: ensure shape (H, W, 1) and resize
        if depth.ndim == 2:
            depth = depth[:, :, None]
        depth_resized = cv2.resize(
            depth.squeeze(-1), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        self._depth.append(depth_resized[:, :, None])

        self._state.append(state.astype(np.float32))
        self._action.append(action.astype(np.float32))
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

            # -- observations/images/rgb --
            obs_grp = f.create_group("observations")
            img_grp = obs_grp.create_group("images")
            img_grp.create_dataset(
                "rgb",
                data=np.stack(self._rgb, axis=0),  # (T, H, W, 3) uint8
                compression="gzip",
                compression_opts=4,
            )
            img_grp.create_dataset(
                "depth",
                data=np.stack(self._depth, axis=0).astype(np.float32),  # (T, H, W, 1)
                compression="gzip",
                compression_opts=4,
            )

            # -- observations/state --
            obs_grp.create_dataset(
                "state",
                data=np.stack(self._state, axis=0),  # (T, D_state)
                compression="gzip",
            )

            # -- actions --
            f.create_dataset(
                "actions",
                data=np.stack(self._action, axis=0),  # (T, D_action)
                compression="gzip",
            )

            # -- timestamps --
            f.create_dataset("timestamps", data=np.array(self._timestamps, dtype=np.float64))

            # -- metadata --
            f.attrs["num_steps"] = T
            f.attrs["dt"] = dt
            f.attrs["task"] = self.task_description
            f.attrs["image_size"] = self.image_size

        return filepath


def next_episode_path(save_dir: str) -> str:
    """Return the path for the next episode file, e.g. episode_0003.hdf5."""
    os.makedirs(save_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith("episode_") and f.endswith(".hdf5")
    ]
    idx = len(existing)
    return os.path.join(save_dir, f"episode_{idx:04d}.hdf5")
