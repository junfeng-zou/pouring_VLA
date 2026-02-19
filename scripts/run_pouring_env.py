"""
DOBOT Nova 5 — Pouring Environment Test Script
=================================================

Launch with:
    /path/to/IsaacLab/isaaclab.sh -p scripts/run_pouring_env.py

    # headless mode:
    /path/to/IsaacLab/isaaclab.sh -p scripts/run_pouring_env.py --headless

This script creates the PouringEnv, runs a simple sinusoidal motion
on all joints, and prints observations / rewards at each step.
"""

from __future__ import annotations

import argparse
import math
import sys
import os
import torch

# -- Make project root importable --
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser(description="DOBOT Nova 5 Pouring Environment Test")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000, help="Max sim steps (0 = infinite)")
    args = parser.parse_args()

    # Launch Sim
    sim_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

    # Now we can import Isaac Lab modules
    from envs.pouring_env import PouringEnv, PouringEnvCfg

    # ------ Configure ------
    cfg = PouringEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # ------ Create env ------
    env = PouringEnv(cfg)
    print("\n" + "=" * 60)
    print("DOBOT Nova 5 — Pouring Environment Test")
    print("=" * 60)
    print(f"  Num environments : {env.num_envs}")
    print(f"  Action space     : {cfg.action_space}")
    print(f"  Observation space: {cfg.observation_space}")
    print(f"  Episode length   : {cfg.episode_length_s} s")
    print(f"  Num water spheres: {len(env._water_spheres)}")
    print("=" * 60 + "\n")

    # ------ Reset ------
    obs, info = env.reset()
    print(f"[Reset] Obs shape: {obs['policy'].shape}")

    # ------ Run loop ------
    step = 0
    t = 0.0
    try:
        while True:
            if 0 < args.max_steps <= step:
                break

            # Generate actions: arm stays still, gripper goes from open → closed
            actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
            # Gripper: +1 = open, -1 = closed.  Linearly close over 10 seconds.
            gripper_val = 1.0 - 2.0 * min(t / 10.0, 1.0)  # +1 → -1
            actions[:, 6] = gripper_val

            # Step
            obs, reward, terminated, truncated, info = env.step(actions)
            t += env.dt

            # Print status periodically
            if step % 100 == 0:
                ee_pos = obs["policy"][0, 14:17]
                grip_pos = obs["policy"][0, 12:14]  # finger positions
                water = info.get("log", {}).get("water_in_cup", 0.0)
                print(
                    f"  [step={step:5d}, t={t:6.2f}s]"
                    f"  gripper_cmd={gripper_val:+.2f}"
                    f"  finger_pos=[{grip_pos[0]:.4f}, {grip_pos[1]:.4f}]"
                    f"  ee_pos=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
                )

            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # ------ Cleanup ------
    env.close()
    sim_app.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
