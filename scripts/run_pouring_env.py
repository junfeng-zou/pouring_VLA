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

            # Generate sinusoidal actions for demo
            actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
            # for j in range(cfg.action_space):
            #     phase = j * (2.0 * math.pi / cfg.action_space)
            #     actions[:, j] = 0.5 * math.sin(0.4 * math.pi * t + phase)

            # Step
            obs, reward, terminated, truncated, info = env.step(actions)
            t += env.dt

            # Print status periodically
            if step % 100 == 0:
                jpos = obs["policy"][0, :6]
                ee_pos = obs["policy"][0, 12:15]
                water = info.get("log", {}).get("water_in_cup", 0.0)
                print(
                    f"  [step={step:5d}, t={t:6.2f}s]"
                    f"  reward={reward[0].item():+.4f}"
                    f"  water_in_cup={water:.1f}"
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
